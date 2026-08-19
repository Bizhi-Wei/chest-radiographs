"""
Grad-CAM error analysis report.

This upgrades Grad-CAM from a display figure to an error-analysis audit:
  - Uses validation-selected operating threshold when available.
  - Stratifies test cases into TP, TN, FP and FN.
  - Generates contact sheets for the most confident cases in each category.
  - Quantifies whether Grad-CAM heat concentrates near image borders/corners.

Default target:
  main-analysis exact-deduplicated ResNet18.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm

from patient_group_utils import extract_patient_id_from_path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("PNEUMONIA_DATASET_DIR", BASE_DIR / "data" / "chest_xray_patient_split_dedup"))
if not DATASET_DIR.is_absolute():
    DATASET_DIR = BASE_DIR / DATASET_DIR
REPORT_DIR = BASE_DIR / "reports" / "gradcam_error_analysis"
TABLE_DIR = BASE_DIR / "reports" / "final_tables"
OUTPUT_PREFIX = os.environ.get("PNEUMONIA_EVAL_PREFIX", "dedup_")
MODEL_NAME = os.environ.get("PNEUMONIA_GRADCAM_MODEL", "ResNet18")
RESNET_CHECKPOINT = Path(os.environ.get("PNEUMONIA_RESNET_CHECKPOINT", BASE_DIR / "output_patient_split_dedup" / "best_resnet18.pth"))
if not RESNET_CHECKPOINT.is_absolute():
    RESNET_CHECKPOINT = BASE_DIR / RESNET_CHECKPOINT
THRESHOLD_SOURCE = os.environ.get("PNEUMONIA_THRESHOLD_SOURCE", "Best F1 on validation")
N_PER_CATEGORY = int(os.environ.get("PNEUMONIA_GRADCAM_N_PER_CATEGORY", "8"))

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_threshold() -> float:
    table_path = TABLE_DIR / f"{OUTPUT_PREFIX}test_performance_with_validation_thresholds.csv"
    if not table_path.exists():
        return 0.5
    df = pd.read_csv(table_path)
    rows = df[(df["Model"] == MODEL_NAME) & (df["Threshold source"] == THRESHOLD_SOURCE)]
    if rows.empty:
        return 0.5
    return float(rows.iloc[0]["Threshold"])


def build_resnet18(checkpoint: Path) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return model.to(DEVICE).eval()


def make_dataset_and_loader():
    resnet_module = import_module(BASE_DIR / "train_resnet18_fixed.py", "train_resnet18_fixed")
    dataset = datasets.ImageFolder(str(DATASET_DIR / "test"), transform=resnet_module.get_transforms(False))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return dataset, loader


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(DEVICE)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
    return np.asarray(labels_all, dtype=int), np.asarray(probs_all, dtype=float)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)
        score = output[:, target_class].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= float(cam.min())
        cam /= float(cam.max()) + 1e-8
        return cam


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_cam(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.55 * image + 0.45 * heatmap, 0, 1)


def cam_audit_metrics(cam: np.ndarray, border_margin: float = 0.12, corner_margin: float = 0.18) -> dict:
    heat = cam.astype(float)
    total_heat = float(heat.sum()) + 1e-8
    h, w = heat.shape
    by = max(1, int(round(h * border_margin)))
    bx = max(1, int(round(w * border_margin)))
    cy = max(1, int(round(h * corner_margin)))
    cx = max(1, int(round(w * corner_margin)))
    border_mask = np.zeros_like(heat, dtype=bool)
    border_mask[:by, :] = True
    border_mask[-by:, :] = True
    border_mask[:, :bx] = True
    border_mask[:, -bx:] = True
    corner_mask = np.zeros_like(heat, dtype=bool)
    corner_mask[:cy, :cx] = True
    corner_mask[:cy, -cx:] = True
    corner_mask[-cy:, :cx] = True
    corner_mask[-cy:, -cx:] = True

    y_grid, x_grid = np.mgrid[0:h, 0:w]
    cx_heat = float((heat * x_grid).sum() / total_heat / max(w - 1, 1))
    cy_heat = float((heat * y_grid).sum() / total_heat / max(h - 1, 1))
    p = heat.ravel() / total_heat
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    top_mask = heat >= np.quantile(heat, 0.90)
    top_border_fraction = float((top_mask & border_mask).sum() / max(top_mask.sum(), 1))

    border_heat_ratio = float(heat[border_mask].sum() / total_heat)
    corner_heat_ratio = float(heat[corner_mask].sum() / total_heat)
    return {
        "border_heat_ratio": border_heat_ratio,
        "corner_heat_ratio": corner_heat_ratio,
        "top10_border_fraction": top_border_fraction,
        "heat_center_x": cx_heat,
        "heat_center_y": cy_heat,
        "normalized_heat_entropy": entropy,
        "artifact_attention_flag": bool(border_heat_ratio >= 0.35 or corner_heat_ratio >= 0.20 or top_border_fraction >= 0.45),
    }


def classify_cases(labels: np.ndarray, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    preds = (probs >= threshold).astype(int)
    rows = []
    for idx, (label, pred, prob) in enumerate(zip(labels, preds, probs)):
        if label == 1 and pred == 1:
            category = "TP"
            rank_score = prob
        elif label == 0 and pred == 0:
            category = "TN"
            rank_score = 1.0 - prob
        elif label == 0 and pred == 1:
            category = "FP"
            rank_score = prob
        else:
            category = "FN"
            rank_score = 1.0 - prob
        rows.append(
            {
                "dataset_index": idx,
                "true_label": int(label),
                "predicted_label": int(pred),
                "pneumonia_probability": float(prob),
                "category": category,
                "rank_score": float(rank_score),
            }
        )
    return pd.DataFrame(rows)


def selected_cases(case_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for category in ("FP", "FN", "TP", "TN"):
        subset = case_df[case_df["category"] == category].sort_values("rank_score", ascending=False).head(N_PER_CATEGORY)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_category_sheet(rows: pd.DataFrame, dataset, images: dict[int, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    if rows.empty:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(8.5, 3.1 * n))
    if n == 1:
        axes = np.asarray([axes])
    for row_idx, (_, row) in enumerate(rows.iterrows()):
        idx = int(row["dataset_index"])
        image, overlay = images[idx]
        sample_path, _ = dataset.samples[idx]
        flag = "FLAG" if bool(row["artifact_attention_flag"]) else "ok"
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(
            f"{row['category']} true={CLASS_NAMES[int(row['true_label'])]} pred={CLASS_NAMES[int(row['predicted_label'])]} P={row['pneumonia_probability']:.3f}"
        )
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].set_title(
            f"{Path(sample_path).name}\nborder={row['border_heat_ratio']:.2f} corner={row['corner_heat_ratio']:.2f} {flag}"
        )
        axes[row_idx, 1].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = load_threshold()
    model = build_resnet18(RESNET_CHECKPOINT)
    dataset, loader = make_dataset_and_loader()
    labels, probs = predict_probs(model, loader)
    case_df = classify_cases(labels, probs, threshold)
    cm = confusion_matrix(labels, case_df["predicted_label"].values, labels=[0, 1])
    selected_df = selected_cases(case_df)

    cam = GradCAM(model, model.layer4[-1])
    image_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    audit_rows = []
    try:
        for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Grad-CAM cases"):
            idx = int(row["dataset_index"])
            image_tensor, _ = dataset[idx]
            path, _ = dataset.samples[idx]
            image_batch = image_tensor.unsqueeze(0).to(DEVICE)
            target_class = int(row["predicted_label"])
            cam_map = cam.generate(image_batch, target_class=target_class)
            image = denormalize(image_tensor)
            overlay = overlay_cam(image, cam_map)
            image_cache[idx] = (image, overlay)
            metrics = cam_audit_metrics(cam_map)
            audit_rows.append(
                {
                    **row.to_dict(),
                    "patient_id": extract_patient_id_from_path(path),
                    "filename": Path(path).name,
                    "path": str(Path(path).relative_to(DATASET_DIR)).replace("\\", "/"),
                    "gradcam_target_class": target_class,
                    **metrics,
                }
            )
    finally:
        cam.close()

    audit_df = pd.DataFrame(audit_rows)
    csv_path = REPORT_DIR / f"{OUTPUT_PREFIX}resnet18_gradcam_error_cases.csv"
    audit_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    figure_paths = {}
    for category, title in [
        ("FP", "False positives: normal images predicted as pneumonia"),
        ("FN", "False negatives: pneumonia images predicted as normal"),
        ("TP", "True positives: pneumonia images predicted as pneumonia"),
        ("TN", "True negatives: normal images predicted as normal"),
    ]:
        rows = audit_df[audit_df["category"] == category]
        figure_path = REPORT_DIR / f"{OUTPUT_PREFIX}resnet18_gradcam_{category.lower()}_top_cases.png"
        plot_category_sheet(rows, dataset, image_cache, figure_path, title)
        if figure_path.exists():
            figure_paths[category] = figure_path

    category_counts = case_df["category"].value_counts().to_dict()
    flag_counts = audit_df.groupby("category")["artifact_attention_flag"].sum().to_dict() if not audit_df.empty else {}
    lines = [
        "# Grad-CAM error analysis",
        "",
        f"Dataset: `{DATASET_DIR}`",
        f"Model: `{MODEL_NAME}`",
        f"Checkpoint: `{RESNET_CHECKPOINT.relative_to(BASE_DIR)}`",
        f"Operating threshold: {threshold:.3f} ({THRESHOLD_SOURCE})",
        "",
        "## Confusion matrix at the analyzed operating point",
        "",
        f"- TN: {int(cm[0, 0])}",
        f"- FP: {int(cm[0, 1])}",
        f"- FN: {int(cm[1, 0])}",
        f"- TP: {int(cm[1, 1])}",
        "",
        "## Audit design",
        "",
        "For each category, the most confident cases were selected: high pneumonia probability for TP/FP and low pneumonia probability for TN/FN. Grad-CAM was generated for the predicted class, because the goal is to inspect the evidence that drove the model decision.",
        "",
        "Artifact-risk flags are heuristic screeners, not final judgments. A case is flagged when Grad-CAM heat is unusually concentrated near image borders or corners. Flagged cases should be manually reviewed for edge artifacts, labels, markers, cropping, diaphragm borders or non-lung cues.",
        "",
        "## Selected case counts",
        "",
    ]
    for category in ("FP", "FN", "TP", "TN"):
        lines.append(
            f"- {category}: total={category_counts.get(category, 0)}, reviewed={int((audit_df['category'] == category).sum()) if not audit_df.empty else 0}, artifact-risk flags={int(flag_counts.get(category, 0))}"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Case-level audit table: `{csv_path.relative_to(BASE_DIR)}`",
        ]
    )
    for category, figure_path in figure_paths.items():
        lines.append(f"- {category} contact sheet: `{figure_path.relative_to(BASE_DIR)}`")
    lines.extend(
        [
            "",
            "## Suggested manuscript wording",
            "",
            "Grad-CAM was used as an error-analysis tool rather than only as a representative visualization. False-positive, false-negative, true-positive and true-negative test cases were stratified at the validation-selected operating threshold, and the most confident cases in each category were reviewed. Heatmap concentration near image borders and corners was quantified to screen for possible attention to non-pulmonary artifacts, labels, markers or cropping effects. These qualitative maps were interpreted as hypothesis-generating model-audit evidence rather than proof of causal localization.",
        ]
    )
    summary_path = REPORT_DIR / f"{OUTPUT_PREFIX}resnet18_gradcam_error_analysis_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved audit table: {csv_path}")
    print(f"Saved summary: {summary_path}")
    for path in figure_paths.values():
        print(f"Saved figure: {path}")
    print(case_df["category"].value_counts().to_string())


if __name__ == "__main__":
    run_analysis()
