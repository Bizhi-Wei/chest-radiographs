"""
Strict validation-selected threshold evaluation.

Rule:
  - Select operating thresholds on the validation set only.
  - Apply the selected thresholds once to the held-out test set.
  - Do not use the test set to choose thresholds.

Outputs:
  reports/final_tables/validation_selected_thresholds.csv
  reports/final_tables/test_performance_with_validation_thresholds.csv
  reports/validation_threshold_final_eval_summary.md
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm
from patient_group_utils import group_bootstrap_ci, group_ids_from_imagefolder


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("PNEUMONIA_DATASET_DIR", BASE_DIR / "data" / "chest_xray_patient_split"))
if not DATASET_DIR.is_absolute():
    DATASET_DIR = BASE_DIR / DATASET_DIR
REPORT_DIR = BASE_DIR / "reports"
TABLE_DIR = REPORT_DIR / "final_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = os.environ.get("PNEUMONIA_EVAL_PREFIX", "")
RUN_MOBILE = os.environ.get("PNEUMONIA_RUN_MOBILE", "1") == "1"
RESNET_CHECKPOINT = Path(os.environ.get("PNEUMONIA_RESNET_CHECKPOINT", BASE_DIR / "output_patient_split" / "best_resnet18.pth"))
if not RESNET_CHECKPOINT.is_absolute():
    RESNET_CHECKPOINT = BASE_DIR / RESNET_CHECKPOINT
MOBILE_CHECKPOINT = Path(
    os.environ.get(
        "PNEUMONIA_MOBILE_CHECKPOINT",
        BASE_DIR / "output_patient_split_baselines" / "mobilenet_v3" / "best_mobilenet_v3.pth",
    )
)
if not MOBILE_CHECKPOINT.is_absolute():
    MOBILE_CHECKPOINT = BASE_DIR / MOBILE_CHECKPOINT

IMAGE_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_loader(split: str, transform) -> DataLoader:
    dataset = datasets.ImageFolder(str(DATASET_DIR / split), transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)


def build_resnet18(checkpoint: Path) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def build_mobilenet_v3(checkpoint: Path) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_all: list[int] = []
    probs_all: list[float] = []
    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(DEVICE)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1]
        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
    group_ids = group_ids_from_imagefolder(loader.dataset)
    return np.asarray(labels_all, dtype=int), np.asarray(probs_all, dtype=float), group_ids


def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float, group_ids: np.ndarray | None = None) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    row = {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(labels, preds)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "Sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "Specificity": float(spec),
        "PPV": float(precision_score(labels, preds, zero_division=0)),
        "NPV": float(npv),
        "Balanced accuracy": float((sens + spec) / 2),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if group_ids is not None:
        row.update(group_bootstrap_ci(labels, preds, probs, group_ids))
    return row


def select_on_validation(labels: np.ndarray, probs: np.ndarray) -> dict[str, dict]:
    thresholds = np.round(np.arange(0.01, 1.00, 0.005), 3)
    rows = [metrics_at_threshold(labels, probs, float(th)) for th in thresholds]
    table = pd.DataFrame(rows)
    default = table.iloc[(table["Threshold"] - 0.5).abs().argmin()].to_dict()
    best_f1 = table.sort_values(["F1", "Balanced accuracy", "Threshold"], ascending=[False, False, True]).iloc[0].to_dict()
    best_balanced = table.sort_values(["Balanced accuracy", "F1", "Threshold"], ascending=[False, False, True]).iloc[0].to_dict()
    return {
        "Default 0.5": default,
        "Best F1 on validation": best_f1,
        "Best balanced accuracy on validation": best_balanced,
    }


def summarize_auc(labels: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "ROC AUC": float(roc_auc_score(labels, probs)),
        "PR AUC": float(average_precision_score(labels, probs)),
    }


def run_model(model_name: str, model: nn.Module, val_loader: DataLoader, test_loader: DataLoader) -> tuple[list[dict], list[dict]]:
    val_labels, val_probs, val_group_ids = predict(model, val_loader)
    test_labels, test_probs, test_group_ids = predict(model, test_loader)
    val_auc = summarize_auc(val_labels, val_probs)
    test_auc = summarize_auc(test_labels, test_probs)

    selected = select_on_validation(val_labels, val_probs)
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for point_name, val_point in selected.items():
        threshold = float(val_point["Threshold"])
        val_rows.append(
            {
                "Model": model_name,
                "Selection rule": point_name,
                **val_point,
                **val_auc,
            }
        )
        test_rows.append(
            {
                "Model": model_name,
                "Threshold source": point_name,
                **metrics_at_threshold(test_labels, test_probs, threshold, test_group_ids),
                **test_auc,
            }
        )

    return val_rows, test_rows


def fmt(x: float) -> str:
    return f"{float(x):.4f}"


def main() -> None:
    resnet_module = import_module(BASE_DIR / "train_resnet18_fixed.py", "train_resnet18_fixed")
    baseline_module = import_module(BASE_DIR / "14_train_patient_split_baseline.py", "baseline")
    val_loader_resnet = make_loader("val", resnet_module.get_transforms(False))
    test_loader_resnet = make_loader("test", resnet_module.get_transforms(False))
    val_loader_mobile = make_loader("val", baseline_module.get_transforms(False))
    test_loader_mobile = make_loader("test", baseline_module.get_transforms(False))

    runs = [
        (
            "ResNet18",
            build_resnet18(RESNET_CHECKPOINT),
            val_loader_resnet,
            test_loader_resnet,
        ),
    ]
    if RUN_MOBILE and MOBILE_CHECKPOINT.exists():
        runs.append(
            (
            "MobileNetV3-Small",
            build_mobilenet_v3(MOBILE_CHECKPOINT),
            val_loader_mobile,
            test_loader_mobile,
            )
        )

    val_rows: list[dict] = []
    test_rows: list[dict] = []
    print(f"Device: {DEVICE}")
    for model_name, model, val_loader, test_loader in runs:
        print(f"\nStrict threshold evaluation: {model_name}")
        v_rows, t_rows = run_model(model_name, model, val_loader, test_loader)
        val_rows.extend(v_rows)
        test_rows.extend(t_rows)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    val_path = TABLE_DIR / f"{OUTPUT_PREFIX}validation_selected_thresholds.csv"
    test_path = TABLE_DIR / f"{OUTPUT_PREFIX}test_performance_with_validation_thresholds.csv"
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    lines = [
        "# Validation-selected threshold final evaluation",
        "",
        "Important rule: thresholds were selected on the validation set only. The test set was used only once for final evaluation at the pre-selected thresholds.",
        "",
        "## Test performance using validation-selected thresholds",
        "",
        "| Model | Threshold source | Threshold | Accuracy | F1 | Sensitivity | Specificity | ROC AUC | PR AUC | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test_df.iterrows():
        lines.append(
            f"| {row['Model']} | {row['Threshold source']} | {float(row['Threshold']):.3f} | "
            f"{fmt(row['Accuracy'])} | {fmt(row['F1'])} | {fmt(row['Sensitivity'])} | "
            f"{fmt(row['Specificity'])} | {fmt(row['ROC AUC'])} | {fmt(row['PR AUC'])} | "
            f"{int(row['TN'])} | {int(row['FP'])} | {int(row['FN'])} | {int(row['TP'])} |"
        )
    lines.extend(
        [
            "",
            "## Confidence intervals",
            "",
            "All reported confidence intervals in the output CSV are computed using patient/group-level bootstrap resampling, not image-level bootstrap resampling.",
            "",
            "## Correction note",
            "",
            "Previous threshold summaries that reported best-F1 thresholds from test-set sweeps should be treated as exploratory only and should not be used as the formal model-selection result.",
        ]
    )
    summary_path = REPORT_DIR / f"{OUTPUT_PREFIX}validation_threshold_final_eval_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved validation thresholds: {val_path}")
    print(f"Saved final test table: {test_path}")
    print(f"Saved summary: {summary_path}")
    print("\nFinal test rows:")
    print(test_df.to_string(index=False))


if __name__ == "__main__":
    main()
