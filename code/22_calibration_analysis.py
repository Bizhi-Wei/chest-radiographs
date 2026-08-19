"""
Calibration analysis for the pneumonia X-ray models.

Rules:
  - Fit calibration parameters on the validation set only.
  - Apply the fixed calibration to the held-out test set.
  - Report discrimination and calibration separately.

Outputs:
  reports/final_tables/<prefix>calibration_metrics.csv
  reports/final_tables/<prefix>calibration_bins.csv
  reports/figures_calibration/<prefix>reliability_curves.png
  reports/<prefix>calibration_analysis_summary.md
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm

from patient_group_utils import group_ids_from_imagefolder


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("PNEUMONIA_DATASET_DIR", BASE_DIR / "data" / "chest_xray_patient_split_dedup"))
if not DATASET_DIR.is_absolute():
    DATASET_DIR = BASE_DIR / DATASET_DIR
REPORT_DIR = BASE_DIR / "reports"
TABLE_DIR = REPORT_DIR / "final_tables"
FIGURE_DIR = REPORT_DIR / "figures_calibration"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = os.environ.get("PNEUMONIA_EVAL_PREFIX", "dedup_")
RUN_MOBILE = os.environ.get("PNEUMONIA_RUN_MOBILE", "1") == "1"
RESNET_CHECKPOINT = Path(os.environ.get("PNEUMONIA_RESNET_CHECKPOINT", BASE_DIR / "output_patient_split_dedup" / "best_resnet18.pth"))
if not RESNET_CHECKPOINT.is_absolute():
    RESNET_CHECKPOINT = BASE_DIR / RESNET_CHECKPOINT
MOBILE_CHECKPOINT = Path(
    os.environ.get(
        "PNEUMONIA_MOBILE_CHECKPOINT",
        BASE_DIR / "output_patient_split_dedup_baselines" / "mobilenet_v3" / "best_mobilenet_v3.pth",
    )
)
if not MOBILE_CHECKPOINT.is_absolute():
    MOBILE_CHECKPOINT = BASE_DIR / MOBILE_CHECKPOINT

BATCH_SIZE = int(os.environ.get("PNEUMONIA_BATCH_SIZE", "32"))
NUM_CLASSES = 2
NUM_BINS = int(os.environ.get("PNEUMONIA_CALIBRATION_BINS", "10"))
SEED = 42
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
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return model.to(DEVICE).eval()


def build_mobilenet_v3(checkpoint: Path) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_all: list[int] = []
    logits_all: list[list[float]] = []
    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(DEVICE)
        logits = model(images)
        labels_all.extend(labels.numpy().tolist())
        logits_all.extend(logits.detach().cpu().numpy().tolist())
    group_ids = group_ids_from_imagefolder(loader.dataset)
    return np.asarray(labels_all, dtype=int), np.asarray(logits_all, dtype=float), group_ids


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits_t = torch.as_tensor(logits / temperature, dtype=torch.float32)
    return F.softmax(logits_t, dim=1)[:, 1].numpy()


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    def objective(log_temp: np.ndarray) -> float:
        temp = float(np.exp(log_temp[0]))
        probs = probs_from_logits(val_logits, temp)
        return float(log_loss(val_labels, probs, labels=[0, 1]))

    result = minimize(objective, x0=np.array([0.0]), method="Nelder-Mead", options={"maxiter": 200})
    temp = float(np.exp(result.x[0]))
    return max(temp, 1e-3)


def calibration_bins(labels: np.ndarray, probs: np.ndarray, num_bins: int = NUM_BINS) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    rows: list[dict] = []
    for index in range(num_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == num_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count:
            mean_pred = float(probs[mask].mean())
            frac_pos = float(labels[mask].mean())
            abs_gap = abs(mean_pred - frac_pos)
        else:
            mean_pred = np.nan
            frac_pos = np.nan
            abs_gap = np.nan
        rows.append(
            {
                "bin": index + 1,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "mean_predicted_probability": mean_pred,
                "observed_event_rate": frac_pos,
                "absolute_gap": abs_gap,
            }
        )
    return pd.DataFrame(rows)


def ece_mce(labels: np.ndarray, probs: np.ndarray, num_bins: int = NUM_BINS) -> tuple[float, float]:
    bins = calibration_bins(labels, probs, num_bins)
    nonempty = bins[bins["count"] > 0].copy()
    total = float(len(labels))
    ece = float(((nonempty["count"] / total) * nonempty["absolute_gap"]).sum())
    mce = float(nonempty["absolute_gap"].max()) if len(nonempty) else 0.0
    return ece, mce


def logit_clip(probs: np.ndarray) -> np.ndarray:
    eps = 1e-6
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def fit_calibration_intercept_slope(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    x = logit_clip(probs)

    def objective(params: np.ndarray) -> float:
        intercept, slope = params
        z = intercept + slope * x
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        return float(log_loss(labels, p, labels=[0, 1]))

    result = minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
    return float(result.x[0]), float(result.x[1])


def group_bootstrap_calibration_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    group_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    rng = np.random.RandomState(seed)
    unique_groups = np.asarray(sorted(set(group_ids.tolist())), dtype=object)
    group_to_indices = {group: np.flatnonzero(group_ids == group) for group in unique_groups}
    values = {"brier": [], "nll": [], "ece": []}
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_indices[group] for group in sampled_groups])
        sampled_labels = labels[idx]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_probs = probs[idx]
        ece, _ = ece_mce(sampled_labels, sampled_probs)
        values["brier"].append(float(brier_score_loss(sampled_labels, sampled_probs)))
        values["nll"].append(float(log_loss(sampled_labels, sampled_probs, labels=[0, 1])))
        values["ece"].append(float(ece))

    def pct(items: list[float]) -> tuple[float, float]:
        return (float(np.percentile(items, 2.5)), float(np.percentile(items, 97.5)))

    return {
        "Brier 95% CI": pct(values["brier"]),
        "NLL 95% CI": pct(values["nll"]),
        "ECE 95% CI": pct(values["ece"]),
        "Calibration CI method": "patient_group_bootstrap",
        "Calibration bootstrap groups": int(len(unique_groups)),
        "Calibration bootstrap samples": int(len(values["brier"])),
    }


def calibration_metrics(labels: np.ndarray, probs: np.ndarray, group_ids: np.ndarray) -> dict:
    ece, mce = ece_mce(labels, probs)
    intercept, slope = fit_calibration_intercept_slope(labels, probs)
    row = {
        "ROC AUC": float(roc_auc_score(labels, probs)),
        "PR AUC": float(average_precision_score(labels, probs)),
        "Brier score": float(brier_score_loss(labels, probs)),
        "NLL": float(log_loss(labels, probs, labels=[0, 1])),
        "ECE": float(ece),
        "MCE": float(mce),
        "Calibration intercept": float(intercept),
        "Calibration slope": float(slope),
    }
    row.update(group_bootstrap_calibration_ci(labels, probs, group_ids))
    return row


def run_model(model_name: str, model: nn.Module, val_loader: DataLoader, test_loader: DataLoader) -> tuple[list[dict], list[dict]]:
    val_labels, val_logits, _ = predict_logits(model, val_loader)
    test_labels, test_logits, test_group_ids = predict_logits(model, test_loader)
    temperature = fit_temperature(val_logits, val_labels)

    outputs: list[tuple[str, float]] = [("raw", 1.0), ("temperature_scaled", temperature)]
    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    for probability_type, temperature_value in outputs:
        probs = probs_from_logits(test_logits, temperature_value)
        metrics = calibration_metrics(test_labels, probs, test_group_ids)
        metric_rows.append(
            {
                "Model": model_name,
                "Probability type": probability_type,
                "Temperature source": "validation" if probability_type == "temperature_scaled" else "none",
                "Temperature": float(temperature_value),
                **metrics,
            }
        )
        bins = calibration_bins(test_labels, probs)
        for _, row in bins.iterrows():
            bin_rows.append(
                {
                    "Model": model_name,
                    "Probability type": probability_type,
                    "Temperature": float(temperature_value),
                    **row.to_dict(),
                }
            )
    return metric_rows, bin_rows


def plot_reliability(bin_df: pd.DataFrame, path: Path) -> None:
    models = list(bin_df["Model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model_name in zip(axes[0], models):
        subset = bin_df[(bin_df["Model"] == model_name) & (bin_df["count"] > 0)]
        for probability_type, group in subset.groupby("Probability type"):
            ax.plot(
                group["mean_predicted_probability"],
                group["observed_event_rate"],
                marker="o",
                linewidth=2,
                label=probability_type.replace("_", " "),
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="perfect calibration")
        ax.set_title(model_name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed pneumonia rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt(x: float) -> str:
    return f"{float(x):.4f}"


def main() -> None:
    resnet_module = import_module(BASE_DIR / "train_resnet18_fixed.py", "train_resnet18_fixed")
    baseline_module = import_module(BASE_DIR / "14_train_patient_split_baseline.py", "baseline")
    val_loader_resnet = make_loader("val", resnet_module.get_transforms(False))
    test_loader_resnet = make_loader("test", resnet_module.get_transforms(False))
    val_loader_mobile = make_loader("val", baseline_module.get_transforms(False))
    test_loader_mobile = make_loader("test", baseline_module.get_transforms(False))

    runs = [("ResNet18", build_resnet18(RESNET_CHECKPOINT), val_loader_resnet, test_loader_resnet)]
    if RUN_MOBILE and MOBILE_CHECKPOINT.exists():
        runs.append(("MobileNetV3-Small", build_mobilenet_v3(MOBILE_CHECKPOINT), val_loader_mobile, test_loader_mobile))

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_DIR}")
    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    for model_name, model, val_loader, test_loader in runs:
        print(f"\nCalibration analysis: {model_name}")
        rows, bins = run_model(model_name, model, val_loader, test_loader)
        metric_rows.extend(rows)
        bin_rows.extend(bins)

    metrics_df = pd.DataFrame(metric_rows)
    bins_df = pd.DataFrame(bin_rows)
    metrics_path = TABLE_DIR / f"{OUTPUT_PREFIX}calibration_metrics.csv"
    bins_path = TABLE_DIR / f"{OUTPUT_PREFIX}calibration_bins.csv"
    figure_path = FIGURE_DIR / f"{OUTPUT_PREFIX}reliability_curves.png"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    bins_df.to_csv(bins_path, index=False, encoding="utf-8-sig")
    plot_reliability(bins_df, figure_path)

    lines = [
        "# Calibration analysis",
        "",
        "Calibration was assessed on the held-out test set. Temperature scaling was fitted on the validation set only and then applied once to the test set.",
        "",
        "| Model | Probability type | Temperature | ROC AUC | PR AUC | Brier score | NLL | ECE | MCE | Calibration intercept | Calibration slope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['Model']} | {row['Probability type']} | {fmt(row['Temperature'])} | "
            f"{fmt(row['ROC AUC'])} | {fmt(row['PR AUC'])} | {fmt(row['Brier score'])} | "
            f"{fmt(row['NLL'])} | {fmt(row['ECE'])} | {fmt(row['MCE'])} | "
            f"{fmt(row['Calibration intercept'])} | {fmt(row['Calibration slope'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide: lower Brier score, NLL and ECE indicate better probability calibration. A calibration intercept near 0 and slope near 1 indicate well-calibrated probabilities. AUC and PR AUC are retained as discrimination metrics but should not be interpreted as calibration.",
            "",
            f"Reliability curve figure: `{figure_path.relative_to(BASE_DIR)}`",
        ]
    )
    summary_path = REPORT_DIR / f"{OUTPUT_PREFIX}calibration_analysis_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved bins: {bins_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print(metrics_df[["Model", "Probability type", "Temperature", "Brier score", "NLL", "ECE", "Calibration intercept", "Calibration slope"]].to_string(index=False))


if __name__ == "__main__":
    main()
