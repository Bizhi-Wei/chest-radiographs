#!/usr/bin/env python3
"""Prepare submission-v2 statistics, tables, and publication figures.

This script never trains a model. It reloads the frozen clean checkpoints,
performs deterministic held-out-test inference, calculates prespecified
patient/group bootstrap intervals, and rebuilds submission figures from formal
saved results.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "01_data/chest_xray_patient_split_dedup_clean"
MANIFEST = ROOT / "04_reports/clean_rebuild_20260706/patient_split_manifest.csv"
INTERNAL_FORMAL = ROOT / "04_reports/final_tables/clean_20260706_test_performance_with_validation_thresholds.csv"
THRESHOLDS = ROOT / "04_reports/final_tables/clean_20260706_validation_selected_thresholds.csv"
CALIBRATION = ROOT / "04_reports/final_tables/clean_20260706_calibration_metrics.csv"
CALIBRATION_BINS = ROOT / "04_reports/final_tables/clean_20260706_calibration_bins.csv"
STRESS_METRICS = ROOT / "04_reports/pneumoniamnist_source_disjoint_metrics.csv"
STRESS_BOOTSTRAP = ROOT / "04_reports/pneumoniamnist_source_disjoint_bootstrap.json"
STRESS_PREDICTIONS = ROOT / "04_reports/pneumoniamnist_source_disjoint_predictions.csv"
PAIRED_SHIFT = ROOT / "04_reports/pneumoniamnist_paired_shift_analysis.csv"
OVERLAP_MAPPING = ROOT / "04_reports/pneumoniamnist_overlap_mapping.csv"
PNEUMONIAMNIST_NPZ = ROOT / "01_data/external/pneumoniamnist.npz"
REPEATED = ROOT / "03_models_and_outputs/output_clean_20260707/repeated_splits_resnet18/reports/clean_20260706_repeated_split_resnet18_per_repeat.csv"

RESNET_CHECKPOINT = ROOT / "03_models_and_outputs/output_clean_20260706/resnet18/best_resnet18_nomixup.pth"
MOBILE_CHECKPOINT = ROOT / "03_models_and_outputs/output_clean_20260706/baselines/mobilenet_v3/best_mobilenet_v3.pth"

REPORT_DIR = ROOT / "04_reports/submission_v2"
TABLE_DIR = ROOT / "04_reports/final_tables/submission_v2"
FIGURE_ROOT = ROOT / "04_reports/figures_submission_v2"
MAIN_FIGURES = FIGURE_ROOT / "main"
SUPP_FIGURES = FIGURE_ROOT / "supplementary"

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260805
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "ResNet18": {"architecture": "resnet18", "checkpoint": RESNET_CHECKPOINT},
    "MobileNetV3-Small": {"architecture": "mobilenet_v3_small", "checkpoint": MOBILE_CHECKPOINT},
}

MODEL_COLORS = {"ResNet18": "#2F6B8A", "MobileNetV3-Small": "#C47A2C"}
CLASS_COLORS = {"NORMAL": "#2F6B8A", "PNEUMONIA": "#C44E52"}
RAW_COLOR = "#6B7280"
SCALED_COLOR = "#2A9D8F"

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_model(architecture: str, checkpoint: Path) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        raise ValueError(architecture)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_internal_parameters() -> tuple[dict[str, float], dict[str, float]]:
    threshold_table = pd.read_csv(THRESHOLDS, encoding="utf-8-sig")
    selected = threshold_table[threshold_table["Selection rule"].eq("Best F1 on validation")]
    thresholds = {str(row.Model): float(row.Threshold) for row in selected.itertuples(index=False)}
    calibration_table = pd.read_csv(CALIBRATION, encoding="utf-8-sig")
    scaled = calibration_table[calibration_table["Probability type"].eq("temperature_scaled")]
    temperatures = {str(row.Model): float(row.Temperature) for row in scaled.itertuples(index=False)}
    expected = {"ResNet18": 0.450, "MobileNetV3-Small": 0.345}
    for name, value in expected.items():
        if not math.isclose(thresholds.get(name, float("nan")), value, abs_tol=1e-12):
            raise RuntimeError(f"Unexpected internal-validation threshold for {name}: {thresholds.get(name)}")
    return thresholds, temperatures


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return value


def metric_values(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece": float(expected_calibration_error(labels, probabilities)),
        "predicted_positive_proportion": float(predictions.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def group_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "roc_auc",
        "pr_auc",
        "brier",
        "nll",
        "ece",
        "predicted_positive_proportion",
    ]
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    valid_two_class = 0
    skipped_single_class = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        if np.unique(labels[sampled_indices]).size < 2:
            skipped_single_class += 1
            continue
        valid_two_class += 1
        values = metric_values(labels[sampled_indices], probabilities[sampled_indices], threshold)
        for name in metric_names:
            value = float(values[name])
            if np.isfinite(value):
                draws[name].append(value)
    intervals = {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in draws.items()
        if values
    }
    return {
        "method": "patient_group_cluster_bootstrap",
        "interval": "percentile_95",
        "resampling_unit": "inferred patient/group",
        "group_membership": "all images retained for each sampled group occurrence",
        "groups": int(len(unique_groups)),
        "requested_samples": BOOTSTRAP_SAMPLES,
        "valid_two_class_samples": int(valid_two_class),
        "skipped_single_class_samples": int(skipped_single_class),
        "valid_samples_by_metric": {name: len(values) for name, values in draws.items()},
        "seed": int(seed),
        "intervals": intervals,
    }


def infer_internal_test(
    thresholds: dict[str, float], temperatures: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dataset = datasets.ImageFolder(str(CLEAN_DATA / "test"), transform=INFERENCE_TRANSFORM)
    # Single-process loading avoids WSL/DrvFS multiprocessing handle failures on D:.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=DEVICE.type == "cuda")
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    manifest_test = manifest[manifest["split"].eq("test")].copy()
    by_name = manifest_test.set_index("destination_name").to_dict("index")
    labels = np.asarray([label for _, label in dataset.samples], dtype=int)
    names = [Path(path).name for path, _ in dataset.samples]
    patient_ids = np.asarray([str(by_name[name]["patient_id"]) for name in names], dtype=str)
    sha256s = [str(by_name[name]["sha256"]) for name in names]
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    bootstrap_output: dict[str, object] = {
        "policy": {
            "threshold_source": "internal_validation_only",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "models": {},
    }
    for model_name, config in MODEL_CONFIG.items():
        model = build_model(str(config["architecture"]), Path(config["checkpoint"]))
        logits_list: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in loader:
                logits_list.append(model(images.to(DEVICE, non_blocking=True)).cpu().numpy())
        logits = np.concatenate(logits_list, axis=0)
        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
        raw_prob = torch.softmax(logits_tensor, dim=1)[:, 1].numpy()
        scaled_prob = torch.softmax(logits_tensor / temperatures[model_name], dim=1)[:, 1].numpy()
        for index, name in enumerate(names):
            prediction_rows.append(
                {
                    "model": model_name,
                    "split": "test",
                    "destination_name": name,
                    "relative_path": str(Path(dataset.samples[index][0]).relative_to(CLEAN_DATA)).replace("\\", "/"),
                    "sha256": sha256s[index],
                    "patient_id": patient_ids[index],
                    "true_label": int(labels[index]),
                    "label_name": dataset.classes[int(labels[index])],
                    "logit_normal": float(logits[index, 0]),
                    "logit_pneumonia": float(logits[index, 1]),
                    "raw_probability": float(raw_prob[index]),
                    "temperature_scaled_probability": float(scaled_prob[index]),
                    "internal_validation_threshold": thresholds[model_name],
                    "temperature": temperatures[model_name],
                    "prediction_default_0_5": int(raw_prob[index] >= 0.5),
                    "prediction_internal_validation_threshold": int(raw_prob[index] >= thresholds[model_name]),
                }
            )
        bootstrap_output["models"][model_name] = {}
        for threshold_name, threshold in (
            ("default", 0.5),
            ("internal_validation_selected", thresholds[model_name]),
        ):
            values = metric_values(labels, raw_prob, threshold)
            boot = group_bootstrap(labels, raw_prob, patient_ids, threshold)
            row: dict[str, object] = {
                "model": model_name,
                "threshold_source": "Default" if threshold_name == "default" else "Internal validation best F1",
                "threshold": float(threshold),
                **values,
                "bootstrap_groups": boot["groups"],
                "bootstrap_requested": boot["requested_samples"],
                "bootstrap_valid": boot["valid_two_class_samples"],
                "bootstrap_seed": boot["seed"],
                "ci_method": "patient/group percentile bootstrap",
            }
            for metric, interval in boot["intervals"].items():
                row[f"{metric}_95ci_low"] = interval[0]
                row[f"{metric}_95ci_high"] = interval[1]
            metric_rows.append(row)
            bootstrap_output["models"][model_name][threshold_name] = boot
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows), bootstrap_output


def verify_internal_metrics(metrics: pd.DataFrame) -> None:
    formal = pd.read_csv(INTERNAL_FORMAL, encoding="utf-8-sig")
    mapping = {
        "Default": "Default 0.5",
        "Internal validation best F1": "Best F1 on validation",
    }
    column_map = {
        "accuracy": "Accuracy",
        "f1": "F1",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
    }
    for row in metrics.itertuples(index=False):
        source = mapping[str(row.threshold_source)]
        formal_row = formal[(formal["Model"].eq(row.model)) & (formal["Threshold source"].eq(source))]
        if len(formal_row) != 1:
            raise RuntimeError(f"Missing formal metric row for {row.model}/{source}")
        formal_row = formal_row.iloc[0]
        for generated_column, formal_column in column_map.items():
            generated = float(getattr(row, generated_column))
            expected = float(formal_row[formal_column])
            # CUDA/library-version differences can shift aggregate probabilities in
            # the final decimal places; 1e-5 remains well below reported precision.
            if not math.isclose(generated, expected, abs_tol=1e-5):
                raise RuntimeError(
                    f"Internal metric mismatch {row.model}/{source}/{generated_column}: "
                    f"generated={generated}, formal={expected}"
                )


def paired_summary() -> tuple[pd.DataFrame, dict[str, object]]:
    paired = pd.read_csv(PAIRED_SHIFT, encoding="utf-8-sig")
    paired = paired[paired["comparison"].eq("PneumoniaMNIST_28x28")].copy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    output: dict[str, object] = {
        "method": "paired patient/group percentile bootstrap",
        "requested_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "groups": {},
    }
    for model_name in MODEL_CONFIG:
        model_data = paired[paired["model"].eq(model_name)]
        for label_name in ("ALL", "NORMAL", "PNEUMONIA"):
            data = model_data if label_name == "ALL" else model_data[model_data["label_name"].eq(label_name)]
            unique_groups = data["patient_id"].astype(str).unique()
            boot = {"mean": [], "median": [], "flip": [], "crossing": []}
            for _ in range(BOOTSTRAP_SAMPLES):
                sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                sampled = pd.concat(
                    [data[data["patient_id"].astype(str).eq(group)] for group in sampled_groups],
                    ignore_index=True,
                )
                boot["mean"].append(float(sampled["probability_shift"].mean()))
                boot["median"].append(float(sampled["probability_shift"].median()))
                boot["flip"].append(float(sampled["prediction_flip"].mean()))
                boot["crossing"].append(float(sampled["threshold_crossing"].mean()))
            q1 = float(data["probability_shift"].quantile(0.25))
            q3 = float(data["probability_shift"].quantile(0.75))
            row = {
                "model": model_name,
                "label": label_name,
                "images": int(len(data)),
                "patient_groups": int(len(unique_groups)),
                "mean_probability_shift": float(data["probability_shift"].mean()),
                "mean_probability_shift_95ci_low": float(np.percentile(boot["mean"], 2.5)),
                "mean_probability_shift_95ci_high": float(np.percentile(boot["mean"], 97.5)),
                "median_probability_shift": float(data["probability_shift"].median()),
                "q1_probability_shift": q1,
                "q3_probability_shift": q3,
                "median_probability_shift_95ci_low": float(np.percentile(boot["median"], 2.5)),
                "median_probability_shift_95ci_high": float(np.percentile(boot["median"], 97.5)),
                "prediction_flips": int(data["prediction_flip"].sum()),
                "prediction_flip_rate": float(data["prediction_flip"].mean()),
                "prediction_flip_rate_95ci_low": float(np.percentile(boot["flip"], 2.5)),
                "prediction_flip_rate_95ci_high": float(np.percentile(boot["flip"], 97.5)),
                "threshold_crossings": int(data["threshold_crossing"].sum()),
                "threshold_crossing_rate": float(data["threshold_crossing"].mean()),
                "threshold_crossing_rate_95ci_low": float(np.percentile(boot["crossing"], 2.5)),
                "threshold_crossing_rate_95ci_high": float(np.percentile(boot["crossing"], 97.5)),
            }
            rows.append(row)
            output["groups"][f"{model_name}|{label_name}"] = row
    return pd.DataFrame(rows), output


def fmt_ci(value: float, low: float, high: float) -> str:
    return f"{value:.4f} ({low:.4f}-{high:.4f})"


def build_submission_tables(internal_metrics: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table2_rows = []
    for row in internal_metrics.to_dict("records"):
        table2_rows.append(
            {
                "Model": row["model"],
                "Threshold source": row["threshold_source"],
                "Threshold": row["threshold"],
                "Accuracy (95% CI)": fmt_ci(row["accuracy"], row["accuracy_95ci_low"], row["accuracy_95ci_high"]),
                "F1 (95% CI)": fmt_ci(row["f1"], row["f1_95ci_low"], row["f1_95ci_high"]),
                "Sensitivity (95% CI)": fmt_ci(row["sensitivity"], row["sensitivity_95ci_low"], row["sensitivity_95ci_high"]),
                "Specificity (95% CI)": fmt_ci(row["specificity"], row["specificity_95ci_low"], row["specificity_95ci_high"]),
                "ROC AUC (95% CI)": fmt_ci(row["roc_auc"], row["roc_auc_95ci_low"], row["roc_auc_95ci_high"]),
                "PR AUC (95% CI)": fmt_ci(row["pr_auc"], row["pr_auc_95ci_low"], row["pr_auc_95ci_high"]),
            }
        )
    pd.DataFrame(table2_rows).to_csv(TABLE_DIR / "Table_2_internal_test_performance_v2.csv", index=False, encoding="utf-8-sig")

    stress = pd.read_csv(STRESS_METRICS, encoding="utf-8-sig")
    table5_rows = []
    full_rows = []
    for row in stress.to_dict("records"):
        source = "Default" if row["threshold_name"] == "default_0.5" else "Internal validation best F1"
        table5_rows.append(
            {
                "Model": row["model"],
                "Threshold source": source,
                "Threshold": row["threshold"],
                "Accuracy (95% CI)": fmt_ci(row["accuracy"], row["accuracy_95ci_low"], row["accuracy_95ci_high"]),
                "Balanced accuracy (95% CI)": fmt_ci(row["balanced_accuracy"], row["balanced_accuracy_95ci_low"], row["balanced_accuracy_95ci_high"]),
                "Sensitivity (95% CI)": fmt_ci(row["sensitivity"], row["sensitivity_95ci_low"], row["sensitivity_95ci_high"]),
                "Specificity (95% CI)": fmt_ci(row["specificity"], row["specificity_95ci_low"], row["specificity_95ci_high"]),
                "ROC AUC (95% CI)": fmt_ci(row["roc_auc"], row["roc_auc_95ci_low"], row["roc_auc_95ci_high"]),
                "PR AUC (95% CI)": fmt_ci(row["pr_auc"], row["pr_auc_95ci_low"], row["pr_auc_95ci_high"]),
                "Brier (95% CI)": fmt_ci(row["brier_raw"], row["brier_95ci_low"], row["brier_95ci_high"]),
                "ECE (95% CI)": fmt_ci(row["ece_raw"], row["ece_95ci_low"], row["ece_95ci_high"]),
            }
        )
        full_rows.append(
            {
                "Model": row["model"],
                "Threshold source": source,
                "Threshold": row["threshold"],
                "N": row["n"],
                "NORMAL": row["normal_n"],
                "PNEUMONIA": row["pneumonia_n"],
                "Accuracy": row["accuracy"],
                "Accuracy 95% CI low": row["accuracy_95ci_low"],
                "Accuracy 95% CI high": row["accuracy_95ci_high"],
                "Balanced accuracy": row["balanced_accuracy"],
                "Balanced accuracy 95% CI low": row["balanced_accuracy_95ci_low"],
                "Balanced accuracy 95% CI high": row["balanced_accuracy_95ci_high"],
                "F1": row["f1"],
                "F1 95% CI low": row["f1_95ci_low"],
                "F1 95% CI high": row["f1_95ci_high"],
                "Sensitivity": row["sensitivity"],
                "Sensitivity 95% CI low": row["sensitivity_95ci_low"],
                "Sensitivity 95% CI high": row["sensitivity_95ci_high"],
                "Specificity": row["specificity"],
                "Specificity 95% CI low": row["specificity_95ci_low"],
                "Specificity 95% CI high": row["specificity_95ci_high"],
                "PPV": row["ppv"],
                "PPV 95% CI low": row["ppv_95ci_low"],
                "PPV 95% CI high": row["ppv_95ci_high"],
                "NPV": "undefined",
                "ROC AUC": row["roc_auc"],
                "ROC AUC 95% CI low": row["roc_auc_95ci_low"],
                "ROC AUC 95% CI high": row["roc_auc_95ci_high"],
                "PR AUC": row["pr_auc"],
                "PR AUC 95% CI low": row["pr_auc_95ci_low"],
                "PR AUC 95% CI high": row["pr_auc_95ci_high"],
                "Brier raw": row["brier_raw"],
                "NLL raw": row["nll_raw"],
                "ECE raw": row["ece_raw"],
                "Brier temperature scaled": row["brier_temperature_scaled"],
                "NLL temperature scaled": row["nll_temperature_scaled"],
                "ECE temperature scaled": row["ece_temperature_scaled"],
                "TN": row["tn"],
                "FP": row["fp"],
                "FN": row["fn"],
                "TP": row["tp"],
                "Bootstrap groups": row["bootstrap_groups"],
                "Bootstrap samples": row["bootstrap_samples"],
            }
        )
    pd.DataFrame(table5_rows).to_csv(TABLE_DIR / "Table_5_same_source_stress_test_v2.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(full_rows).to_csv(TABLE_DIR / "Supplementary_Table_S5_full_stress_metrics_v2.csv", index=False, encoding="utf-8-sig")


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def figure_1_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fill: str, edge: str = "#4B5563", size: float = 8.3) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.05", linewidth=0.9, edgecolor=edge, facecolor=fill)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.2)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#6B7280", style: str = "-") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=color, linestyle=style))

    ax.text(0.2, 6.85, "a", fontsize=12, fontweight="bold")
    ax.text(0.55, 6.85, "Data provenance and primary internal evaluation", fontsize=11, fontweight="bold")
    box(0.4, 5.5, 2.2, 0.85, "Prior local working copy\n+746 label-encoded placeholders", "#FDE8E7", "#B94A48", 7.8)
    box(0.4, 4.2, 2.2, 0.75, "Prior checkpoints, thresholds,\ncalibration and manuscript discarded", "#F6F6F6", "#6B7280", 7.8)
    arrow(1.5, 5.5, 1.5, 4.95, "#B94A48")

    box(3.05, 5.5, 2.0, 0.85, "5,856 verified real\nKermany source images", "#E8F1F7", "#2F6B8A")
    box(5.55, 5.5, 1.8, 0.85, "SHA-256 audit\n32 redundant exact\nduplicates removed", "#EEF4E8", "#648A45", 7.7)
    box(7.85, 5.5, 1.65, 0.85, "5,824 retained\nreal images", "#E8F1F7", "#2F6B8A")
    box(10.0, 5.5, 1.8, 0.85, "Patient/group split\nTrain 4,077\nVal 874 | Test 873", "#FFF3DC", "#C47A2C", 7.7)
    arrow(5.05, 5.93, 5.55, 5.93)
    arrow(7.35, 5.93, 7.85, 5.93)
    arrow(9.5, 5.93, 10.0, 5.93)

    box(0.5, 2.85, 2.1, 1.0, "Frozen-model internal test\nResNet18 and MobileNetV3-Small\nValidation-selected thresholds", "#E8F1F7", "#2F6B8A", 7.7)
    box(3.05, 2.85, 1.9, 1.0, "Five repeated\nstratified group splits\nindependent reinitialization", "#F1EAF4", "#7A5C8E", 7.7)
    box(5.4, 2.85, 1.9, 1.0, "Calibration\nValidation-fitted temperature\nGroup-bootstrap uncertainty", "#E8F3EF", "#2A9D8F", 7.7)
    box(7.75, 2.85, 1.9, 1.0, "Near-duplicate audit\nHashes, SSIM and\nImageNet embeddings", "#F6F2E8", "#9B7B3B", 7.7)
    box(10.1, 2.85, 1.6, 1.0, "Grad-CAM\nerror audit", "#F6F6F6", "#6B7280", 8.0)
    for x in (1.55, 4.0, 6.35, 8.7, 10.9):
        arrow(10.9, 5.5, x, 3.85, "#A0A0A0")

    ax.text(0.2, 2.35, "b", fontsize=12, fontweight="bold")
    ax.text(0.55, 2.35, "Post hoc PneumoniaMNIST provenance and preprocessing-shift analysis", fontsize=11, fontweight="bold")
    box(0.5, 0.65, 2.2, 1.1, "PneumoniaMNIST test\n624 transformed 28x28 images\nSame Kermany source collection", "#FDE8E7", "#B94A48", 7.8)
    box(3.25, 0.65, 2.2, 1.1, "Source-overlap audit\n549 confirmed | 1 probable\n48 ambiguous | 26 unmatched", "#FFF3DC", "#C47A2C", 7.8)
    box(6.0, 0.65, 2.3, 1.1, "Accepted mappings by\ninternal split\nTrain 366 | Val 75 | Test 109", "#F1EAF4", "#7A5C8E", 7.8)
    box(8.85, 0.65, 2.85, 1.1, "Held-out-source-matched\nsame-source stress subset\n108 images | 68 groups\n32 NORMAL | 76 PNEUMONIA\nNo tuning or recalibration", "#E8F3EF", "#2A9D8F", 7.25)
    arrow(2.7, 1.2, 3.25, 1.2)
    arrow(5.45, 1.2, 6.0, 1.2)
    arrow(8.3, 1.2, 8.85, 1.2)
    ax.text(6.1, 0.12, "The same-source stress subset evaluates preprocessing and threshold stability, not independent clinical generalization.", ha="center", fontsize=8.4, color="#7F1D1D")
    save_figure(fig, MAIN_FIGURES, "Figure_1_workflow")


def calibration_points(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    means, observed, counts = [], [], []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1])
        if mask.any():
            means.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    return np.asarray(means), np.asarray(observed), np.asarray(counts)


def figure_2_internal_performance(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    prevalence = predictions[predictions["model"].eq("ResNet18")]["true_label"].mean()
    for model_name, group in predictions.groupby("model", sort=False):
        labels = group["true_label"].to_numpy(dtype=int)
        probs = group["raw_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, probs)
        precision, recall, _ = precision_recall_curve(labels, probs)
        axes[0, 0].plot(fpr, tpr, color=MODEL_COLORS[model_name], label=f"{model_name} (AUC {roc_auc_score(labels, probs):.4f})")
        axes[0, 1].plot(recall, precision, color=MODEL_COLORS[model_name], label=f"{model_name} (AUC {average_precision_score(labels, probs):.4f})")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    axes[0, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Internal held-out test ROC curves", xlim=(0, 1), ylim=(0, 1))
    axes[0, 0].legend(frameon=False, loc="lower right")
    axes[0, 1].axhline(prevalence, color="#777777", linestyle="--", linewidth=1, label=f"Prevalence {prevalence:.3f}")
    axes[0, 1].set(xlabel="Recall", ylabel="Precision", title="Internal held-out test precision-recall curves", xlim=(0, 1), ylim=(0, 1.02))
    axes[0, 1].legend(frameon=False, loc="lower left")
    for axis, model_name in zip(axes[1], MODEL_CONFIG, strict=True):
        group = predictions[predictions["model"].eq(model_name)]
        labels = group["true_label"].to_numpy(dtype=int)
        for column, label, color in (
            ("raw_probability", "Raw", RAW_COLOR),
            ("temperature_scaled_probability", "Temperature scaled", SCALED_COLOR),
        ):
            means, observed, counts = calibration_points(labels, group[column].to_numpy(dtype=float))
            axis.plot(means, observed, marker="o", markersize=4, color=color, label=label)
        axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        axis.set(xlabel="Mean predicted probability", ylabel="Observed pneumonia proportion", title=f"{model_name} reliability (10 equal-width bins)", xlim=(0, 1), ylim=(0, 1))
        axis.legend(frameon=False, loc="upper left")
    for axis, label in zip(axes.flat, "abcd", strict=True):
        add_panel_label(axis, label)
        axis.grid(alpha=0.15)
    save_figure(fig, MAIN_FIGURES, "Figure_2_internal_performance")


def figure_3_probability_shift() -> None:
    paired = pd.read_csv(PAIRED_SHIFT, encoding="utf-8-sig")
    paired = paired[paired["comparison"].eq("PneumoniaMNIST_28x28")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    rng = np.random.default_rng(20260805)
    for axis, model_name in zip(axes, MODEL_CONFIG, strict=True):
        data = paired[paired["model"].eq(model_name)]
        threshold = float(data["internal_validation_threshold"].iloc[0])
        for label_name in ("NORMAL", "PNEUMONIA"):
            group = data[data["label_name"].eq(label_name)]
            jitter_x = np.clip(group["original_probability"].to_numpy() + rng.normal(0, 0.002, len(group)), 0, 1)
            jitter_y = np.clip(group["transformed_probability"].to_numpy() + rng.normal(0, 0.002, len(group)), 0, 1)
            axis.scatter(jitter_x, jitter_y, s=24, alpha=0.70, color=CLASS_COLORS[label_name], edgecolor="white", linewidth=0.35, label=f"{label_name} (n={len(group)})")
        axis.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.1, label="No probability shift")
        axis.axvline(threshold, color="#7A3E9D", linestyle=":", linewidth=1.3)
        axis.axhline(threshold, color="#7A3E9D", linestyle=":", linewidth=1.3, label=f"Frozen threshold {threshold:.3f}")
        axis.set(xlabel="Original high-resolution pneumonia probability", ylabel="PneumoniaMNIST 28x28 pneumonia probability", title=model_name, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.set_aspect("equal", adjustable="box")
        axis.legend(frameon=False, loc="lower right", fontsize=7.3)
        axis.grid(alpha=0.12)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_figure(fig, MAIN_FIGURES, "Figure_3_paired_probability_shift")


def figure_4_repeated_splits() -> None:
    repeated = pd.read_csv(REPEATED, encoding="utf-8-sig")
    metric_panels = [
        ("Operating-point metrics", ["selected_accuracy", "selected_f1", "selected_sensitivity", "selected_specificity"], ["Accuracy", "F1", "Sensitivity", "Specificity"]),
        ("Ranking metrics", ["selected_roc_auc", "selected_pr_auc"], ["ROC AUC", "PR AUC"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for axis, (title, columns, labels) in zip(axes, metric_panels, strict=True):
        for x, (column, label) in enumerate(zip(columns, labels, strict=True)):
            values = repeated[column].to_numpy(dtype=float)
            offsets = np.linspace(-0.07, 0.07, len(values))
            axis.scatter(np.full(len(values), x) + offsets, values, color="#2F6B8A", s=28, alpha=0.82, zorder=3)
            axis.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt="D", color="#C44E52", ecolor="#C44E52", capsize=4, markersize=5, zorder=4)
            axis.text(x, values.min() - 0.006 if title == "Operating-point metrics" else values.min() - 0.0012, f"n=5", ha="center", va="top", fontsize=7)
        axis.set_xticks(range(len(labels)), labels)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("Metric value")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylim(0.86, 1.005)
    axes[1].set_ylim(0.987, 1.0005)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.text(0.5, -0.01, "Points show independently trained repeated patient/group splits; diamonds and bars show mean and SD.", ha="center", fontsize=8)
    save_figure(fig, MAIN_FIGURES, "Figure_4_repeated_split_performance")


def windows_path_to_local(value: str) -> Path:
    text = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:\\", text):
        return Path(f"/mnt/{text[0].lower()}/{text[3:].replace(chr(92), '/')}")
    return Path(text)


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    return image.crop(((width - side) // 2, (height - side) // 2, (width - side) // 2 + side, (height - side) // 2 + side))


def reconstruct_28(image: Image.Image, method: str) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("L")
    if method == "direct_pil_bilinear":
        return np.asarray(image.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.uint8)
    if method == "aspect_fill_pil_bilinear":
        return np.asarray(ImageOps.fit(image, (28, 28), Image.Resampling.BILINEAR), dtype=np.uint8)
    crop = center_square(image)
    if method == "center_crop_pil_bilinear":
        return np.asarray(crop.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.uint8)
    if method == "center_crop_pil_bicubic":
        return np.asarray(crop.resize((28, 28), Image.Resampling.BICUBIC), dtype=np.uint8)
    array = np.asarray(crop, dtype=np.uint8)
    if method == "center_crop_cv_area":
        return cv2.resize(array, (28, 28), interpolation=cv2.INTER_AREA).astype(np.uint8)
    if method == "center_crop_skimage_linear":
        output = sk_resize(array, (28, 28), order=1, anti_aliasing=True, preserve_range=True)
        return np.rint(output).clip(0, 255).astype(np.uint8)
    raise ValueError(method)


def supplementary_figure_s1_mapping_examples() -> None:
    mapping = pd.read_csv(OVERLAP_MAPPING, encoding="utf-8-sig")
    mapping = mapping[(mapping["candidate_rank"].eq(1)) & (mapping["mapping_status"].eq("confirmed_match"))]
    selections = []
    for subtype in ("NORMAL", "BACTERIA", "VIRUS"):
        group = mapping[mapping["official_subtype"].eq(subtype)].sort_values("ssim")
        selections.extend([group.iloc[-1], group.iloc[0]])
    med = np.load(PNEUMONIAMNIST_NPZ)
    med_images = med["test_images"]
    fig, axes = plt.subplots(6, 3, figsize=(8.8, 12.2), constrained_layout=False)
    for row_index, row in enumerate(selections):
        med_image = med_images[int(row["pneumoniamnist_index"])]
        source_path = windows_path_to_local(str(row["source_path"]))
        with Image.open(source_path) as source:
            source_display = ImageOps.exif_transpose(source).convert("L")
            reconstructed = reconstruct_28(source, str(row["best_reconstruction_method"]))
        axes[row_index, 0].imshow(med_image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[row_index, 1].imshow(source_display, cmap="gray")
        axes[row_index, 2].imshow(reconstructed, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[row_index, 0].set_ylabel(
            f"{row['official_subtype']} | {'higher' if row_index % 2 == 0 else 'lower'}-SSIM\n"
            f"internal {row['internal_split']} | {row['patient_id']}\n"
            f"SSIM {float(row['ssim']):.3f} | MAE {float(row['pixel_mae']):.2f}",
            fontsize=7.2,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#777777")
    for axis, title in zip(axes[0], ["PneumoniaMNIST 28x28", "Matched Kermany source", "Reconstructed 28x28"], strict=True):
        axis.set_title(title, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.955, bottom=0.04, hspace=0.045, wspace=0.08)
    fig.suptitle("Stratified confirmed source-mapping examples", y=0.985, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.012, "Selection rule: within each official subtype, one confirmed top-1 match from the higher and lower ends of the SSIM distribution.", ha="center", fontsize=7.5)
    save_figure(fig, SUPP_FIGURES, "Supplementary_Figure_S1_mapping_examples")


def supplementary_figure_s2_logit_shift() -> None:
    paired = pd.read_csv(PAIRED_SHIFT, encoding="utf-8-sig")
    paired = paired[paired["comparison"].eq("PneumoniaMNIST_28x28")]
    max_abs = float(np.nanmax(np.abs(paired["logit_shift"].to_numpy())))
    limit = math.ceil(max_abs * 1.05)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
    rng = np.random.default_rng(20260805)
    for axis, model_name in zip(axes, MODEL_CONFIG, strict=True):
        data = paired[paired["model"].eq(model_name)]
        arrays = [data[data["label_name"].eq(label)]["logit_shift"].to_numpy() for label in ("NORMAL", "PNEUMONIA")]
        box = axis.boxplot(arrays, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.3})
        for patch, label in zip(box["boxes"], ("NORMAL", "PNEUMONIA"), strict=True):
            patch.set_facecolor(CLASS_COLORS[label])
            patch.set_alpha(0.35)
            patch.set_edgecolor(CLASS_COLORS[label])
        for x, (label, values) in enumerate(zip(("NORMAL", "PNEUMONIA"), arrays, strict=True)):
            jitter = rng.normal(0, 0.055, len(values))
            axis.scatter(np.full(len(values), x) + jitter, values, s=16, alpha=0.52, color=CLASS_COLORS[label], edgecolor="none")
            axis.text(x, -limit * 0.94, f"n={len(values)}", ha="center", va="bottom", fontsize=7.5)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
        axis.set_ylim(-limit, limit)
        axis.set_ylabel("Transformed - original logit")
        axis.set_title(model_name)
        axis.grid(axis="y", alpha=0.15)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_figure(fig, SUPP_FIGURES, "Supplementary_Figure_S2_paired_logit_shift")


def supplementary_figure_s3_threshold_crossing() -> None:
    paired = pd.read_csv(PAIRED_SHIFT, encoding="utf-8-sig")
    paired = paired[paired["comparison"].eq("PneumoniaMNIST_28x28")]
    rows = []
    for model_name in MODEL_CONFIG:
        for label_name in ("NORMAL", "PNEUMONIA"):
            group = paired[(paired["model"].eq(model_name)) & (paired["label_name"].eq(label_name))]
            below_above = int(((group["original_prediction"].eq(0)) & (group["transformed_prediction"].eq(1))).sum())
            above_below = int(((group["original_prediction"].eq(1)) & (group["transformed_prediction"].eq(0))).sum())
            rows.append((model_name, label_name, len(group), below_above, above_below))
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    x = np.arange(len(rows))
    rates = [(up + down) / n for _, _, n, up, down in rows]
    colors = ["#C44E52" if up >= down else "#2F6B8A" for _, _, _, up, down in rows]
    bars = ax.bar(x, rates, color=colors, width=0.62, alpha=0.85)
    for bar, (_, _, n, up, down) in zip(bars, rows, strict=True):
        count = up + down
        direction = "below-to-above" if up >= down else "above-to-below"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{count}/{n}\n{100 * count / n:.2f}%\n{direction}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x, [f"{model}\n{label}" for model, label, *_ in rows])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Threshold-crossing proportion")
    ax.set_title("Crossings of the frozen internal-validation operating threshold")
    ax.grid(axis="y", alpha=0.18)
    ax.text(0.01, 0.98, "Red: below-to-above (toward PNEUMONIA)\nBlue: above-to-below (toward NORMAL)", transform=ax.transAxes, va="top", fontsize=7.5)
    save_figure(fig, SUPP_FIGURES, "Supplementary_Figure_S3_threshold_crossing")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def supplementary_figure_s4_probability_distributions() -> None:
    paired = pd.read_csv(PAIRED_SHIFT, encoding="utf-8-sig")
    paired = paired[paired["comparison"].eq("PneumoniaMNIST_28x28")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True, sharex=True, sharey=True)
    for row_index, model_name in enumerate(MODEL_CONFIG):
        data_model = paired[paired["model"].eq(model_name)]
        threshold = float(data_model["internal_validation_threshold"].iloc[0])
        for col_index, label_name in enumerate(("NORMAL", "PNEUMONIA")):
            axis = axes[row_index, col_index]
            data = data_model[data_model["label_name"].eq(label_name)]
            for column, label, color in (
                ("original_probability", "Original high resolution", RAW_COLOR),
                ("transformed_probability", "PneumoniaMNIST 28x28", CLASS_COLORS[label_name]),
            ):
                x, y = ecdf(data[column].to_numpy())
                axis.step(x, y, where="post", color=color, label=label)
            axis.axvline(threshold, color="#7A3E9D", linestyle=":", linewidth=1.3, label=f"Threshold {threshold:.3f}")
            axis.set(xlabel="Predicted pneumonia probability", ylabel="Empirical cumulative proportion", title=f"{model_name}: {label_name} (n={len(data)})", xlim=(0, 1), ylim=(0, 1.02))
            axis.grid(alpha=0.15)
            axis.legend(frameon=False, fontsize=7.2, loc="lower right")
    for axis, label in zip(axes.flat, "abcd", strict=True):
        add_panel_label(axis, label)
    save_figure(fig, SUPP_FIGURES, "Supplementary_Figure_S4_probability_ecdf")


def write_figure_contract() -> None:
    lines = [
        "# Submission v2 figure contract",
        "",
        "All figures use the existing Python/matplotlib analysis workflow and formal clean results only.",
        "",
        "| Figure | Core conclusion | Evidence role | Archetype |",
        "|---|---|---|---|",
        "| Figure 1 | A provenance-first pipeline removed local contamination and constrained all evaluation units before modelling. | Study flow and audit denominators | Schematic-led composite |",
        "| Figure 2 | Frozen models show high internal ranking performance, while reliability differs before and after validation-fitted temperature scaling. | Internal discrimination and calibration | Quantitative grid |",
        "| Figure 3 | Severe 28x28 transformation shifts probability scale, especially for normal images, and crosses frozen thresholds. | Paired preprocessing shift | Quantitative comparison |",
        "| Figure 4 | Internal performance is stable across five independently trained group splits, with greater variability in specificity. | Robustness analysis | Quantitative grid |",
        "| Supplementary Figure S1 | Confirmed local source mappings occur across normal, bacterial and viral source subtypes and across the observed SSIM range. | Provenance examples | Image plate + quantification |",
        "| Supplementary Figure S2 | Logit shifts differ by class and model. | Paired score-scale analysis | Quantitative comparison |",
        "| Supplementary Figure S3 | Threshold crossings are predominantly below-to-above for normal images. | Operating-point failure mode | Quantitative comparison |",
        "| Supplementary Figure S4 | ECDFs show class-stratified probability redistribution without count-scale distortion. | Distributional context | Quantitative grid |",
        "",
        "Export contract: 300 dpi PNG plus vector PDF and SVG; editable vector text; fixed class/model color mapping; source data retained in formal CSV files.",
    ]
    (REPORT_DIR / "figure_contract_v2.md").write_text("\n".join(lines), encoding="utf-8")


def validate_figure_outputs() -> None:
    expected = [
        MAIN_FIGURES / "Figure_1_workflow.png",
        MAIN_FIGURES / "Figure_2_internal_performance.png",
        MAIN_FIGURES / "Figure_3_paired_probability_shift.png",
        MAIN_FIGURES / "Figure_4_repeated_split_performance.png",
        SUPP_FIGURES / "Supplementary_Figure_S1_mapping_examples.png",
        SUPP_FIGURES / "Supplementary_Figure_S2_paired_logit_shift.png",
        SUPP_FIGURES / "Supplementary_Figure_S3_threshold_crossing.png",
        SUPP_FIGURES / "Supplementary_Figure_S4_probability_ecdf.png",
    ]
    for path in expected:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape[0] < 900 or array.shape[1] < 1200 or float(array.std()) < 5.0:
            raise RuntimeError(f"Figure QA failed for {path}: shape={array.shape}, std={array.std()}")
        for suffix in ("pdf", "svg"):
            vector = path.with_suffix(f".{suffix}")
            if not vector.exists() or vector.stat().st_size < 1000:
                raise RuntimeError(f"Missing vector export: {vector}")


def main() -> None:
    configure_matplotlib()
    for folder in (REPORT_DIR, TABLE_DIR, MAIN_FIGURES, SUPP_FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    thresholds, temperatures = load_internal_parameters()
    print(f"Device: {DEVICE}")
    print("Frozen internal held-out inference...")
    predictions, metrics, bootstrap = infer_internal_test(thresholds, temperatures)
    verify_internal_metrics(metrics)
    predictions.to_csv(REPORT_DIR / "internal_test_predictions_v2.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(REPORT_DIR / "internal_test_metrics_with_group_ci_v2.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "internal_test_bootstrap_v2.json").write_text(json.dumps(bootstrap, indent=2, allow_nan=True), encoding="utf-8")

    pair_table, pair_json = paired_summary()
    pair_table.to_csv(REPORT_DIR / "paired_shift_summary_v2.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "paired_shift_bootstrap_v2.json").write_text(json.dumps(pair_json, indent=2), encoding="utf-8")
    build_submission_tables(metrics)

    print("Generating publication figures...")
    figure_1_workflow()
    figure_2_internal_performance(predictions)
    figure_3_probability_shift()
    figure_4_repeated_splits()
    supplementary_figure_s1_mapping_examples()
    supplementary_figure_s2_logit_shift()
    supplementary_figure_s3_threshold_crossing()
    supplementary_figure_s4_probability_distributions()
    write_figure_contract()
    validate_figure_outputs()
    print("PASS: submission-v2 statistics and figures generated without training.")


if __name__ == "__main__":
    main()
