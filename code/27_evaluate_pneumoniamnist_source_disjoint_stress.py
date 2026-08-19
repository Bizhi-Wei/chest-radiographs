#!/usr/bin/env python3
"""Evaluate the source-disjoint PneumoniaMNIST preprocessing stress subset.

The models, operating thresholds, and temperature parameters are frozen from
the clean internal workflow.  PneumoniaMNIST is never used for optimization.
"""

from __future__ import annotations

import json
import math
import os
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from skimage.transform import resize as sk_resize
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "01_data/manifests/pneumoniamnist_source_disjoint_stress_test.csv"
NPZ_PATH = ROOT / "01_data/external/pneumoniamnist.npz"
CLEAN_DATA_DIR = ROOT / "01_data/chest_xray_patient_split_dedup_clean"
THRESHOLD_TABLE = ROOT / "04_reports/final_tables/clean_20260706_validation_selected_thresholds.csv"
CALIBRATION_TABLE = ROOT / "04_reports/final_tables/clean_20260706_calibration_metrics.csv"
RESNET_CHECKPOINT = ROOT / "03_models_and_outputs/output_clean_20260706/resnet18/best_resnet18_nomixup.pth"
MOBILE_CHECKPOINT = ROOT / "03_models_and_outputs/output_clean_20260706/baselines/mobilenet_v3/best_mobilenet_v3.pth"
REPORT_DIR = ROOT / "04_reports"
FIGURE_DIR = REPORT_DIR / "figures"

THRESHOLD_POLICY = "internal_validation_only"
TEMPERATURE_POLICY = "internal_validation_only"
PNEUMONIAMNIST_THRESHOLD_OPTIMIZATION = False
PNEUMONIAMNIST_TEMPERATURE_REFIT = False
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260804
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "ResNet18": {
        "checkpoint": RESNET_CHECKPOINT,
        "architecture": "resnet18",
    },
    "MobileNetV3-Small": {
        "checkpoint": MOBILE_CHECKPOINT,
        "architecture": "mobilenet_v3_small",
    },
}

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def windows_path_to_local(value: str) -> Path:
    path = str(value)
    if os.name != "nt" and len(path) >= 3 and path[1:3] == ":\\":
        path = f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return Path(path)


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def reconstruct_28(image: Image.Image, method: str) -> np.ndarray:
    image = image.convert("L")
    if method == "direct_pil_bilinear":
        return np.asarray(image.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.uint8)
    if method == "aspect_fill_pil_bilinear":
        return np.asarray(
            ImageOps.fit(image, (28, 28), Image.Resampling.BILINEAR, centering=(0.5, 0.5)),
            dtype=np.uint8,
        )
    cropped = center_square(image)
    if method == "center_crop_pil_bilinear":
        return np.asarray(cropped.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.uint8)
    if method == "center_crop_pil_bicubic":
        return np.asarray(cropped.resize((28, 28), Image.Resampling.BICUBIC), dtype=np.uint8)
    array = np.asarray(cropped, dtype=np.uint8)
    if method == "center_crop_cv_area":
        return cv2.resize(array, (28, 28), interpolation=cv2.INTER_AREA).astype(np.uint8)
    if method == "center_crop_skimage_linear":
        result = sk_resize(
            array,
            (28, 28),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.rint(result).clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported confirmed reconstruction method: {method}")


def to_model_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        pil = image.convert("L")
    return INFERENCE_TRANSFORM(pil.convert("RGB"))


def load_internal_parameters() -> tuple[dict[str, float], dict[str, float]]:
    thresholds = pd.read_csv(THRESHOLD_TABLE, encoding="utf-8-sig")
    selected = thresholds[thresholds["Selection rule"].eq("Best F1 on validation")]
    threshold_map = {
        str(row.Model): float(row.Threshold) for row in selected.itertuples(index=False)
    }
    calibration = pd.read_csv(CALIBRATION_TABLE, encoding="utf-8-sig")
    scaled = calibration[calibration["Probability type"].eq("temperature_scaled")]
    temperature_map = {
        str(row.Model): float(row.Temperature) for row in scaled.itertuples(index=False)
    }
    expected = {"ResNet18": 0.450, "MobileNetV3-Small": 0.345}
    for model_name, expected_threshold in expected.items():
        actual = threshold_map.get(model_name)
        if actual is None or not math.isclose(actual, expected_threshold, abs_tol=1e-12):
            raise ValueError(
                f"Internal threshold mismatch for {model_name}: {actual}; expected {expected_threshold}"
            )
        if model_name not in temperature_map:
            raise ValueError(f"Missing internal validation temperature for {model_name}")
    return threshold_map, temperature_map


def build_model(architecture: str, checkpoint: Path) -> nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
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


@torch.no_grad()
def infer_logits(model: nn.Module, tensors: list[torch.Tensor], batch_size: int = 32) -> np.ndarray:
    results = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size]).to(DEVICE)
        outputs = model(batch)
        results.append(outputs.detach().cpu().numpy())
    return np.concatenate(results, axis=0)


def binary_logit_and_probability(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary_logit = logits[:, 1] - logits[:, 0]
    probability = 1.0 / (1.0 + np.exp(-np.clip(binary_logit, -50, 50)))
    return binary_logit, probability


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= boundaries[index]) & (probabilities <= boundaries[index + 1])
        else:
            mask = (probabilities >= boundaries[index]) & (probabilities < boundaries[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return total


def metric_values(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "normal_n": int((labels == 0).sum()),
        "pneumonia_n": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(precision_score(labels, predictions, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece": expected_calibration_error(labels, probabilities),
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
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    unique_groups = np.unique(groups.astype(str))
    group_indices = {group: np.flatnonzero(groups.astype(str) == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    valid = 0
    for _ in range(samples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        if np.unique(labels[indices]).size < 2:
            continue
        values = metric_values(labels[indices], probabilities[indices], threshold)
        valid += 1
        for key, value in values.items():
            if key not in {"n", "normal_n", "pneumonia_n", "tn", "fp", "fn", "tp"}:
                draws[key].append(float(value))
    intervals = {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in draws.items()
        if values
    }
    return {
        "method": "patient_group_bootstrap",
        "groups": int(len(unique_groups)),
        "requested_samples": int(samples),
        "valid_samples": int(valid),
        "seed": int(seed),
        "intervals": intervals,
    }


def prepare_inputs(manifest: pd.DataFrame, med_images: np.ndarray) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    original_tensors: list[torch.Tensor] = []
    med_tensors: list[torch.Tensor] = []
    reconstructed_tensors: list[torch.Tensor] = []
    for row in manifest.itertuples(index=False):
        clean_path = (
            CLEAN_DATA_DIR
            / str(row.internal_split)
            / str(row.source_label)
            / str(row.destination_name)
        )
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean held-out source image: {clean_path}")
        with Image.open(clean_path) as image:
            original_tensors.append(to_model_tensor(image))
        med_array = med_images[int(row.pneumoniamnist_index)]
        med_tensors.append(to_model_tensor(med_array))
        with Image.open(windows_path_to_local(str(row.source_path))) as source_image:
            reconstructed = reconstruct_28(source_image, str(row.best_reconstruction_method))
        reconstructed_tensors.append(to_model_tensor(reconstructed))
    return original_tensors, med_tensors, reconstructed_tensors


def paired_bootstrap_summary(table: pd.DataFrame) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: dict[str, object] = {}
    for (model_name, comparison, label_name), group in table.groupby(
        ["model", "comparison", "label_name"], dropna=False
    ):
        unique_groups = group["patient_id"].astype(str).unique()
        values = {"mean_probability_shift": [], "flip_rate": [], "threshold_crossing_rate": []}
        for _ in range(BOOTSTRAP_SAMPLES):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            frame = pd.concat(
                [group[group["patient_id"].astype(str).eq(patient)] for patient in sampled],
                ignore_index=True,
            )
            values["mean_probability_shift"].append(float(frame["probability_shift"].mean()))
            values["flip_rate"].append(float(frame["prediction_flip"].mean()))
            values["threshold_crossing_rate"].append(float(frame["threshold_crossing"].mean()))
        key = f"{model_name}|{comparison}|{label_name}"
        output[key] = {
            name: {
                "estimate": float(
                    group["probability_shift"].mean()
                    if name == "mean_probability_shift"
                    else group["prediction_flip"].mean()
                    if name == "flip_rate"
                    else group["threshold_crossing"].mean()
                ),
                "95%_ci": [
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                ],
            }
            for name, samples in values.items()
        }
    return output


def make_shift_figures(paired: pd.DataFrame) -> None:
    colors = {"NORMAL": "#277DA1", "PNEUMONIA": "#D1495B"}
    models_order = list(MODEL_CONFIG)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, model_name in zip(axes, models_order, strict=True):
        data = paired[(paired["model"].eq(model_name)) & (paired["comparison"].eq("PneumoniaMNIST_28x28"))]
        for label_name, group in data.groupby("label_name"):
            axis.scatter(
                group["original_probability"],
                group["transformed_probability"],
                s=25,
                alpha=0.75,
                color=colors[label_name],
                label=label_name,
            )
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Original high-resolution probability", ylabel="PneumoniaMNIST probability", title=model_name, xlim=(0, 1), ylim=(0, 1))
        axis.legend(frameon=False)
    fig.suptitle("Paired probability shift under 28x28 preprocessing")
    fig.savefig(FIGURE_DIR / "pneumoniamnist_probability_shift.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, model_name in zip(axes, models_order, strict=True):
        data = paired[(paired["model"].eq(model_name)) & (paired["comparison"].eq("PneumoniaMNIST_28x28"))]
        groups = [data.loc[data["label_name"].eq(label), "logit_shift"].to_numpy() for label in ("NORMAL", "PNEUMONIA")]
        box = axis.boxplot(
            groups,
            tick_labels=["NORMAL", "PNEUMONIA"],
            patch_artist=True,
            showfliers=False,
        )
        for patch, label in zip(box["boxes"], ("NORMAL", "PNEUMONIA"), strict=True):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.65)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set(ylabel="Transformed - original logit", title=model_name)
    fig.suptitle("Paired logit shift under 28x28 preprocessing")
    fig.savefig(FIGURE_DIR / "pneumoniamnist_logit_shift.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    crossing = (
        paired.groupby(["model", "comparison", "label_name"])["threshold_crossing"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, model_name in zip(axes, models_order, strict=True):
        data = crossing[crossing["model"].eq(model_name)]
        x = np.arange(2)
        width = 0.35
        for offset, comparison in enumerate(("PneumoniaMNIST_28x28", "reconstructed_28x28")):
            values = []
            for label_name in ("NORMAL", "PNEUMONIA"):
                cell = data[data["comparison"].eq(comparison) & data["label_name"].eq(label_name)]
                values.append(float(cell["threshold_crossing"].iloc[0]) if len(cell) else 0.0)
            axis.bar(x + (offset - 0.5) * width, values, width, label=comparison)
        axis.set_xticks(x, ["NORMAL", "PNEUMONIA"])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Threshold-crossing proportion")
        axis.set_title(model_name)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Internal-validation threshold crossings")
    fig.savefig(FIGURE_DIR / "pneumoniamnist_threshold_crossing.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    variants = [
        ("original_probability", "Original high resolution"),
        ("transformed_probability", "PneumoniaMNIST 28x28"),
    ]
    for model_index, model_name in enumerate(models_order):
        data = paired[(paired["model"].eq(model_name)) & (paired["comparison"].eq("PneumoniaMNIST_28x28"))]
        for variant_index, (column, title) in enumerate(variants):
            axis = axes[model_index, variant_index]
            for label_name, group in data.groupby("label_name"):
                axis.hist(group[column], bins=np.linspace(0, 1, 16), alpha=0.55, color=colors[label_name], label=label_name)
            axis.set(xlabel="Predicted pneumonia probability", ylabel="Count", title=f"{model_name}: {title}")
            axis.legend(frameon=False)
    fig.suptitle("Probability distributions before and after severe preprocessing")
    fig.savefig(FIGURE_DIR / "pneumoniamnist_probability_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, paired: pd.DataFrame, subset_n: int, groups_n: int) -> None:
    lines = [
        "# Same-source PneumoniaMNIST preprocessing and resolution-shift stress test",
        "",
        "> PneumoniaMNIST is derived from the same underlying Kermany pediatric chest radiograph collection. This is not independent external validation.",
        "",
        f"- Conservative source-disjoint transformed subset: {subset_n} images from {groups_n} patient/groups",
        "- Source requirement: confirmed local mapping to the internal held-out test split only",
        "- Model weights: frozen clean checkpoints",
        "- Operating thresholds: frozen from internal validation (ResNet18 0.450; MobileNetV3-Small 0.345)",
        "- Temperature scaling: frozen from internal validation",
        "- Uncertainty: 1,000-sample patient/group bootstrap",
        "- No PneumoniaMNIST threshold optimization or temperature refitting was performed",
        "",
        "## Metrics",
        "",
        metrics.to_csv(index=False),
        "",
        "## Paired preprocessing shift",
        "",
    ]
    for model_name in MODEL_CONFIG:
        data = paired[(paired["model"].eq(model_name)) & (paired["comparison"].eq("PneumoniaMNIST_28x28"))]
        lines.extend(
            [
                f"### {model_name}",
                "",
                f"- Mean probability shift: {data['probability_shift'].mean():.4f}",
                f"- Median probability shift: {data['probability_shift'].median():.4f}",
                f"- Prediction flip rate: {data['prediction_flip'].mean():.4f}",
                f"- Internal-threshold crossing rate: {data['threshold_crossing'].mean():.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific interpretation",
            "",
            "The compliant question is: How stable are model ranking, probability scale, and fixed operating thresholds when the same underlying source radiographs undergo severe resolution reduction and preprocessing changes?",
            "",
            "No independent external clinical cohort was evaluated. These results cannot establish cross-hospital generalization or clinical portability.",
            "",
        ]
    )
    (REPORT_DIR / "pneumoniamnist_source_disjoint_stress_test_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    if PNEUMONIAMNIST_THRESHOLD_OPTIMIZATION or PNEUMONIAMNIST_TEMPERATURE_REFIT:
        raise RuntimeError("PneumoniaMNIST optimization is prohibited")
    required = [MANIFEST, NPZ_PATH, THRESHOLD_TABLE, CALIBRATION_TABLE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if manifest.empty:
        raise ValueError("No conservative source-disjoint subset was created")
    if not manifest["mapping_status"].eq("confirmed_match").all():
        raise ValueError("Stress subset contains non-confirmed mappings")
    if not manifest["internal_split"].eq("test").all():
        raise ValueError("Stress subset contains internal train/validation source images")
    med = dict(np.load(NPZ_PATH))
    threshold_map, temperature_map = load_internal_parameters()
    original_inputs, med_inputs, reconstructed_inputs = prepare_inputs(manifest, med["test_images"])

    labels = manifest["pneumoniamnist_label"].to_numpy(dtype=int)
    groups = manifest["patient_id"].astype(str).to_numpy()
    prediction_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    bootstrap_output: dict[str, object] = {
        "policy": {
            "threshold_source": THRESHOLD_POLICY,
            "temperature_source": TEMPERATURE_POLICY,
            "pneumoniamnist_threshold_optimization": False,
            "pneumoniamnist_temperature_refit": False,
        },
        "models": {},
    }

    for model_name, config in MODEL_CONFIG.items():
        model = build_model(str(config["architecture"]), Path(config["checkpoint"]))
        original_logits_raw = infer_logits(model, original_inputs)
        med_logits_raw = infer_logits(model, med_inputs)
        reconstructed_logits_raw = infer_logits(model, reconstructed_inputs)
        original_logit, original_probability = binary_logit_and_probability(original_logits_raw)
        med_logit, med_probability = binary_logit_and_probability(med_logits_raw)
        reconstructed_logit, reconstructed_probability = binary_logit_and_probability(reconstructed_logits_raw)
        temperature = temperature_map[model_name]
        med_scaled_probability = 1.0 / (1.0 + np.exp(-np.clip(med_logit / temperature, -50, 50)))
        threshold = threshold_map[model_name]

        for row_index, source_row in enumerate(manifest.itertuples(index=False)):
            base = {
                "model": model_name,
                "pneumoniamnist_index": int(source_row.pneumoniamnist_index),
                "true_label": int(labels[row_index]),
                "label_name": "PNEUMONIA" if labels[row_index] else "NORMAL",
                "patient_id": str(source_row.patient_id),
                "source_path": str(source_row.source_path),
                "source_sha256": str(source_row.source_sha256),
                "internal_split": str(source_row.internal_split),
                "mapping_status": str(source_row.mapping_status),
                "internal_validation_threshold": threshold,
                "internal_validation_temperature": temperature,
            }
            prediction_rows.append(
                {
                    **base,
                    "raw_logit": float(med_logit[row_index]),
                    "raw_probability": float(med_probability[row_index]),
                    "temperature_scaled_probability": float(med_scaled_probability[row_index]),
                    "prediction_default_0_5": int(med_probability[row_index] >= 0.5),
                    "prediction_internal_validation_threshold": int(med_probability[row_index] >= threshold),
                    "original_probability": float(original_probability[row_index]),
                    "reconstructed_probability": float(reconstructed_probability[row_index]),
                }
            )
            for comparison, transformed_logit, transformed_probability in (
                ("PneumoniaMNIST_28x28", med_logit, med_probability),
                ("reconstructed_28x28", reconstructed_logit, reconstructed_probability),
            ):
                original_prediction = int(original_probability[row_index] >= threshold)
                transformed_prediction = int(transformed_probability[row_index] >= threshold)
                paired_rows.append(
                    {
                        **base,
                        "comparison": comparison,
                        "original_logit": float(original_logit[row_index]),
                        "transformed_logit": float(transformed_logit[row_index]),
                        "logit_shift": float(transformed_logit[row_index] - original_logit[row_index]),
                        "original_probability": float(original_probability[row_index]),
                        "transformed_probability": float(transformed_probability[row_index]),
                        "probability_shift": float(transformed_probability[row_index] - original_probability[row_index]),
                        "original_prediction": original_prediction,
                        "transformed_prediction": transformed_prediction,
                        "prediction_flip": int(original_prediction != transformed_prediction),
                        "threshold_crossing": int(original_prediction != transformed_prediction),
                    }
                )

        model_bootstrap: dict[str, object] = {}
        for threshold_name, current_threshold in (
            ("default_0.5", 0.5),
            ("internal_validation_selected", threshold),
        ):
            values = metric_values(labels, med_probability, current_threshold)
            bootstrap = group_bootstrap(labels, med_probability, groups, current_threshold)
            calibration_raw = {
                "brier_raw": values["brier"],
                "nll_raw": values["nll"],
                "ece_raw": values["ece"],
                "brier_temperature_scaled": float(brier_score_loss(labels, med_scaled_probability)),
                "nll_temperature_scaled": float(log_loss(labels, med_scaled_probability, labels=[0, 1])),
                "ece_temperature_scaled": expected_calibration_error(labels, med_scaled_probability),
            }
            metric_row = {
                "model": model_name,
                "stress_test_role": "same-source preprocessing and resolution-shift stress test",
                "threshold_name": threshold_name,
                "threshold": current_threshold,
                "threshold_source": "fixed_default" if threshold_name == "default_0.5" else THRESHOLD_POLICY,
                "temperature": temperature,
                "temperature_source": TEMPERATURE_POLICY,
                **values,
                **calibration_raw,
                "ci_method": bootstrap["method"],
                "bootstrap_groups": bootstrap["groups"],
                "bootstrap_samples": bootstrap["valid_samples"],
            }
            for key, interval in bootstrap["intervals"].items():
                metric_row[f"{key}_95ci_low"] = interval[0]
                metric_row[f"{key}_95ci_high"] = interval[1]
            metric_rows.append(metric_row)
            model_bootstrap[threshold_name] = bootstrap
        bootstrap_output["models"][model_name] = model_bootstrap

    predictions = pd.DataFrame(prediction_rows)
    paired = pd.DataFrame(paired_rows)
    metrics = pd.DataFrame(metric_rows)
    bootstrap_output["paired_shift"] = paired_bootstrap_summary(paired)

    predictions.to_csv(
        REPORT_DIR / "pneumoniamnist_source_disjoint_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        REPORT_DIR / "pneumoniamnist_source_disjoint_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        REPORT_DIR / "pneumoniamnist_paired_shift_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (REPORT_DIR / "pneumoniamnist_source_disjoint_bootstrap.json").write_text(
        json.dumps(bootstrap_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_shift_figures(paired)
    write_report(metrics, paired, len(manifest), manifest["patient_id"].nunique())
    print(metrics.to_string(index=False))
    print(f"PASS: frozen-model stress test completed on {len(manifest)} images / {manifest['patient_id'].nunique()} groups")


if __name__ == "__main__":
    main()
