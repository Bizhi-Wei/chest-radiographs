"""Patient/group helpers for leakage-aware evaluation.

Confidence intervals should bootstrap independent patient/group units, not
individual images. This module keeps the patient-id extraction and grouped
bootstrap logic shared across model scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


SEED = 42


def extract_patient_id_from_path(path: str | Path) -> str:
    """Extract the project patient_id from an image path."""
    image_path = Path(path)
    label = image_path.parent.name.upper()
    stem = image_path.stem
    if label == "PNEUMONIA":
        match = re.search(r"(person\d+)", stem, flags=re.IGNORECASE)
    else:
        match = re.search(r"(NORMAL2-IM-\d+|IM-\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return stem.upper()


def group_ids_from_imagefolder(dataset) -> np.ndarray:
    """Return patient/group ids in the same order as ImageFolder samples."""
    return np.asarray([extract_patient_id_from_path(path) for path, _ in dataset.samples], dtype=object)


def group_bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    group_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """Calculate 95% CIs by resampling patient/group ids with replacement."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    group_ids = np.asarray(group_ids)
    if not (len(labels) == len(preds) == len(probs) == len(group_ids)):
        raise ValueError("labels, preds, probs and group_ids must have the same length.")

    rng = np.random.RandomState(seed)
    unique_groups = np.asarray(sorted(set(group_ids.tolist())), dtype=object)
    group_to_indices = {group: np.flatnonzero(group_ids == group) for group in unique_groups}
    accs: list[float] = []
    f1s: list[float] = []
    aucs: list[float] = []
    pr_aucs: list[float] = []

    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        sampled_labels = labels[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_preds = preds[sampled_indices]
        sampled_probs = probs[sampled_indices]
        accs.append(float(accuracy_score(sampled_labels, sampled_preds)))
        f1s.append(float(f1_score(sampled_labels, sampled_preds, zero_division=0)))
        aucs.append(float(roc_auc_score(sampled_labels, sampled_probs)))
        pr_aucs.append(float(average_precision_score(sampled_labels, sampled_probs)))

    if not accs:
        raise RuntimeError("No valid bootstrap samples contained both classes.")

    def pct(values: list[float]) -> tuple[float, float]:
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

    return {
        "ci_acc": pct(accs),
        "ci_f1": pct(f1s),
        "ci_auc": pct(aucs),
        "ci_pr_auc": pct(pr_aucs),
        "ci_method": "patient_group_bootstrap",
        "bootstrap_groups": int(len(unique_groups)),
        "bootstrap_samples": int(len(accs)),
    }


def group_bootstrap_metric_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    group_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> tuple[float, float]:
    """Generic group-level bootstrap CI for a probability-based metric."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    group_ids = np.asarray(group_ids)
    rng = np.random.RandomState(seed)
    unique_groups = np.asarray(sorted(set(group_ids.tolist())), dtype=object)
    group_to_indices = {group: np.flatnonzero(group_ids == group) for group in unique_groups}
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        sampled_labels = labels[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(float(metric_fn(sampled_labels, probs[sampled_indices])))
    if not values:
        raise RuntimeError("No valid bootstrap samples contained both classes.")
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
