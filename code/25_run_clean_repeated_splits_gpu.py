"""
Train ResNet18 across repeated leakage-free patient/group splits.

This script is intended for the clean 2026-07-06 analysis. For every repeat it:
  - trains only on that repeat's train split
  - selects the operating threshold on that repeat's validation split
  - evaluates the selected threshold once on that repeat's test split
  - reports confidence intervals using patient/group bootstrap

Environment variables:
  PNEUMONIA_REPEATED_TARGET_ROOT   folder containing repeat_01...repeat_05
  PNEUMONIA_REPEATED_OUTPUT_DIR    output folder for checkpoints/reports
  PNEUMONIA_BATCH_SIZE             default 32
  PNEUMONIA_NUM_EPOCHS             default 15
  PNEUMONIA_NUM_WORKERS            default 2
"""

from __future__ import annotations

import copy
import csv
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from patient_group_utils import group_bootstrap_ci, group_ids_from_imagefolder
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


BASE_DIR = Path(__file__).resolve().parent
TARGET_ROOT = Path(os.environ.get("PNEUMONIA_REPEATED_TARGET_ROOT", BASE_DIR / "data" / "repeated_group_splits"))
OUTPUT_DIR = Path(os.environ.get("PNEUMONIA_REPEATED_OUTPUT_DIR", BASE_DIR / "output_clean_20260706" / "repeated_splits_resnet18"))
REPORT_DIR = OUTPUT_DIR / "reports"

IMAGE_SIZE = 224
BATCH_SIZE = int(os.environ.get("PNEUMONIA_BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.environ.get("PNEUMONIA_NUM_EPOCHS", "15"))
NUM_WORKERS = int(os.environ.get("PNEUMONIA_NUM_WORKERS", "2"))
LR = float(os.environ.get("PNEUMONIA_LR", "0.0001"))
PATIENCE = int(os.environ.get("PNEUMONIA_PATIENCE", "5"))
REPEATS = [f"repeat_{i:02d}" for i in range(1, int(os.environ.get("PNEUMONIA_REPEATS", "5")) + 1)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_transforms(augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def make_loaders(data_dir: Path) -> tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=get_transforms(True))
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=get_transforms(False))
    test_ds = datasets.ImageFolder(str(data_dir / "test"), transform=get_transforms(False))

    targets = np.asarray([label for _, label in train_ds.samples])
    counts = np.bincount(targets, minlength=2)
    sample_weights = (1.0 / np.maximum(counts, 1))[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def run_inference(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(images)
                probs = F.softmax(logits, dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())
    return np.asarray(labels_all), np.asarray(probs_all)


def metric_row(labels: np.ndarray, probs: np.ndarray, threshold: float, group_ids: np.ndarray) -> dict[str, object]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    ci = group_bootstrap_ci(labels, preds, probs, group_ids, n_bootstrap=1000, seed=SEED)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "ci_acc_low": ci["ci_acc"][0],
        "ci_acc_high": ci["ci_acc"][1],
        "ci_f1_low": ci["ci_f1"][0],
        "ci_f1_high": ci["ci_f1"][1],
        "ci_auc_low": ci["ci_auc"][0],
        "ci_auc_high": ci["ci_auc"][1],
        "ci_pr_auc_low": ci["ci_pr_auc"][0],
        "ci_pr_auc_high": ci["ci_pr_auc"][1],
        "bootstrap_groups": ci["bootstrap_groups"],
        "bootstrap_samples": ci["bootstrap_samples"],
    }


def select_best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.0, 1.0, 201):
        preds = (probs >= threshold).astype(int)
        value = f1_score(labels, preds, zero_division=0)
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    return best_threshold, best_f1


def train_one_repeat(repeat_name: str) -> dict[str, object]:
    data_dir = TARGET_ROOT / repeat_name
    if not (data_dir / "train").exists():
        raise FileNotFoundError(f"Missing materialized repeat split: {data_dir}")

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_loaders(data_dir)
    class_counts = Counter(label for _, label in train_ds.samples)
    print(f"\n{repeat_name}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} counts={dict(class_counts)}")

    model = build_model()
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    stale_epochs = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.numel()
        scheduler.step()

        val_labels, val_probs = run_inference(model, val_loader)
        val_preds = (val_probs >= 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_preds)
        train_acc = train_correct / max(train_total, 1)
        print(f"  epoch {epoch:02d}/{NUM_EPOCHS}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    ckpt_path = OUTPUT_DIR / f"best_resnet18_{repeat_name}.pth"
    torch.save(best_state, ckpt_path)

    val_labels, val_probs = run_inference(model, val_loader)
    test_labels, test_probs = run_inference(model, test_loader)
    test_groups = group_ids_from_imagefolder(test_ds)
    selected_threshold, val_best_f1 = select_best_f1_threshold(val_labels, val_probs)

    selected = metric_row(test_labels, test_probs, selected_threshold, test_groups)
    default = metric_row(test_labels, test_probs, 0.5, test_groups)
    print(
        f"  selected_threshold={selected_threshold:.3f} val_f1={val_best_f1:.4f} "
        f"test_acc={selected['accuracy']:.4f} test_f1={selected['f1']:.4f} auc={selected['roc_auc']:.4f}"
    )

    row = {
        "repeat": repeat_name,
        "checkpoint": str(ckpt_path),
        "best_val_acc_default_threshold": best_val_acc,
        "validation_selected_threshold": selected_threshold,
        "validation_best_f1": val_best_f1,
    }
    for prefix, metrics in [("selected", selected), ("default_0_5", default)]:
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]]) -> None:
    metric_keys = [
        "selected_accuracy",
        "selected_f1",
        "selected_roc_auc",
        "selected_pr_auc",
        "selected_sensitivity",
        "selected_specificity",
    ]
    summary_rows = []
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows])
        summary_rows.append(
            {
                "metric": key.replace("selected_", ""),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    write_csv(REPORT_DIR / "clean_20260706_repeated_split_resnet18_summary.csv", summary_rows)

    lines = [
        "# Clean repeated patient/group split ResNet18 results",
        "",
        f"- Target root: `{TARGET_ROOT}`",
        f"- Output dir: `{OUTPUT_DIR}`",
        f"- Device: `{DEVICE}`",
        f"- Repeats: {len(rows)}",
        f"- Epoch cap: {NUM_EPOCHS}",
        "- Threshold policy: selected on validation set by maximum F1, then frozen for test evaluation.",
        "- Confidence intervals: patient/group bootstrap on test groups.",
        "",
        "## Summary",
        "",
        "| Metric | Mean | SD | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['metric']} | {item['mean']:.4f} | {item['std']:.4f} | "
            f"{item['min']:.4f} | {item['max']:.4f} |"
        )
    (REPORT_DIR / "clean_20260706_repeated_split_resnet18_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Clean repeated patient/group split ResNet18 training")
    print(f"Target root: {TARGET_ROOT}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"Device:      {DEVICE}")
    print("=" * 72)

    rows = [train_one_repeat(repeat) for repeat in REPEATS]
    write_csv(REPORT_DIR / "clean_20260706_repeated_split_resnet18_per_repeat.csv", rows)
    write_summary(rows)
    print("\nPASS: repeated split training/evaluation complete.")
    print(f"Report: {REPORT_DIR / 'clean_20260706_repeated_split_resnet18_report.md'}")


if __name__ == "__main__":
    main()
