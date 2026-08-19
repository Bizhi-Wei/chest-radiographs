"""
Train a lightweight baseline model on the strict patient-level split.

Default model: MobileNetV3-Small.
Optional model: DenseNet121.

Examples:
    python 14_train_patient_split_baseline.py --model mobilenet_v3
    python 14_train_patient_split_baseline.py --model densenet121
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from patient_group_utils import group_bootstrap_ci, group_ids_from_imagefolder


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("PNEUMONIA_DATASET_DIR", BASE_DIR / "data" / "chest_xray_patient_split"))
if not DATASET_DIR.is_absolute():
    DATASET_DIR = BASE_DIR / DATASET_DIR
OUTPUT_ROOT = Path(os.environ.get("PNEUMONIA_OUTPUT_ROOT", BASE_DIR / "output_patient_split_baselines"))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = BASE_DIR / OUTPUT_ROOT

IMAGE_SIZE = 224
BATCH_SIZE = int(os.environ.get("PNEUMONIA_BATCH_SIZE", "64"))
NUM_EPOCHS = int(os.environ.get("PNEUMONIA_NUM_EPOCHS", "20"))
LEARNING_RATE = 1e-4
PATIENCE = 5
SEED = 42
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
NUM_WORKERS = int(os.environ.get("PNEUMONIA_NUM_WORKERS", str(min(6, max(2, (os.cpu_count() or 4) // 2)))))
torch.backends.cudnn.benchmark = DEVICE.type == "cuda"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best: float | None = None
        self.stop = False

    def __call__(self, score: float) -> None:
        if self.best is None:
            self.best = score
        elif score < self.best + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best = score
            self.counter = 0


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


def prepare_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"
    for path in (train_dir, val_dir, test_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing patient split folder: {path}")

    train_ds = datasets.ImageFolder(str(train_dir), transform=get_transforms(True))
    val_ds = datasets.ImageFolder(str(val_dir), transform=get_transforms(False))
    test_ds = datasets.ImageFolder(str(test_dir), transform=get_transforms(False))

    targets = np.array([sample[1] for sample in train_ds.samples])
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    weights = (1.0 / counts)[targets]
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }
    if NUM_WORKERS > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_ds)} images, class counts={counts.tolist()}")
    print(f"Val:   {len(val_ds)} images")
    print(f"Test:  {len(test_ds)} images")
    return train_loader, val_loader, test_loader


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "mobilenet_v3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    else:
        raise ValueError("Unsupported model. Choose: mobilenet_v3 or densenet121")
    return model.to(DEVICE)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_sum = 0.0
    total = 0
    correct = 0
    preds: list[int] = []
    labels_all: list[int] = []
    probs: list[float] = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        iterator = tqdm(loader, desc="Training" if train_mode else "Evaluating", leave=False)
        for images, labels in iterator:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train_mode:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            prob = F.softmax(outputs.detach(), dim=1)
            pred = outputs.detach().argmax(dim=1)
            total += batch_size
            correct += pred.eq(labels).sum().item()
            preds.extend(pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs.extend(prob[:, 1].cpu().numpy())
            iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    return (
        loss_sum / total,
        correct / total,
        np.asarray(preds),
        np.asarray(labels_all),
        np.asarray(probs),
    )


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    criterion = FocalLoss()
    _, acc, preds, labels, probs = run_epoch(model, loader, criterion)
    group_ids = group_ids_from_imagefolder(loader.dataset)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    results = {
        "accuracy": float(acc),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "cm": cm,
        "preds": preds,
        "labels": labels,
        "probs": probs,
        "group_ids": group_ids,
    }
    results.update(group_bootstrap_ci(labels, preds, probs, group_ids, seed=SEED))
    return results


def train_model(model_name: str) -> None:
    seed_everything(SEED)
    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Patient-level baseline training: {model_name}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {output_dir}")
    print(f"Device:  {DEVICE}")
    print("=" * 70)

    train_loader, val_loader, test_loader = prepare_data()
    model = build_model(model_name)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    early_stopping = EarlyStopping()

    best_val_acc = 0.0
    best_state = None
    history = []
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss, train_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, _, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / f"best_{model_name}.pth")
            print(f"  Best val_acc={best_val_acc:.4f}")

        early_stopping(val_acc)
        if early_stopping.stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    results = evaluate(model, test_loader)
    results["model"] = model_name
    results["best_val_acc"] = float(best_val_acc)
    results["total_time_sec"] = float(time.time() - start)
    torch.save(results, output_dir / f"results_{model_name}.pth")

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        key: results[key]
        for key in [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "ci_acc",
            "ci_f1",
            "ci_auc",
            "ci_pr_auc",
            "ci_method",
            "bootstrap_groups",
            "bootstrap_samples",
            "best_val_acc",
            "total_time_sec",
        ]
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nFinal test results")
    print("=" * 70)
    for key in ["accuracy", "f1", "roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "npv"]:
        print(f"{key:12s}: {results[key]:.4f}")
    print("Confusion matrix:")
    print(results["cm"])
    print(f"Saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mobilenet_v3", "densenet121"], default="mobilenet_v3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.model)
