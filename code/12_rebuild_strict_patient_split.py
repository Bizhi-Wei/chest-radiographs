"""Rebuild an exact-deduplicated, patient-level train/val/test split.

The script deliberately scans only the direct train/val/test folders under
data/chest_xray. Nested extraction copies and __MACOSX resource forks are not
included.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "data" / "chest_xray"
OUTPUT_DIR = BASE_DIR / "data" / "chest_xray_patient_split_dedup"
REPORT_DIR = BASE_DIR / "reports" / "recovery"
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
SPLITS = ("train", "val", "test")
LABELS = ("NORMAL", "PNEUMONIA")
SEED = 42

PNEUMONIA_RE = re.compile(r"(person\d+)", re.IGNORECASE)
NORMAL_RE = re.compile(r"(NORMAL2-IM-\d+|IM-\d+)", re.IGNORECASE)
SYNTHETIC_PLACEHOLDER_RE = re.compile(r"^(NORMAL|PNEUMONIA)_\d+\.(?:jpe?g|png)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_id_for(path: Path, label: str) -> tuple[str, str]:
    pattern = PNEUMONIA_RE if label == "PNEUMONIA" else NORMAL_RE
    match = pattern.search(path.name)
    if match:
        value = match.group(1)
        return (value.lower() if label == "PNEUMONIA" else value.upper()), "regex"
    return path.stem, "fallback"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scan_images(source_dir: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    excluded: list[dict] = []
    for source_split in SPLITS:
        for label in LABELS:
            class_dir = source_dir / source_split / label
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing input directory: {class_dir}")
            for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if SYNTHETIC_PLACEHOLDER_RE.fullmatch(path.name):
                    excluded.append(
                        {
                            "source_path": str(path.resolve()),
                            "source_split": source_split,
                            "label": label,
                            "filename": path.name,
                            "size_bytes": path.stat().st_size,
                            "reason": "synthetic_noise_placeholder_filename",
                        }
                    )
                    continue
                patient_id, patient_source = patient_id_for(path, label)
                records.append(
                    {
                        "source_path": str(path.resolve()),
                        "source_split": source_split,
                        "label": label,
                        "filename": path.name,
                        "patient_id": patient_id,
                        "patient_id_source": patient_source,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    if not records:
        raise RuntimeError(f"No images found under {source_dir}")
    return records, excluded


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_hash[row["sha256"]].append(row)

    retained: list[dict] = []
    duplicate_rows: list[dict] = []
    for digest, group in sorted(by_hash.items()):
        group = sorted(group, key=lambda row: (row["source_split"], row["label"], row["source_path"]))
        labels = {row["label"] for row in group}
        if len(labels) > 1:
            raise RuntimeError(f"Conflicting labels for SHA-256 {digest}: {sorted(labels)}")
        retained.append(group[0])
        for index, row in enumerate(group):
            duplicate_rows.append(
                {
                    **row,
                    "duplicate_group_size": len(group),
                    "retained": index == 0,
                    "retained_source_path": group[0]["source_path"],
                }
            )
    retained.sort(key=lambda row: row["source_path"])
    return retained, duplicate_rows


def subset_sum(groups: list[tuple[str, int]], target: int, rng: random.Random) -> set[str] | None:
    order = list(range(len(groups)))
    rng.shuffle(order)
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for index in order:
        weight = groups[index][1]
        for current in sorted(list(parents), reverse=True):
            new_sum = current + weight
            if new_sum > target or new_sum in parents:
                continue
            parents[new_sum] = (current, index)
        if target in parents:
            break
    if target not in parents:
        return None
    selected: set[str] = set()
    current = target
    while current:
        previous, index = parents[current]  # type: ignore[misc]
        selected.add(groups[index][0])
        current = previous
    return selected


def split_class(records: list[dict], seed: int) -> dict[str, str]:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_patient[row["patient_id"]].append(row)
    groups = sorted((patient_id, len(rows)) for patient_id, rows in by_patient.items())
    total = len(records)
    target_train = round(total * 0.70)
    target_val = round(total * 0.15)
    target_test = total - target_train - target_val

    for attempt in range(2000):
        rng = random.Random(seed + attempt * 1009)
        val_ids = subset_sum(groups, target_val, rng)
        if val_ids is None:
            continue
        remaining = [group for group in groups if group[0] not in val_ids]
        test_ids = subset_sum(remaining, target_test, rng)
        if test_ids is None:
            continue
        assignment = {}
        for patient_id, _ in groups:
            if patient_id in val_ids:
                assignment[patient_id] = "val"
            elif patient_id in test_ids:
                assignment[patient_id] = "test"
            else:
                assignment[patient_id] = "train"
        counts = Counter(assignment[row["patient_id"]] for row in records)
        if counts == Counter({"train": target_train, "val": target_val, "test": target_test}):
            return assignment
    raise RuntimeError(
        f"Could not produce exact 70/15/15 image counts for class with {total} images."
    )


def assign_splits(records: list[dict]) -> list[dict]:
    label_assignments: dict[str, dict[str, str]] = {}
    for offset, label in enumerate(LABELS):
        label_records = [row for row in records if row["label"] == label]
        patient_labels: dict[str, set[str]] = defaultdict(set)
        for row in label_records:
            patient_labels[row["patient_id"]].add(row["label"])
        label_assignments[label] = split_class(label_records, SEED + offset * 100_000)

    manifest: list[dict] = []
    for row in records:
        split = label_assignments[row["label"]][row["patient_id"]]
        destination_name = f"orig_{row['source_split']}__{row['filename']}"
        manifest.append(
            {
                **row,
                "split": split,
                "destination_name": destination_name,
            }
        )
    return manifest


def validate_manifest(manifest: list[dict]) -> list[dict]:
    checks: list[dict] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    patient_labels: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        patient_splits[row["patient_id"]].add(row["split"])
        patient_labels[row["patient_id"]].add(row["label"])
        hash_splits[row["sha256"]].add(row["split"])

    for patient_id, splits in sorted(patient_splits.items()):
        if len(splits) > 1:
            checks.append({"check": "patient_cross_split", "key": patient_id, "status": "FAIL", "splits": ";".join(sorted(splits))})
    for patient_id, labels in sorted(patient_labels.items()):
        if len(labels) > 1:
            checks.append({"check": "patient_label_conflict", "key": patient_id, "status": "FAIL", "splits": ";".join(sorted(labels))})
    for digest, splits in sorted(hash_splits.items()):
        if len(splits) > 1:
            checks.append({"check": "sha256_cross_split", "key": digest, "status": "FAIL", "splits": ";".join(sorted(splits))})
    if not checks:
        checks.append({"check": "all_leakage_checks", "key": "", "status": "PASS", "splits": ""})
    return checks


def safe_clear_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    expected_parent = (BASE_DIR / "data").resolve()
    if expected_parent not in resolved.parents or resolved == expected_parent:
        raise RuntimeError(f"Refusing to remove directory outside project data: {resolved}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def materialize(manifest: list[dict], output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite to rebuild it.")
        safe_clear_output(output_dir)
    for split in SPLITS:
        for label in LABELS:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    for row in manifest:
        source = Path(row["source_path"])
        destination = output_dir / row["split"] / row["label"] / row["destination_name"]
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    report_dir = args.reports.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning direct split folders under: {source_dir}")
    records, excluded = scan_images(source_dir)
    retained, duplicate_rows = deduplicate(records)
    manifest = assign_splits(retained)
    checks = validate_manifest(manifest)

    inventory_fields = [
        "source_path", "source_split", "label", "filename", "patient_id",
        "patient_id_source", "size_bytes", "sha256",
    ]
    write_csv(report_dir / "source_image_inventory.csv", records, inventory_fields)
    write_csv(
        report_dir / "excluded_synthetic_placeholders.csv",
        excluded,
        ["source_path", "source_split", "label", "filename", "size_bytes", "reason"],
    )
    write_csv(
        report_dir / "exact_duplicate_audit.csv",
        duplicate_rows,
        inventory_fields + ["duplicate_group_size", "retained", "retained_source_path"],
    )
    write_csv(
        report_dir / "patient_id_fallback_log.csv",
        [row for row in records if row["patient_id_source"] == "fallback"],
        inventory_fields,
    )
    write_csv(
        report_dir / "patient_split_manifest.csv",
        manifest,
        inventory_fields + ["split", "destination_name"],
    )
    write_csv(report_dir / "leakage_check.csv", checks, ["check", "key", "status", "splits"])

    summary_rows: list[dict] = []
    for split in SPLITS:
        split_rows = [row for row in manifest if row["split"] == split]
        normal = sum(row["label"] == "NORMAL" for row in split_rows)
        pneumonia = sum(row["label"] == "PNEUMONIA" for row in split_rows)
        total = len(split_rows)
        summary_rows.append(
            {
                "split": split,
                "normal_images": normal,
                "pneumonia_images": pneumonia,
                "total_images": total,
                "normal_ratio": f"{normal / total:.6f}",
                "pneumonia_ratio": f"{pneumonia / total:.6f}",
            }
        )
    write_csv(
        report_dir / "patient_split_summary.csv",
        summary_rows,
        ["split", "normal_images", "pneumonia_images", "total_images", "normal_ratio", "pneumonia_ratio"],
    )

    removed = len(records) - len(retained)
    print(f"Synthetic placeholder images excluded: {len(excluded)}")
    print(f"Real source images: {len(records)}")
    print(f"Exact duplicate images removed: {removed}")
    print(f"Retained images: {len(retained)}")
    for row in summary_rows:
        print(
            f"{row['split']}: total={row['total_images']} NORMAL={row['normal_images']} "
            f"PNEUMONIA={row['pneumonia_images']} normal_ratio={row['normal_ratio']}"
        )

    failures = [row for row in checks if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(f"Leakage checks failed: {len(failures)}")
    if args.materialize:
        materialize(manifest, output_dir, args.overwrite)
        print(f"Materialized split: {output_dir}")
    else:
        print("Audit-only mode: no image files were materialized.")
    print("PASS")


if __name__ == "__main__":
    main()
