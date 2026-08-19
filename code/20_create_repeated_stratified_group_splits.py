"""
Create 5 repeated stratified patient/group splits for robustness analysis.

Default input:
  data/chest_xray_patient_split_dedup

Default output:
  data/repeated_group_splits/repeat_01
  ...
  data/repeated_group_splits/repeat_05

Each repeat:
  - merges all images from the input train/val/test folders
  - extracts patient_id from filenames
  - assigns patient groups exclusively to train/val/test
  - keeps NORMAL/PNEUMONIA ratios close to the full source pool
  - checks patient_id and SHA-256 leakage across splits
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
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = Path(os.environ.get("PNEUMONIA_REPEATED_SOURCE_DIR", BASE_DIR / "data" / "chest_xray_patient_split_dedup"))
DEFAULT_TARGET_ROOT = Path(os.environ.get("PNEUMONIA_REPEATED_TARGET_ROOT", BASE_DIR / "data" / "repeated_group_splits"))
REPORT_DIR = Path(os.environ.get("PNEUMONIA_REPEATED_REPORT_DIR", BASE_DIR / "reports" / "repeated_group_splits"))

SOURCE_SPLITS = ("train", "val", "test")
TARGET_SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEEDS = (42, 143, 244, 345, 446)


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    source_split: str
    label: str
    filename: str
    patient_id: str
    sha256: str


def extract_patient_id(filename: str, label: str) -> str:
    stem = Path(filename).stem
    if label == "PNEUMONIA":
        match = re.search(r"(person\d+)", stem, flags=re.IGNORECASE)
    else:
        match = re.search(r"(NORMAL2-IM-\d+|IM-\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return stem.upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def scan_source(source_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for source_split in SOURCE_SPLITS:
        for label in CLASSES:
            folder = source_dir / source_split / label
            if not folder.exists():
                raise FileNotFoundError(f"Missing source folder: {folder}")
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                records.append(
                    ImageRecord(
                        source_path=path,
                        source_split=source_split,
                        label=label,
                        filename=path.name,
                        patient_id=extract_patient_id(path.name, label),
                        sha256=file_sha256(path),
                    )
                )
    if not records:
        raise RuntimeError(f"No images found under {source_dir}")
    return records


def group_records(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.patient_id].append(record)
    return dict(groups)


def group_counts(records: list[ImageRecord]) -> Counter:
    counts = Counter({"NORMAL": 0, "PNEUMONIA": 0})
    counts.update(record.label for record in records)
    return counts


def split_groups(groups: dict[str, list[ImageRecord]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    total_counts = Counter()
    for records in groups.values():
        total_counts.update(record.label for record in records)

    target_counts = {
        split: {label: total_counts[label] * ratio for label in CLASSES}
        for split, ratio in SPLIT_TARGETS.items()
    }
    assigned_counts = {split: Counter({"NORMAL": 0, "PNEUMONIA": 0}) for split in TARGET_SPLITS}
    assigned_groups: dict[str, str] = {}

    def split_score(split: str, counts: Counter) -> float:
        trial = assigned_counts[split] + counts
        value = 0.0
        for label in CLASSES:
            target = max(target_counts[split][label], 1.0)
            value += ((trial[label] - target_counts[split][label]) / target) ** 2
        target_size = sum(target_counts[split].values())
        trial_size = sum(trial.values())
        value += 0.25 * ((trial_size - target_size) / max(target_size, 1.0)) ** 2
        return value

    pure_groups = {label: [] for label in CLASSES}
    mixed_groups = []
    for patient_id, records in groups.items():
        counts = group_counts(records)
        present = [label for label in CLASSES if counts[label] > 0]
        if len(present) == 1:
            pure_groups[present[0]].append((patient_id, records, counts[present[0]]))
        else:
            mixed_groups.append((patient_id, records, counts))

    def choose_for_single_class(label: str, count: int) -> str:
        remaining = {
            split: target_counts[split][label] - assigned_counts[split][label]
            for split in TARGET_SPLITS
        }
        positive_remaining = {split: value for split, value in remaining.items() if value > 0}
        if positive_remaining:
            best_remaining = max(positive_remaining.values())
            candidates = [split for split, value in positive_remaining.items() if abs(value - best_remaining) < 1e-12]
            return rng.choice(candidates)
        best_overfill = min(
            assigned_counts[split][label] + count - target_counts[split][label]
            for split in TARGET_SPLITS
        )
        candidates = [
            split
            for split in TARGET_SPLITS
            if abs(assigned_counts[split][label] + count - target_counts[split][label] - best_overfill) < 1e-12
        ]
        return rng.choice(candidates)

    def assign_mixed_item(patient_id: str, records: list[ImageRecord], counts: Counter) -> None:
        best_score = min(split_score(split, counts) for split in TARGET_SPLITS)
        candidates = [split for split in TARGET_SPLITS if abs(split_score(split, counts) - best_score) < 1e-12]
        chosen = rng.choice(candidates)
        assigned_groups[patient_id] = chosen
        assigned_counts[chosen].update(record.label for record in records)

    for label in CLASSES:
        items = pure_groups[label]
        rng.shuffle(items)
        items.sort(key=lambda item: item[2], reverse=True)
        for patient_id, records, count in items:
            chosen = choose_for_single_class(label, count)
            assigned_groups[patient_id] = chosen
            assigned_counts[chosen].update(record.label for record in records)

    rng.shuffle(mixed_groups)
    mixed_groups.sort(key=lambda item: len(item[1]), reverse=True)
    for patient_id, records, counts in mixed_groups:
        assign_mixed_item(patient_id, records, counts)

    return assigned_groups


def reset_repeat_dir(target_dir: Path) -> None:
    allowed_roots = [BASE_DIR.resolve(), DEFAULT_TARGET_ROOT.resolve()]
    if target_dir.exists():
        resolved = target_dir.resolve()
        allowed = any(str(resolved).lower().startswith(str(root).lower()) for root in allowed_roots)
        if not allowed:
            roots = ", ".join(str(root) for root in allowed_roots)
            raise RuntimeError(f"Refusing to remove directory outside allowed roots ({roots}): {resolved}")
        shutil.rmtree(target_dir)
    for split in TARGET_SPLITS:
        for label in CLASSES:
            (target_dir / split / label).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dest: Path) -> str:
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def materialize_split(records: list[ImageRecord], patient_to_split: dict[str, str], target_dir: Path) -> Counter:
    modes = Counter()
    used_destinations: set[Path] = set()
    for record in records:
        split = patient_to_split[record.patient_id]
        dest_dir = target_dir / split / record.label
        dest = dest_dir / f"{record.source_split}__{record.filename}"
        if dest in used_destinations or dest.exists():
            dest = dest_dir / f"{record.source_split}__{record.source_path.stem}__{record.sha256[:10]}{record.source_path.suffix.lower()}"
        modes[link_or_copy(record.source_path, dest)] += 1
        used_destinations.add(dest)
    return modes


def validate_repeat(records: list[ImageRecord], patient_to_split: dict[str, str], repeat_name: str) -> list[list[object]]:
    patient_to_splits: dict[str, set[str]] = defaultdict(set)
    sha_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = patient_to_split[record.patient_id]
        patient_to_splits[record.patient_id].add(split)
        sha_to_splits[record.sha256].add(split)

    rows: list[list[object]] = []
    for patient_id, splits in sorted(patient_to_splits.items()):
        rows.append([repeat_name, "patient_id", patient_id, ";".join(sorted(splits)), "PASS" if len(splits) == 1 else "FAIL"])
    for sha, splits in sorted(sha_to_splits.items()):
        rows.append([repeat_name, "sha256", sha, ";".join(sorted(splits)), "PASS" if len(splits) == 1 else "FAIL"])
    failures = [row for row in rows if row[-1] != "PASS"]
    if failures:
        raise RuntimeError(f"{repeat_name}: leakage detected in {len(failures)} records")
    return rows


def summarize_repeat(records: list[ImageRecord], patient_to_split: dict[str, str], repeat_name: str, seed: int) -> list[list[object]]:
    rows: list[list[object]] = []
    for split in TARGET_SPLITS:
        split_records = [record for record in records if patient_to_split[record.patient_id] == split]
        counts = Counter(record.label for record in split_records)
        total = sum(counts.values())
        patient_count = len({record.patient_id for record in split_records})
        rows.append(
            [
                repeat_name,
                seed,
                split,
                patient_count,
                counts["NORMAL"],
                counts["PNEUMONIA"],
                total,
                counts["NORMAL"] / total if total else 0.0,
                counts["PNEUMONIA"] / total if total else 0.0,
            ]
        )
    return rows


def patient_distribution(records: list[ImageRecord], patient_to_split: dict[str, str], repeat_name: str) -> list[list[object]]:
    rows: list[list[object]] = []
    by_patient = group_records(records)
    for patient_id, items in sorted(by_patient.items()):
        counts = Counter(record.label for record in items)
        rows.append([repeat_name, patient_id, patient_to_split[patient_id], counts["NORMAL"], counts["PNEUMONIA"], len(items)])
    return rows


def image_manifest(records: list[ImageRecord], patient_to_split: dict[str, str], repeat_name: str) -> list[list[object]]:
    rows: list[list[object]] = []
    for record in records:
        rows.append(
            [
                repeat_name,
                patient_to_split[record.patient_id],
                record.label,
                record.patient_id,
                record.sha256,
                record.filename,
                display_path(record.source_path),
            ]
        )
    return rows


def write_protocol(source_dir: Path, target_root: Path, seeds: list[int], records: list[ImageRecord]) -> None:
    counts = Counter(record.label for record in records)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "repeated_group_split_protocol.md").write_text(
        "\n".join(
            [
                "# Repeated stratified group split protocol",
                "",
                f"- Source pool: `{display_path(source_dir)}`",
                f"- Output root: `{display_path(target_root)}`",
                f"- Repeats: {len(seeds)}",
                f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
                f"- Total images per repeat: {len(records)}",
                f"- NORMAL images: {counts['NORMAL']}",
                f"- PNEUMONIA images: {counts['PNEUMONIA']}",
                "- Unit of splitting: patient_id/group, not image.",
                "- Target split ratio: train 70%, validation 15%, test 15%.",
                "- Use this as a robustness/sensitivity analysis; the exact-deduplicated patient split remains the primary analysis.",
                "- If generated with `--manifest-only`, use `repeated_group_image_manifest.csv` as the authoritative split definition.",
                "",
                "Suggested manuscript wording:",
                "",
                "As a robustness analysis, we generated five repeated stratified patient/group splits from the exact-deduplicated image pool. For each repeat, all images from a given patient/group were assigned exclusively to train, validation or test partitions, while preserving the NORMAL/PNEUMONIA class ratio as closely as possible. Patient/group and SHA-256 cross-split leakage checks were required to pass for every repeat.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--manifest-only", action="store_true", help="Write split manifests/reports without materializing image folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir if args.source_dir.is_absolute() else BASE_DIR / args.source_dir
    target_root = args.target_root if args.target_root.is_absolute() else BASE_DIR / args.target_root
    if args.repeats == 5 and args.seed_start == 42:
        seeds = list(DEFAULT_SEEDS)
    else:
        seeds = [args.seed_start + i * 101 for i in range(args.repeats)]

    print("=" * 72)
    print("Creating repeated stratified patient/group splits")
    print(f"Source: {source_dir}")
    print(f"Target: {target_root}")
    print(f"Seeds:  {seeds}")
    print("=" * 72)

    records = scan_source(source_dir)
    groups = group_records(records)
    print(f"Images: {len(records)}")
    print(f"Patient/groups: {len(groups)}")

    all_summary_rows: list[list[object]] = []
    all_patient_rows: list[list[object]] = []
    all_leakage_rows: list[list[object]] = []
    all_manifest_rows: list[list[object]] = []

    for index, seed in enumerate(seeds, start=1):
        repeat_name = f"repeat_{index:02d}"
        target_dir = target_root / repeat_name
        print(f"\n{repeat_name}: seed={seed}")
        patient_to_split = split_groups(groups, seed)
        modes = Counter()
        if not args.manifest_only:
            reset_repeat_dir(target_dir)
            modes = materialize_split(records, patient_to_split, target_dir)
        summary_rows = summarize_repeat(records, patient_to_split, repeat_name, seed)
        leakage_rows = validate_repeat(records, patient_to_split, repeat_name)
        patient_rows = patient_distribution(records, patient_to_split, repeat_name)
        manifest_rows = image_manifest(records, patient_to_split, repeat_name)
        all_summary_rows.extend(summary_rows)
        all_leakage_rows.extend(leakage_rows)
        all_patient_rows.extend(patient_rows)
        all_manifest_rows.extend(manifest_rows)
        for row in summary_rows:
            _, _, split, patients, normal, pneumonia, total, normal_ratio, pneumonia_ratio = row
            print(
                f"  {split:5s} patients={patients:4d} total={total:4d} "
                f"NORMAL={normal:4d} ({normal_ratio:.3f}) "
                f"PNEUMONIA={pneumonia:4d} ({pneumonia_ratio:.3f})"
            )
        if args.manifest_only:
            print("  files: manifest only")
        else:
            print(f"  files: {dict(modes)}")
        print("  PASS: no patient_id or SHA-256 cross-split leakage")

    write_csv(
        REPORT_DIR / "repeated_group_split_summary.csv",
        ["repeat", "seed", "split", "patient_groups", "normal_images", "pneumonia_images", "total_images", "normal_ratio", "pneumonia_ratio"],
        all_summary_rows,
    )
    write_csv(
        REPORT_DIR / "repeated_group_patient_distribution.csv",
        ["repeat", "patient_id", "assigned_split", "normal_images", "pneumonia_images", "total_images"],
        all_patient_rows,
    )
    write_csv(
        REPORT_DIR / "repeated_group_image_manifest.csv",
        ["repeat", "assigned_split", "class", "patient_id", "sha256", "filename", "source_path"],
        all_manifest_rows,
    )
    write_csv(
        REPORT_DIR / "repeated_group_leakage_check.csv",
        ["repeat", "check_type", "key", "assigned_splits", "status"],
        all_leakage_rows,
    )
    write_protocol(source_dir, target_root, seeds, records)
    print("\nPASS: all repeated stratified group splits created without leakage.")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
