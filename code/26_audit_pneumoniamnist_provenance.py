#!/usr/bin/env python3
"""Audit PneumoniaMNIST provenance and map its test images to Kermany sources.

This script is deliberately independent of model training.  It uses the clean
2026-07-06 source whitelist and split manifest, the official MedMNIST ID mapping,
and conservative image reconstruction criteria.  It never optimizes a model
threshold or temperature on PneumoniaMNIST.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import pairwise_distances
from skimage.metrics import structural_similarity
from skimage.transform import resize as sk_resize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INVENTORY = ROOT / "04_reports/clean_rebuild_20260706/source_image_inventory.csv"
CLEAN_MANIFEST = ROOT / "04_reports/clean_rebuild_20260706/patient_split_manifest.csv"
EXACT_DUPLICATE_AUDIT = ROOT / "04_reports/clean_rebuild_20260706/exact_duplicate_audit.csv"
PNEUMONIAMNIST_NPZ = ROOT / "01_data/external/pneumoniamnist.npz"
OFFICIAL_ID_MAPPING = (
    ROOT
    / "01_data/manifests/medmnist_official_id_mapping/pneumoniamnist_split_info.csv"
)
CLEAN_DATA_DIR = ROOT / "01_data/chest_xray_patient_split_dedup_clean"
REPEATED_SPLIT_DIR = ROOT / "01_data/clean_repeated_group_splits"
REPORT_DIR = ROOT / "04_reports"
FIGURE_DIR = REPORT_DIR / "figures"
SOURCE_DISJOINT_MANIFEST = (
    ROOT / "01_data/manifests/pneumoniamnist_source_disjoint_stress_test.csv"
)

RESNET_CHECKPOINT = (
    ROOT
    / "03_models_and_outputs/output_clean_20260706/resnet18/best_resnet18_nomixup.pth"
)
MOBILE_CHECKPOINT = (
    ROOT
    / "03_models_and_outputs/output_clean_20260706/baselines/mobilenet_v3/best_mobilenet_v3.pth"
)
OLD_STRESS_RESULTS = (
    ROOT
    / "03_models_and_outputs/output_clean_20260706/external_pneumoniamnist/pneumoniamnist_results.pth"
)
CALIBRATION_TABLE = (
    ROOT / "04_reports/final_tables/clean_20260706_calibration_metrics.csv"
)
THRESHOLD_TABLE = (
    ROOT / "04_reports/final_tables/clean_20260706_validation_selected_thresholds.csv"
)
LEGACY_STRESS_SCRIPT = ROOT / "02_scripts/external_validate_pneumoniamnist.py"

OFFICIAL_MAPPING_URL = (
    "https://drive.google.com/drive/folders/"
    "1A_99qH_c-J0p_SatwSiaP_i1CvLUOzVo?usp=sharing"
)
OFFICIAL_MEDMNIST_URL = "https://github.com/MedMNIST/MedMNIST"


# These thresholds are intentionally visible and are written verbatim to the report.
# They were fixed before the final mapping run and must not be relaxed to preserve a
# desired sample count or manuscript conclusion.
MATCH_THRESHOLDS = {
    "confirmed": {
        "mutual_top1": True,
        "subtype_match": True,
        "max_mse": 100.0,
        "max_mae": 8.0,
        "min_ncc": 0.985,
        "min_ssim": 0.950,
        "max_phash_distance": 12,
        "max_dhash_distance": 14,
        "min_second_to_first_mse_ratio": 1.20,
    },
    "probable": {
        "subtype_match": True,
        "max_mse": 400.0,
        "max_mae": 15.0,
        "min_ncc": 0.950,
        "min_ssim": 0.850,
        "max_phash_distance": 18,
        "max_dhash_distance": 20,
        "min_second_to_first_mse_ratio": 1.10,
        "requires_mutual_or_strong_margin": True,
    },
    "ambiguous": {
        "min_ncc": 0.800,
        "min_ssim": 0.650,
    },
}

PRIMARY_METHODS = (
    "direct_pil_bilinear",
    "aspect_fill_pil_bilinear",
    "center_crop_pil_bilinear",
    "center_crop_pil_bicubic",
    "center_crop_cv_area",
    "center_crop_skimage_linear",
)
DIAGNOSTIC_METHODS = (
    "center_crop_skimage_minmax",
    "center_crop_skimage_inverted",
)


@dataclass(frozen=True)
class MatchMetrics:
    mae: float
    mse: float
    ncc: float
    ssim: float
    ahash_distance: int
    phash_distance: int
    dhash_distance: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def windows_path_to_local(value: str) -> Path:
    """Translate an existing Windows drive path to WSL when necessary."""
    path = str(value)
    if os.name != "nt" and len(path) >= 3 and path[1:3] == ":\\":
        path = f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return Path(path)


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    suffixes = {".jpeg", ".jpg", ".png"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def read_gray(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").copy()


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def transform_28(image: Image.Image, method: str) -> np.ndarray:
    if method == "direct_pil_bilinear":
        result = image.resize((28, 28), Image.Resampling.BILINEAR)
        return np.asarray(result, dtype=np.uint8)

    if method == "aspect_fill_pil_bilinear":
        result = ImageOps.fit(
            image,
            (28, 28),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )
        return np.asarray(result, dtype=np.uint8)

    cropped = center_square(image)
    if method == "center_crop_pil_bilinear":
        return np.asarray(
            cropped.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.uint8
        )
    if method == "center_crop_pil_bicubic":
        return np.asarray(
            cropped.resize((28, 28), Image.Resampling.BICUBIC), dtype=np.uint8
        )

    array = np.asarray(cropped, dtype=np.uint8)
    if method == "center_crop_cv_area":
        return cv2.resize(array, (28, 28), interpolation=cv2.INTER_AREA).astype(np.uint8)

    resized = sk_resize(
        array,
        (28, 28),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )
    if method == "center_crop_skimage_linear":
        return np.rint(resized).clip(0, 255).astype(np.uint8)
    if method == "center_crop_skimage_minmax":
        low = float(resized.min())
        high = float(resized.max())
        if high <= low:
            return np.zeros((28, 28), dtype=np.uint8)
        return np.rint((resized - low) * (255.0 / (high - low))).astype(np.uint8)
    if method == "center_crop_skimage_inverted":
        return 255 - np.rint(resized).clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unknown transform method: {method}")


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel().astype(bool):
        value = (value << 1) | int(bit)
    return value


def ahash(array: np.ndarray) -> int:
    small = cv2.resize(array, (8, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_int(small >= float(small.mean()))


def dhash(array: np.ndarray) -> int:
    small = cv2.resize(array, (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_int(small[:, 1:] >= small[:, :-1])


def phash(array: np.ndarray) -> int:
    small = cv2.resize(array, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    median = float(np.median(dct.ravel()[1:]))
    return _bits_to_int(dct >= median)


def hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def detailed_metrics(query: np.ndarray, candidate: np.ndarray) -> MatchMetrics:
    query_f = query.astype(np.float64)
    candidate_f = candidate.astype(np.float64)
    difference = query_f - candidate_f
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    query_z = query_f - query_f.mean()
    candidate_z = candidate_f - candidate_f.mean()
    denominator = float(np.linalg.norm(query_z) * np.linalg.norm(candidate_z))
    ncc = float(np.dot(query_z.ravel(), candidate_z.ravel()) / denominator) if denominator else 0.0
    ssim = float(structural_similarity(query, candidate, data_range=255))
    return MatchMetrics(
        mae=mae,
        mse=mse,
        ncc=ncc,
        ssim=ssim,
        ahash_distance=hamming(ahash(query), ahash(candidate)),
        phash_distance=hamming(phash(query), phash(candidate)),
        dhash_distance=hamming(dhash(query), dhash(candidate)),
    )


def source_subtype(label: str, filename: str) -> str:
    if str(label).upper() == "NORMAL":
        return "NORMAL"
    lower = str(filename).lower()
    if "bacteria" in lower:
        return "BACTERIA"
    if "virus" in lower:
        return "VIRUS"
    return "PNEUMONIA_UNSPECIFIED"


def official_subtype(image_id: str, label: int) -> str:
    prefix = str(image_id).split("-", 1)[0].upper()
    if prefix in {"NORMAL", "BACTERIA", "VIRUS"}:
        return prefix
    return "NORMAL" if int(label) == 0 else "PNEUMONIA_UNSPECIFIED"


def classify_match(
    metrics: MatchMetrics,
    mutual: bool,
    subtype_matches: bool,
    margin_ratio: float,
    method: str,
) -> str:
    if method.endswith("inverted"):
        return "ambiguous_match"

    confirmed = MATCH_THRESHOLDS["confirmed"]
    if (
        mutual
        and subtype_matches
        and metrics.mse <= confirmed["max_mse"]
        and metrics.mae <= confirmed["max_mae"]
        and metrics.ncc >= confirmed["min_ncc"]
        and metrics.ssim >= confirmed["min_ssim"]
        and metrics.phash_distance <= confirmed["max_phash_distance"]
        and metrics.dhash_distance <= confirmed["max_dhash_distance"]
        and margin_ratio >= confirmed["min_second_to_first_mse_ratio"]
    ):
        return "confirmed_match"

    probable = MATCH_THRESHOLDS["probable"]
    if (
        subtype_matches
        and metrics.mse <= probable["max_mse"]
        and metrics.mae <= probable["max_mae"]
        and metrics.ncc >= probable["min_ncc"]
        and metrics.ssim >= probable["min_ssim"]
        and metrics.phash_distance <= probable["max_phash_distance"]
        and metrics.dhash_distance <= probable["max_dhash_distance"]
        and margin_ratio >= probable["min_second_to_first_mse_ratio"]
        and (mutual or margin_ratio >= 1.50)
    ):
        return "probable_match"

    ambiguous = MATCH_THRESHOLDS["ambiguous"]
    if metrics.ncc >= ambiguous["min_ncc"] and metrics.ssim >= ambiguous["min_ssim"]:
        return "ambiguous_match"
    return "unmatched"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    required = [SOURCE_INVENTORY, CLEAN_MANIFEST, EXACT_DUPLICATE_AUDIT, PNEUMONIAMNIST_NPZ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required clean inputs:\n" + "\n".join(missing))

    source = pd.read_csv(SOURCE_INVENTORY, encoding="utf-8-sig")
    clean = pd.read_csv(CLEAN_MANIFEST, encoding="utf-8-sig")
    duplicate = pd.read_csv(EXACT_DUPLICATE_AUDIT, encoding="utf-8-sig")
    med = dict(np.load(PNEUMONIAMNIST_NPZ))
    if OFFICIAL_ID_MAPPING.exists():
        official = pd.read_csv(OFFICIAL_ID_MAPPING)
    else:
        official = pd.DataFrame(columns=["split", "index", "image_id"])

    if len(source) != 5856:
        raise ValueError(f"Expected 5,856 clean-whitelisted source images, found {len(source)}")
    if len(clean) != 5824:
        raise ValueError(f"Expected 5,824 exact-deduplicated images, found {len(clean)}")
    expected_shapes = {
        "train_images": (4708, 28, 28),
        "val_images": (524, 28, 28),
        "test_images": (624, 28, 28),
    }
    for key, expected in expected_shapes.items():
        if med[key].shape != expected:
            raise ValueError(f"Unexpected {key} shape: {med[key].shape}; expected {expected}")
    return source, clean, duplicate, med, official


def enrich_source_inventory(
    source: pd.DataFrame, clean: pd.DataFrame, duplicate: pd.DataFrame
) -> pd.DataFrame:
    clean_by_source = clean.set_index("source_path").to_dict("index")
    duplicate_by_source = duplicate.set_index("source_path").to_dict("index")
    rows = []
    for row in source.to_dict("records"):
        source_path = row["source_path"]
        retained_source = duplicate_by_source.get(source_path, {}).get(
            "retained_source_path", source_path
        )
        clean_row = clean_by_source.get(source_path) or clean_by_source.get(retained_source)
        local_path = windows_path_to_local(source_path)
        width = height = None
        mode = None
        error = ""
        try:
            with Image.open(local_path) as image:
                width, height = image.size
                mode = image.mode
        except Exception as exc:  # recorded rather than hidden
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                **row,
                "local_path": str(local_path),
                "file_exists": local_path.exists(),
                "width": width,
                "height": height,
                "image_mode": mode,
                "image_read_error": error,
                "source_subtype": source_subtype(row["label"], row["filename"]),
                "retained_source_path": retained_source,
                "internal_split": clean_row.get("split") if clean_row else "",
                "internal_patient_id": clean_row.get("patient_id") if clean_row else "",
                "destination_name": clean_row.get("destination_name") if clean_row else "",
                "retained_after_exact_dedup": bool(clean_row),
            }
        )
    return pd.DataFrame(rows)


def med_sample_inventory(med: dict[str, np.ndarray], official: pd.DataFrame) -> pd.DataFrame:
    official_lookup = {
        (str(row.split), int(row.index)): str(row.image_id)
        for row in official.itertuples(index=False)
    }
    rows = []
    for split in ("train", "val", "test"):
        images = med[f"{split}_images"]
        labels = med[f"{split}_labels"].reshape(-1)
        for index, (image, label) in enumerate(zip(images, labels, strict=True)):
            image_id = official_lookup.get((split, index), "")
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "label": int(label),
                    "label_name": "PNEUMONIA" if int(label) else "NORMAL",
                    "official_image_id": image_id,
                    "official_subtype": official_subtype(image_id, int(label)),
                    "shape": "x".join(str(value) for value in image.shape),
                    "pixel_min": int(image.min()),
                    "pixel_max": int(image.max()),
                    "dtype": str(image.dtype),
                    "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def build_asset_inventory(source: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_file(name: str, category: str, path: Path, details: str = "") -> None:
        exists = path.exists()
        rows.append(
            {
                "asset": name,
                "category": category,
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else "",
                "sha256": sha256_file(path) if exists and path.is_file() else "",
                "count": "",
                "details": details,
            }
        )

    def add_dir(name: str, category: str, path: Path, count: int | str, details: str = "") -> None:
        rows.append(
            {
                "asset": name,
                "category": category,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": "",
                "sha256": "",
                "count": count,
                "details": details,
            }
        )

    first_source = windows_path_to_local(str(source.iloc[0]["source_path"]))
    source_root = first_source.parents[2] if len(first_source.parents) >= 3 else first_source.parent
    add_dir(
        "Kermany real source whitelist",
        "data",
        source_root,
        len(source),
        "5,856 rows are defined by source_image_inventory.csv; the surrounding E: directory contains nested copies and excluded synthetic placeholders and must not be scanned as an unrestricted source.",
    )
    add_dir(
        "Exact-deduplicated clean primary split",
        "data",
        CLEAN_DATA_DIR,
        len(image_files(CLEAN_DATA_DIR)),
        "Expected 5,824 images.",
    )
    add_dir(
        "Five repeated clean group splits",
        "data",
        REPEATED_SPLIT_DIR,
        len(image_files(REPEATED_SPLIT_DIR)),
        "Expected 29,120 image copies across five repeats.",
    )
    add_file(
        "Combined train/validation/test manifest",
        "manifest",
        CLEAN_MANIFEST,
        f"{len(clean)} rows; fields include split, patient_id, and sha256.",
    )
    add_file("Kermany source SHA-256 inventory", "manifest", SOURCE_INVENTORY, f"{len(source)} rows.")
    add_file("Exact duplicate audit", "manifest", EXACT_DUPLICATE_AUDIT, "Records the 32 removed exact duplicate images.")
    for split in ("train", "val", "test"):
        add_file(
            f"Standalone {split} manifest",
            "manifest",
            ROOT / f"01_data/manifests/{split}_manifest.csv",
            "Not required because the combined clean manifest has an explicit split column; listed to make absence explicit.",
        )
    add_file("ResNet18 clean checkpoint", "model", RESNET_CHECKPOINT, "Frozen formal clean checkpoint.")
    add_file("MobileNetV3-Small clean checkpoint", "model", MOBILE_CHECKPOINT, "Frozen formal clean checkpoint.")
    add_file("PneumoniaMNIST 28x28 package", "stress_input", PNEUMONIAMNIST_NPZ, "Official MD5 expected: 28209eda62fecd6e6a2d98b1501bb15f.")
    add_file("Official MedMNIST source-ID mapping", "provenance", OFFICIAL_ID_MAPPING, f"Downloaded from {OFFICIAL_MAPPING_URL}.")
    add_file("Prior 624-image predictions", "historical_result", OLD_STRESS_RESULTS, "Superseded pending provenance audit; retained unchanged.")
    add_file("Internal validation thresholds", "configuration", THRESHOLD_TABLE, "ResNet18 0.450; MobileNetV3-Small 0.345.")
    add_file("Internal validation temperature parameters", "configuration", CALIBRATION_TABLE, "Temperature values are stored in the generated calibration table, not in a separate serialized parameter file.")
    add_file("Legacy fixed-0.5 stress-test script", "code", LEGACY_STRESS_SCRIPT, "Historical script; terminology and inference policy require correction.")
    add_file("Serialized temperature parameter file", "configuration", ROOT / "03_models_and_outputs/output_clean_20260706/calibration/temperatures.json", "Missing; authoritative values are recoverable from the clean calibration table.")
    return pd.DataFrame(rows)


def write_asset_inventory(table: pd.DataFrame) -> None:
    csv_path = REPORT_DIR / "pneumoniamnist_asset_inventory.csv"
    md_path = REPORT_DIR / "pneumoniamnist_asset_inventory.md"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    present = int(table["exists"].astype(bool).sum())
    missing = table.loc[~table["exists"].astype(bool), ["asset", "path", "details"]]
    columns = list(table.columns)
    markdown_rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in table.fillna("").astype(str).to_dict("records"):
        markdown_rows.append(
            "| "
            + " | ".join(record[column].replace("|", "\\|") for column in columns)
            + " |"
        )
    lines = [
        "# PneumoniaMNIST asset inventory",
        "",
        "> Terminology status: PneumoniaMNIST is a **same-source preprocessing and resolution-shift stress test**, not an independent external cohort.",
        "",
        f"- Assets present: {present}/{len(table)}",
        f"- Assets explicitly missing: {len(missing)}",
        "- Formal result provenance is restricted to `clean_20260706`, `clean_20260707`, and the formal clean data/manifests.",
        "",
        "## Inventory",
        "",
        *markdown_rows,
        "",
        "## Missing files",
        "",
    ]
    if missing.empty:
        lines.append("None.")
    else:
        for row in missing.itertuples(index=False):
            lines.append(f"- `{row.asset}`: `{row.path}`. {row.details}")
    lines.extend(
        [
            "",
            "## Provenance warning",
            "",
            "PneumoniaMNIST was derived from the same underlying Kermany pediatric chest radiograph collection and therefore does not constitute an independent external cohort.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_transform_cache(source_test: pd.DataFrame) -> dict[str, np.ndarray]:
    cache = {method: [] for method in (*PRIMARY_METHODS, *DIAGNOSTIC_METHODS)}
    for row in source_test.itertuples(index=False):
        path = Path(row.local_path)
        image = read_gray(path)
        for method in cache:
            cache[method].append(transform_28(image, method))
    return {method: np.stack(arrays) for method, arrays in cache.items()}


def compute_mapping(
    source_enriched: pd.DataFrame,
    med_samples: pd.DataFrame,
    med: dict[str, np.ndarray],
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_test = source_enriched[source_enriched["source_split"].eq("test")].reset_index(drop=True)
    med_test = med_samples[med_samples["split"].eq("test")].sort_values("index").reset_index(drop=True)
    med_images = med["test_images"]
    if len(source_test) != 624 or len(med_test) != 624:
        raise ValueError("The official Kermany/PneumoniaMNIST test pools must each contain 624 images.")

    cache = prepare_transform_cache(source_test)
    mapping_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for subtype in ("NORMAL", "BACTERIA", "VIRUS"):
        query_positions = np.flatnonzero(med_test["official_subtype"].eq(subtype).to_numpy())
        source_positions = np.flatnonzero(source_test["source_subtype"].eq(subtype).to_numpy())
        if len(query_positions) != len(source_positions):
            raise ValueError(
                f"Official subtype pool mismatch for {subtype}: "
                f"PneumoniaMNIST={len(query_positions)}, Kermany={len(source_positions)}"
            )

        query_flat = med_images[query_positions].reshape(len(query_positions), -1).astype(np.float32)
        best_mse = np.full((len(query_positions), len(source_positions)), np.inf, dtype=np.float64)
        best_method = np.zeros(best_mse.shape, dtype=np.int16)
        for method_index, method in enumerate(PRIMARY_METHODS):
            source_flat = cache[method][source_positions].reshape(len(source_positions), -1).astype(np.float32)
            mse = pairwise_distances(query_flat, source_flat, metric="sqeuclidean") / (28 * 28)
            update = mse < best_mse
            best_mse[update] = mse[update]
            best_method[update] = method_index

        forward_best = np.argmin(best_mse, axis=1)
        reverse_best = np.argmin(best_mse, axis=0)

        for local_query_index, query_position in enumerate(query_positions):
            order = np.argsort(best_mse[local_query_index])[:top_k]
            first_mse = float(best_mse[local_query_index, order[0]])
            second_mse = float(best_mse[local_query_index, order[1]]) if len(order) > 1 else math.inf
            margin_ratio = second_mse / max(first_mse, 1e-12)
            query_array = med_images[query_position]
            query_meta = med_test.iloc[query_position]

            for rank, local_source_index in enumerate(order, start=1):
                source_position = source_positions[local_source_index]
                source_meta = source_test.iloc[source_position]
                method = PRIMARY_METHODS[int(best_method[local_query_index, local_source_index])]
                candidate_array = cache[method][source_position]
                metrics = detailed_metrics(query_array, candidate_array)
                mutual = bool(
                    rank == 1
                    and forward_best[local_query_index] == local_source_index
                    and reverse_best[local_source_index] == local_query_index
                )
                subtype_matches = query_meta["official_subtype"] == source_meta["source_subtype"]
                status = (
                    classify_match(metrics, mutual, subtype_matches, margin_ratio, method)
                    if rank == 1
                    else "candidate"
                )
                minmax_mse = float(
                    np.mean(
                        np.square(
                            query_array.astype(np.float64)
                            - cache["center_crop_skimage_minmax"][source_position].astype(np.float64)
                        )
                    )
                )
                inverted_mse = float(
                    np.mean(
                        np.square(
                            query_array.astype(np.float64)
                            - cache["center_crop_skimage_inverted"][source_position].astype(np.float64)
                        )
                    )
                )
                row = {
                    "pneumoniamnist_split": "test",
                    "pneumoniamnist_index": int(query_meta["index"]),
                    "pneumoniamnist_label": int(query_meta["label"]),
                    "pneumoniamnist_label_name": query_meta["label_name"],
                    "official_image_id": query_meta["official_image_id"],
                    "official_subtype": query_meta["official_subtype"],
                    "candidate_rank": rank,
                    "mapping_status": status,
                    "source_path": source_meta["source_path"],
                    "source_filename": source_meta["filename"],
                    "source_label": source_meta["label"],
                    "source_subtype": source_meta["source_subtype"],
                    "source_original_split": source_meta["source_split"],
                    "source_sha256": source_meta["sha256"],
                    "patient_id": source_meta["patient_id"],
                    "internal_split": source_meta["internal_split"],
                    "internal_patient_id": source_meta["internal_patient_id"],
                    "destination_name": source_meta["destination_name"],
                    "retained_after_exact_dedup": source_meta["retained_after_exact_dedup"],
                    "best_reconstruction_method": method,
                    "mutual_top1": mutual,
                    "second_to_first_mse_ratio": margin_ratio if rank == 1 else "",
                    "pixel_mae": metrics.mae,
                    "pixel_mse": metrics.mse,
                    "normalized_cross_correlation": metrics.ncc,
                    "ssim": metrics.ssim,
                    "ahash_distance": metrics.ahash_distance,
                    "phash_distance": metrics.phash_distance,
                    "dhash_distance": metrics.dhash_distance,
                    "diagnostic_minmax_mse": minmax_mse,
                    "diagnostic_inverted_mse": inverted_mse,
                    "official_source_split_constraint_used": True,
                    "embedding_similarity_used": False,
                }
                mapping_rows.append(row)
                if rank == 1:
                    best_rows.append(row)

    mapping = pd.DataFrame(mapping_rows).sort_values(
        ["pneumoniamnist_index", "candidate_rank"]
    )
    best = pd.DataFrame(best_rows).sort_values("pneumoniamnist_index")
    return mapping, best


def summarize_mapping(best: pd.DataFrame, clean: pd.DataFrame, official_available: bool) -> dict[str, object]:
    accepted = best[best["mapping_status"].isin(["confirmed_match", "probable_match"])].copy()
    status_counts = best["mapping_status"].value_counts().to_dict()
    by_internal_split = (
        accepted["internal_split"].replace("", "not_retained").value_counts().to_dict()
    )
    groups_by_split = {
        split: int(accepted.loc[accepted["internal_split"].eq(split), "patient_id"].nunique())
        for split in ("train", "val", "test")
    }
    source_reuse = accepted["source_path"].value_counts()
    patient_to_splits = clean.groupby("patient_id")["split"].nunique()
    label_conflicts = int(
        (
            (best["pneumoniamnist_label"].eq(0) & ~best["source_label"].eq("NORMAL"))
            | (best["pneumoniamnist_label"].eq(1) & ~best["source_label"].eq("PNEUMONIA"))
        ).sum()
    )
    return {
        "terminology": "same-source preprocessing and resolution-shift stress test",
        "independent_external_cohort": False,
        "official_same_source_provenance": True,
        "official_mapping_available": official_available,
        "pneumoniamnist_test_n": int(len(best)),
        "mapping_status_counts": {key: int(value) for key, value in status_counts.items()},
        "accepted_source_image_matches": int(len(accepted)),
        "accepted_source_image_match_fraction": float(len(accepted) / len(best)),
        "accepted_matches_by_internal_split": {
            key if key else "not_retained": int(value) for key, value in by_internal_split.items()
        },
        "accepted_patient_groups_by_internal_split": groups_by_split,
        "ambiguous_n": int(status_counts.get("ambiguous_match", 0)),
        "unmatched_n": int(status_counts.get("unmatched", 0)),
        "label_conflict_n": label_conflicts,
        "one_source_to_multiple_pneumoniamnist_n": int((source_reuse > 1).sum()),
        "multiple_source_to_one_pneumoniamnist_n": 0,
        "clean_manifest_patient_cross_split_groups": int((patient_to_splits > 1).sum()),
        "byte_exact_overlap_assessment": "Not applicable after the documented 28x28 transformation; source-image identity is inferred from official provenance plus reconstruction metrics.",
        "limitations": [
            "The official mapping uses MedMNIST source image IDs, whereas the local Kermany copy uses Kaggle-style filenames; local file identity therefore requires image reconstruction matching.",
            "Failure to find a high-confidence local match is not evidence that source overlap is absent.",
            "Embedding similarity was not used because official split/index metadata plus pixel, SSIM, NCC, and perceptual-hash evidence were available; ambiguous samples remain unforced.",
        ],
        "match_thresholds": MATCH_THRESHOLDS,
        "primary_reconstruction_methods": list(PRIMARY_METHODS),
        "diagnostic_reconstruction_methods": list(DIAGNOSTIC_METHODS),
    }


def source_disjoint_subset(best: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    train_groups = set(clean.loc[clean["split"].eq("train"), "patient_id"].astype(str))
    val_groups = set(clean.loc[clean["split"].eq("val"), "patient_id"].astype(str))
    subset = best[
        best["mapping_status"].eq("confirmed_match")
        & best["internal_split"].eq("test")
        & ~best["patient_id"].astype(str).isin(train_groups)
        & ~best["patient_id"].astype(str).isin(val_groups)
        & best["retained_after_exact_dedup"].astype(bool)
    ].copy()
    subset["checkpoint_selection_involved_source"] = False
    subset["threshold_source"] = "internal_validation_only"
    subset["pneumoniamnist_threshold_optimization"] = False
    subset["temperature_source"] = "internal_validation_only"
    subset["pneumoniamnist_temperature_refit"] = False
    subset["stress_test_role"] = "same-source preprocessing and resolution-shift stress test"
    return subset.sort_values("pneumoniamnist_index")


def create_match_figure(best: pd.DataFrame, med: dict[str, np.ndarray], path: Path) -> None:
    confirmed = best[best["mapping_status"].eq("confirmed_match")].copy()
    if confirmed.empty:
        return
    selected_parts = []
    for subtype, n in (("NORMAL", 7), ("BACTERIA", 8), ("VIRUS", 5)):
        group = confirmed[confirmed["official_subtype"].eq(subtype)].sort_values("ssim", ascending=False)
        selected_parts.append(group.head(n))
    selected = pd.concat(selected_parts).head(20).reset_index(drop=True)

    fig, axes = plt.subplots(5, 12, figsize=(24, 13), constrained_layout=True)
    for axis in axes.ravel():
        axis.axis("off")
    for item_index, row in selected.iterrows():
        grid_row = item_index // 4
        grid_col = (item_index % 4) * 3
        query = med["test_images"][int(row["pneumoniamnist_index"])]
        source_path = windows_path_to_local(str(row["source_path"]))
        original = read_gray(source_path)
        reconstructed = transform_28(original, str(row["best_reconstruction_method"]))
        panels = [query, np.asarray(original), reconstructed]
        labels = ["PneumoniaMNIST 28x28", "Kermany source", "Reconstructed 28x28"]
        for offset, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            axis = axes[grid_row, grid_col + offset]
            axis.imshow(panel, cmap="gray", vmin=0, vmax=255)
            axis.axis("off")
            if offset == 0:
                axis.set_title(
                    f"{label}\n{row['official_subtype']} | internal={row['internal_split']}\n"
                    f"group={row['patient_id']} | {row['mapping_status']}\n"
                    f"SSIM={row['ssim']:.3f}, MSE={row['pixel_mse']:.1f}",
                    fontsize=6.2,
                )
            else:
                axis.set_title(label, fontsize=8)
    fig.suptitle(
        "PneumoniaMNIST to Kermany source mapping examples (same-source stress test)",
        fontsize=16,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_audit_report(summary: dict[str, object], subset: pd.DataFrame) -> None:
    counts = summary["mapping_status_counts"]
    by_split = summary["accepted_matches_by_internal_split"]
    groups = summary["accepted_patient_groups_by_internal_split"]
    decision = "A" if len(subset) > 0 else "B"
    lines = [
        "# PneumoniaMNIST provenance and source-overlap audit",
        "",
        "## Immediate correction",
        "",
        "PneumoniaMNIST was derived from the same underlying Kermany pediatric chest radiograph collection and therefore does not constitute an independent external cohort.",
        "",
        "The analysis is termed a **same-source preprocessing and resolution-shift stress test**. It does not test cross-institutional or independent-patient clinical generalization.",
        "",
        "## Evidence base",
        "",
        f"- Official MedMNIST project: {OFFICIAL_MEDMNIST_URL}",
        f"- Official source-ID mapping folder: {OFFICIAL_MAPPING_URL}",
        "- The official metadata states that PneumoniaMNIST is based on the 5,856-image Kermany pediatric chest radiograph collection, center-cropped and resized to 28x28.",
        "- The official mapping contains 5,856 rows and 624 test rows. The test subtype counts are 234 NORMAL, 242 BACTERIA, and 148 VIRUS, exactly matching the local Kermany source test pool.",
        "- Local filenames differ from the official MedMNIST image IDs, so mapping to local files was independently checked with multiple documented 28x28 reconstructions, pixel distances, NCC, SSIM, aHash, pHash, dHash, and bidirectional top-1 consistency.",
        "",
        "## Mapping thresholds",
        "",
        "```json",
        json.dumps(MATCH_THRESHOLDS, indent=2, ensure_ascii=False),
        "```",
        "",
        "## PneumoniaMNIST test mapping",
        "",
        f"- Total: {summary['pneumoniamnist_test_n']}",
        f"- Confirmed: {counts.get('confirmed_match', 0)}",
        f"- Probable: {counts.get('probable_match', 0)}",
        f"- Ambiguous: {counts.get('ambiguous_match', 0)}",
        f"- Unmatched: {counts.get('unmatched', 0)}",
        f"- Label conflicts: {summary['label_conflict_n']}",
        f"- One source image mapped to multiple accepted PneumoniaMNIST samples: {summary['one_source_to_multiple_pneumoniamnist_n']}",
        "",
        "Accepted confirmed/probable source-image matches by internal clean split:",
        "",
        f"- Internal train: {by_split.get('train', 0)} images, {groups.get('train', 0)} patient/groups",
        f"- Internal validation: {by_split.get('val', 0)} images, {groups.get('val', 0)} patient/groups",
        f"- Internal test: {by_split.get('test', 0)} images, {groups.get('test', 0)} patient/groups",
        f"- Not retained after exact deduplication or unresolved: {by_split.get('not_retained', 0)}",
        "",
        "No-match warning: not finding a high-confidence local mapping does not prove that overlap is absent.",
        "",
        "## Source-disjoint stress-test decision",
        "",
        f"Decision: **Scheme {decision}**.",
        "",
    ]
    if len(subset):
        lines.extend(
            [
                f"A conservative source-disjoint stress-test subset was created with {len(subset)} samples. Every retained sample is a confirmed local source match, belongs to the internal held-out test split, has no patient/group in internal train or validation, and did not participate in checkpoint or operating-threshold selection. Thresholds and temperature parameters remain frozen from internal validation.",
                "",
                f"Manifest: `{SOURCE_DISJOINT_MANIFEST}`",
            ]
        )
    else:
        lines.extend(
            [
                "Reliable local mapping coverage was insufficient to create a compliant source-disjoint subset. PneumoniaMNIST performance must be removed from the main results and may be retained only as an exploratory same-source stress test.",
                "",
                "No independent external clinical cohort was evaluated.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- This audit can establish same-source provenance and identify conservative transformed counterparts of internally held-out images.",
            "- It cannot convert PneumoniaMNIST into an independent external cohort.",
            "- It cannot validate cross-hospital performance or clinical portability.",
            "- Independent external clinical validation remains necessary.",
            "",
        ]
    )
    (REPORT_DIR / "pneumoniamnist_provenance_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DISJOINT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    source, clean, duplicate, med, official = load_inputs()
    enriched_path = REPORT_DIR / "kermany_source_inventory_enriched.csv"
    if enriched_path.exists():
        cached = pd.read_csv(enriched_path, encoding="utf-8-sig")
        source_enriched = cached if len(cached) == len(source) else enrich_source_inventory(source, clean, duplicate)
    else:
        source_enriched = enrich_source_inventory(source, clean, duplicate)
    source_enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    med_samples = med_sample_inventory(med, official)
    assets = build_asset_inventory(source, clean)
    write_asset_inventory(assets)

    med_samples.to_csv(
        REPORT_DIR / "pneumoniamnist_sample_inventory.csv", index=False, encoding="utf-8-sig"
    )

    mapping, best = compute_mapping(source_enriched, med_samples, med, top_k=args.top_k)
    mapping.to_csv(
        REPORT_DIR / "pneumoniamnist_overlap_mapping.csv", index=False, encoding="utf-8-sig"
    )
    best[best["mapping_status"].eq("ambiguous_match")].to_csv(
        REPORT_DIR / "pneumoniamnist_ambiguous_matches.csv", index=False, encoding="utf-8-sig"
    )
    best[best["mapping_status"].eq("unmatched")].to_csv(
        REPORT_DIR / "pneumoniamnist_unmatched.csv", index=False, encoding="utf-8-sig"
    )

    summary = summarize_mapping(best, clean, OFFICIAL_ID_MAPPING.exists())
    subset = source_disjoint_subset(best, clean)
    subset.to_csv(SOURCE_DISJOINT_MANIFEST, index=False, encoding="utf-8-sig")
    summary["source_disjoint_subset_n"] = int(len(subset))
    summary["source_disjoint_decision"] = "A" if len(subset) else "B"
    summary["source_disjoint_manifest"] = str(SOURCE_DISJOINT_MANIFEST)
    (REPORT_DIR / "pneumoniamnist_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_audit_report(summary, subset)
    create_match_figure(
        best,
        med,
        FIGURE_DIR / "pneumoniamnist_match_examples.png",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"PASS: provenance audit completed; source-disjoint subset N={len(subset)}")


if __name__ == "__main__":
    main()
