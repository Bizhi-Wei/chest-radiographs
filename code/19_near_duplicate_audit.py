"""
Near-duplicate audit for the deduplicated patient-level split.

Checks:
  - perceptual hashes: aHash, dHash, pHash
  - SSIM on candidate similar pairs
  - ImageNet ResNet18 embedding cosine similarity
  - manual-review contact sheets for top similar pairs

Default input:
  data/chest_xray_patient_split_dedup

Outputs:
  reports/near_duplicate_audit/
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fftpack import dct
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("PNEUMONIA_DATASET_DIR", BASE_DIR / "data" / "chest_xray_patient_split_dedup"))
if not DATASET_DIR.is_absolute():
    DATASET_DIR = BASE_DIR / DATASET_DIR
OUT_DIR = BASE_DIR / "reports" / "near_duplicate_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = os.environ.get("PNEUMONIA_EVAL_PREFIX", "")

SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}

HASH_CANDIDATE_LIMIT = int(os.environ.get("NEAR_DUP_HASH_LIMIT", "2500"))
EMBED_TOPK_PER_IMAGE = int(os.environ.get("NEAR_DUP_EMBED_TOPK", "3"))
SSIM_EVAL_LIMIT = int(os.environ.get("NEAR_DUP_SSIM_LIMIT", "4000"))
MANUAL_REVIEW_N = int(os.environ.get("NEAR_DUP_MANUAL_N", "40"))
BATCH_SIZE = int(os.environ.get("NEAR_DUP_BATCH_SIZE", "64"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ImageRecord:
    idx: int
    path: Path
    split: str
    label: str
    patient_id: str
    filename: str
    sha256: str
    ahash: int
    dhash: int
    phash: int


def extract_patient_id(filename: str, label: str) -> str:
    stem = Path(filename).stem
    if label == "PNEUMONIA":
        match = re.search(r"(person\d+)", stem, flags=re.IGNORECASE)
    else:
        match = re.search(r"(NORMAL2-IM-\d+|IM-\d+)", stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else stem.upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pil_gray(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize(size, Image.Resampling.LANCZOS)
    return image


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.astype(bool).ravel():
        value = (value << 1) | int(bit)
    return int(value)


def ahash(path: Path) -> int:
    arr = np.asarray(pil_gray(path, (8, 8)), dtype=np.float32)
    return bits_to_int(arr >= arr.mean())


def dhash(path: Path) -> int:
    arr = np.asarray(pil_gray(path, (9, 8)), dtype=np.float32)
    return bits_to_int(arr[:, 1:] >= arr[:, :-1])


def phash(path: Path) -> int:
    arr = np.asarray(pil_gray(path, (32, 32)), dtype=np.float32)
    coeff = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    vals = coeff.ravel()[1:]
    median = np.median(vals)
    return bits_to_int(coeff >= median)


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def scan_records() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    idx = 0
    for split in SPLITS:
        for label in CLASSES:
            folder = DATASET_DIR / split / label
            if not folder.exists():
                raise FileNotFoundError(folder)
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                records.append(
                    ImageRecord(
                        idx=idx,
                        path=path,
                        split=split,
                        label=label,
                        patient_id=extract_patient_id(path.name, label),
                        filename=path.name,
                        sha256=file_sha256(path),
                        ahash=ahash(path),
                        dhash=dhash(path),
                        phash=phash(path),
                    )
                )
                idx += 1
    return records


def hash_candidates(records: list[ImageRecord]) -> pd.DataFrame:
    rows = []
    n = len(records)
    for i in tqdm(range(n), desc="Hash pair scan"):
        a = records[i]
        for j in range(i + 1, n):
            b = records[j]
            # Near-duplicate review is most meaningful within the same label,
            # but cross-label very similar images are retained as suspicious.
            dh = hamming(a.dhash, b.dhash)
            ph = hamming(a.phash, b.phash)
            ah = hamming(a.ahash, b.ahash)
            if dh <= 8 or ph <= 10 or ah <= 8:
                rows.append(
                    {
                        "idx1": i,
                        "idx2": j,
                        "source": "hash",
                        "dhash_hamming": dh,
                        "phash_hamming": ph,
                        "ahash_hamming": ah,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["hash_score"] = df["dhash_hamming"] + df["phash_hamming"] + df["ahash_hamming"]
    return df.sort_values(["hash_score", "dhash_hamming", "phash_hamming"]).head(HASH_CANDIDATE_LIMIT)


class ImagePathDataset(Dataset):
    def __init__(self, records: list[ImageRecord]) -> None:
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), index


def compute_embeddings(records: list[ImageRecord]) -> torch.Tensor:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(DEVICE).eval()
    loader = DataLoader(ImagePathDataset(records), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    features = torch.zeros((len(records), 512), dtype=torch.float32)
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="Embedding"):
            images = images.to(DEVICE)
            feat = model(images)
            feat = F.normalize(feat, dim=1).cpu()
            features[indices] = feat
    return features


def embedding_candidates(features: torch.Tensor, records: list[ImageRecord]) -> pd.DataFrame:
    rows = []
    feat = features.to(DEVICE)
    n = feat.shape[0]
    for start in tqdm(range(0, n, 256), desc="Embedding nearest pairs"):
        end = min(start + 256, n)
        sims = feat[start:end] @ feat.T
        for local_i in range(end - start):
            i = start + local_i
            sims[local_i, i] = -1.0
            # Restricting to topk per image keeps the table reviewable while
            # still catching high-similarity pairs missed by simple hashes.
            vals, inds = torch.topk(sims[local_i], k=min(EMBED_TOPK_PER_IMAGE + 1, n))
            for val, j_tensor in zip(vals.tolist(), inds.tolist()):
                j = int(j_tensor)
                if i == j:
                    continue
                a, b = sorted((i, j))
                rows.append({"idx1": a, "idx2": b, "source": "embedding", "embedding_cosine": float(val)})
    df = pd.DataFrame(rows).drop_duplicates(["idx1", "idx2"])
    return df


def ssim_score(path1: Path, path2: Path) -> float:
    img1 = np.asarray(pil_gray(path1, (256, 256)), dtype=np.float32)
    img2 = np.asarray(pil_gray(path2, (256, 256)), dtype=np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    sigma1 = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1 * mu1
    sigma2 = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2 * mu2
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2)
    return float(np.mean(numerator / (denominator + 1e-8)))


def enrich_candidates(candidates: pd.DataFrame, embeddings: torch.Tensor, records: list[ImageRecord]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    candidates = candidates.drop_duplicates(["idx1", "idx2"]).copy()
    emb_np = embeddings.numpy()
    meta_rows = []
    ordered = candidates.copy()
    # Prioritize strong hash or embedding candidates for expensive SSIM.
    if "embedding_cosine" not in ordered.columns:
        ordered["embedding_cosine"] = np.nan
    if "hash_score" not in ordered.columns:
        ordered["hash_score"] = np.nan
    ordered = ordered.sort_values(["embedding_cosine", "hash_score"], ascending=[False, True], na_position="last")
    ssim_pairs = set(map(tuple, ordered[["idx1", "idx2"]].head(SSIM_EVAL_LIMIT).to_numpy()))
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Candidate metrics"):
        i, j = int(row["idx1"]), int(row["idx2"])
        a, b = records[i], records[j]
        emb_sim = float(np.dot(emb_np[i], emb_np[j]))
        pair_ssim = ssim_score(a.path, b.path) if (i, j) in ssim_pairs else np.nan
        meta_rows.append(
            {
                **row.to_dict(),
                "embedding_cosine": emb_sim,
                "ssim": pair_ssim,
                "same_split": a.split == b.split,
                "cross_split": a.split != b.split,
                "same_patient_id": a.patient_id == b.patient_id,
                "same_label": a.label == b.label,
                "split1": a.split,
                "split2": b.split,
                "label1": a.label,
                "label2": b.label,
                "patient_id1": a.patient_id,
                "patient_id2": b.patient_id,
                "filename1": a.filename,
                "filename2": b.filename,
                "path1": str(a.path.relative_to(DATASET_DIR)).replace("\\", "/"),
                "path2": str(b.path.relative_to(DATASET_DIR)).replace("\\", "/"),
                "sha256_equal": a.sha256 == b.sha256,
            }
        )
    out = pd.DataFrame(meta_rows)
    return out.sort_values(["cross_split", "ssim", "embedding_cosine"], ascending=[False, False, False])


def make_contact_sheet(df: pd.DataFrame, records: list[ImageRecord], name: str, n: int) -> Path | None:
    if df.empty:
        return None
    rows = []
    for _, row in df.head(n).iterrows():
        rows.append((records[int(row["idx1"])], records[int(row["idx2"])], row))
    tile_w, tile_h = 256, 256
    label_h = 78
    sheet = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for r, (a, b, row) in enumerate(rows):
        y = r * (tile_h + label_h)
        for col, rec in enumerate((a, b)):
            with Image.open(rec.path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((tile_w, tile_h))
                canvas = Image.new("RGB", (tile_w, tile_h), "black")
                canvas.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
            sheet.paste(canvas, (col * tile_w, y))
        text = (
            f"{r+1}. {a.split}/{a.label}/{a.patient_id}  vs  {b.split}/{b.label}/{b.patient_id}\n"
            f"SSIM={row.get('ssim', np.nan):.4f}  Emb={row.get('embedding_cosine', np.nan):.4f}  "
            f"dH={row.get('dhash_hamming', np.nan)} pH={row.get('phash_hamming', np.nan)} aH={row.get('ahash_hamming', np.nan)}\n"
            f"{a.filename} | {b.filename}"
        )
        draw.text((6, y + tile_h + 4), text, fill="black", font=font)
    out_path = OUT_DIR / f"{OUTPUT_PREFIX}{name}"
    sheet.save(out_path)
    return out_path


def write_summary(df: pd.DataFrame, records: list[ImageRecord], contact_paths: list[Path | None]) -> None:
    cross = df[df["cross_split"]] if not df.empty else df
    high_ssim = df[df["ssim"].fillna(0) >= 0.95] if not df.empty else df
    high_emb = df[df["embedding_cosine"].fillna(0) >= 0.99] if not df.empty else df
    cross_high = cross[(cross["ssim"].fillna(0) >= 0.95) | (cross["embedding_cosine"].fillna(0) >= 0.99)] if not df.empty else df
    lines = [
        "# Near-duplicate audit",
        "",
        f"Dataset: `{DATASET_DIR}`",
        f"Images scanned: {len(records)}",
        "",
        "## Methods",
        "",
        "- Computed aHash, dHash and pHash for each image.",
        "- Generated candidate pairs using perceptual-hash Hamming thresholds and ResNet18 embedding nearest neighbours.",
        "- Computed SSIM for the strongest candidate pairs.",
        "- Generated contact sheets for manual review.",
        "",
        "## Summary counts",
        "",
        f"- Candidate pairs reviewed: {len(df)}",
        f"- Cross-split candidate pairs: {len(cross)}",
        f"- Pairs with SSIM >= 0.95: {len(high_ssim)}",
        f"- Pairs with embedding cosine >= 0.99: {len(high_emb)}",
        f"- Cross-split high-similarity pairs needing manual review: {len(cross_high)}",
        "",
        "## Manual review images",
        "",
    ]
    for path in contact_paths:
        if path is not None:
            lines.append(f"- `{path.relative_to(BASE_DIR)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These files are an audit queue, not automatic proof of leakage. Top cross-split pairs should be manually inspected and, if judged to be near duplicates, handled by excluding one image or grouping the pair before final modelling.",
        ]
    )
    (OUT_DIR / f"{OUTPUT_PREFIX}near_duplicate_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Near-duplicate audit")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUT_DIR}")
    print(f"Device:  {DEVICE}")
    print("=" * 80)
    records = scan_records()
    records_df = pd.DataFrame([r.__dict__ | {"path": str(r.path.relative_to(DATASET_DIR)).replace("\\", "/")} for r in records])
    records_df.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}near_duplicate_image_index.csv", index=False, encoding="utf-8-sig")

    hdf = hash_candidates(records)
    embeddings = compute_embeddings(records)
    edf = embedding_candidates(embeddings, records)
    candidates = pd.concat([hdf, edf], ignore_index=True, sort=False)
    enriched = enrich_candidates(candidates, embeddings, records)
    enriched.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}near_duplicate_candidates.csv", index=False, encoding="utf-8-sig")

    cross = enriched[enriched["cross_split"]].sort_values(["ssim", "embedding_cosine"], ascending=[False, False])
    cross.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}near_duplicate_top_cross_split.csv", index=False, encoding="utf-8-sig")
    top_overall = enriched.sort_values(["ssim", "embedding_cosine"], ascending=[False, False])
    top_overall.to_csv(OUT_DIR / f"{OUTPUT_PREFIX}near_duplicate_top_overall.csv", index=False, encoding="utf-8-sig")

    contact_paths = [
        make_contact_sheet(cross, records, "manual_review_top_cross_split_pairs.png", MANUAL_REVIEW_N),
        make_contact_sheet(top_overall, records, "manual_review_top_overall_pairs.png", MANUAL_REVIEW_N),
    ]
    write_summary(enriched, records, contact_paths)
    print(f"Saved: {OUT_DIR / f'{OUTPUT_PREFIX}near_duplicate_candidates.csv'}")
    print(f"Saved: {OUT_DIR / f'{OUTPUT_PREFIX}near_duplicate_audit_summary.md'}")


if __name__ == "__main__":
    main()
