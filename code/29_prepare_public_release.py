from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PureWindowsPath

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "04_reports/public_release/github_release_20260805_v2"
PUBLIC_REPOSITORY_URL = os.environ.get("PUBLIC_REPOSITORY_URL", "").strip().rstrip("/")

CSV_FILES = {
    "data/source_image_inventory.csv": "04_reports/clean_rebuild_20260706/source_image_inventory.csv",
    "data/patient_split_manifest.csv": "04_reports/clean_rebuild_20260706/patient_split_manifest.csv",
    "data/patient_split_summary.csv": "04_reports/clean_rebuild_20260706/patient_split_summary.csv",
    "data/exact_duplicate_audit.csv": "04_reports/clean_rebuild_20260706/exact_duplicate_audit.csv",
    "data/excluded_synthetic_placeholders.csv": "04_reports/clean_rebuild_20260706/excluded_synthetic_placeholders.csv",
    "data/cross_split_leakage_check.csv": "04_reports/clean_rebuild_20260706/leakage_check.csv",
    "data/patient_id_fallback_log.csv": "04_reports/clean_rebuild_20260706/patient_id_fallback_log.csv",
    "data/pneumoniamnist_official_split_info.csv": "01_data/manifests/medmnist_official_id_mapping/pneumoniamnist_split_info.csv",
    "data/pneumoniamnist_overlap_mapping.csv": "04_reports/pneumoniamnist_overlap_mapping.csv",
    "data/pneumoniamnist_ambiguous_matches.csv": "04_reports/pneumoniamnist_ambiguous_matches.csv",
    "data/pneumoniamnist_unmatched.csv": "04_reports/pneumoniamnist_unmatched.csv",
    "data/pneumoniamnist_source_matched_stress_manifest.csv": "01_data/manifests/pneumoniamnist_source_disjoint_stress_test.csv",
    "results/pneumoniamnist_stress_predictions.csv": "04_reports/pneumoniamnist_source_disjoint_predictions.csv",
    "results/pneumoniamnist_stress_metrics.csv": "04_reports/pneumoniamnist_source_disjoint_metrics.csv",
    "results/pneumoniamnist_paired_shift.csv": "04_reports/pneumoniamnist_paired_shift_analysis.csv",
    "results/internal_test_predictions.csv": "04_reports/submission_v2/internal_test_predictions_v2.csv",
    "results/internal_test_metrics_group_ci.csv": "04_reports/submission_v2/internal_test_metrics_with_group_ci_v2.csv",
    "results/paired_shift_summary_group_ci.csv": "04_reports/submission_v2/paired_shift_summary_v2.csv",
    "results/validation_selected_thresholds.csv": "04_reports/final_tables/clean_20260706_validation_selected_thresholds.csv",
    "results/calibration_metrics.csv": "04_reports/final_tables/clean_20260706_calibration_metrics.csv",
    "results/calibration_bins.csv": "04_reports/final_tables/clean_20260706_calibration_bins.csv",
    "results/repeated_group_split_per_repeat.csv": "03_models_and_outputs/output_clean_20260707/repeated_splits_resnet18/reports/clean_20260706_repeated_split_resnet18_per_repeat.csv",
    "results/repeated_group_split_summary.csv": "03_models_and_outputs/output_clean_20260707/repeated_splits_resnet18/reports/clean_20260706_repeated_split_resnet18_summary.csv",
    "results/near_duplicate_top_cross_split.csv": "04_reports/near_duplicate_audit/clean_20260706_near_duplicate_top_cross_split.csv",
    "results/gradcam_error_cases.csv": "04_reports/gradcam_error_analysis/clean_20260706_resnet18_gradcam_error_cases.csv",
    "tables/Table_2_internal_test_performance.csv": "04_reports/final_tables/submission_v2/Table_2_internal_test_performance_v2.csv",
    "tables/Table_5_same_source_stress_test.csv": "04_reports/final_tables/submission_v2/Table_5_same_source_stress_test_v2.csv",
    "tables/Supplementary_Table_S5_full_stress_metrics.csv": "04_reports/final_tables/submission_v2/Supplementary_Table_S5_full_stress_metrics_v2.csv",
}

JSON_FILES = {
    "results/pneumoniamnist_overlap_summary.json": "04_reports/pneumoniamnist_overlap_summary.json",
    "results/pneumoniamnist_stress_bootstrap.json": "04_reports/pneumoniamnist_source_disjoint_bootstrap.json",
    "results/internal_test_bootstrap.json": "04_reports/submission_v2/internal_test_bootstrap_v2.json",
    "results/paired_shift_bootstrap.json": "04_reports/submission_v2/paired_shift_bootstrap_v2.json",
}

TEXT_FILES = {
    "reports/pneumoniamnist_provenance_audit.md": "04_reports/pneumoniamnist_provenance_audit.md",
    "reports/pneumoniamnist_stress_test_report.md": "04_reports/pneumoniamnist_source_disjoint_stress_test_report.md",
    "reports/near_duplicate_audit_summary.md": "04_reports/near_duplicate_audit/clean_20260706_near_duplicate_audit_summary.md",
    "reports/gradcam_error_analysis_summary.md": "04_reports/gradcam_error_analysis/clean_20260706_resnet18_gradcam_error_analysis_summary.md",
    "reports/repeated_group_split_report.md": "03_models_and_outputs/output_clean_20260707/repeated_splits_resnet18/reports/clean_20260706_repeated_split_resnet18_report.md",
    "reports/data_contamination_notice.md": "02_scripts/DATA_CONTAMINATION_NOTICE_20260706.md",
    "manuscript/Leakage_aware_pneumonia_JXST_submission_v4.md": "05_manuscript/Leakage_aware_pneumonia_JXST_submission_v4.md",
    "manuscript/figure_legends_JXST_v4.md": "05_manuscript/figure_legends_JXST_v4.md",
}

SCRIPT_FILES = [
    "12_rebuild_strict_patient_split.py",
    "14_train_patient_split_baseline.py",
    "17_validation_threshold_final_eval.py",
    "19_near_duplicate_audit.py",
    "20_create_repeated_stratified_group_splits.py",
    "22_calibration_analysis.py",
    "23_gradcam_error_analysis.py",
    "25_run_clean_repeated_splits_gpu.py",
    "26_audit_pneumoniamnist_provenance.py",
    "27_evaluate_pneumoniamnist_source_disjoint_stress.py",
    "28_prepare_submission_revision_v2.py",
    "29_prepare_public_release.py",
    "patient_group_utils.py",
    "requirements.txt",
]

TEST_FILES = [
    "test_public_release.py",
]

FIGURE_FILES = [
    "main/Figure_1_workflow",
    "main/Figure_2_internal_performance",
    "main/Figure_3_paired_probability_shift",
    "main/Figure_4_repeated_split_performance",
    "supplementary/Supplementary_Figure_S2_paired_logit_shift",
    "supplementary/Supplementary_Figure_S3_threshold_crossing",
    "supplementary/Supplementary_Figure_S4_probability_ecdf",
]

ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/mnt/[a-z]/)")
PRIVATE_ROOTS = (
    re.compile(r"E:" + r"\\胸部X光肺炎AI诊断研究", re.IGNORECASE),
    re.compile("/mnt/" + "e/" + "胸部X光肺炎AI诊断研究"),
    re.compile("/mnt/" + "d/" + "AAA_MMP9_NLRP3_Pneumonia_Workspace/02_pneumonia_xray_ai"),
    re.compile("/mnt/" + "d/" + "Leakage-aware deep learning for pediatric pneumonia diagnosis on chest radiographs"),
    re.compile(r"D:" + r"\\Leakage-aware deep learning for pediatric pneumonia diagnosis on chest radiographs", re.IGNORECASE),
)


def ensure_empty_release() -> None:
    if RELEASE.exists() and any(RELEASE.iterdir()):
        raise FileExistsError(f"Release directory already contains files: {RELEASE}")
    RELEASE.mkdir(parents=True, exist_ok=True)


def source_filename(value: object) -> str:
    text = str(value or "")
    return PureWindowsPath(text).name if "\\" in text else Path(text).name


def first_present(row: pd.Series, names: tuple[str, ...]) -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return "unknown"


def relative_source_id(row: pd.Series, path_column: str) -> str:
    filename = first_present(row, ("source_filename", "filename", "destination_name"))
    if filename == "unknown":
        filename = source_filename(row.get(path_column, "")) or "unknown"
    split = first_present(row, ("source_original_split", "source_split", "internal_split", "split"))
    label = first_present(row, ("source_label", "label", "label_name", "pneumoniamnist_label_name"))
    return f"kermany_source/{split}/{label}/{filename}"


def sanitize_csv(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source, encoding="utf-8-sig")
    for column in list(frame.columns):
        lowered = column.lower()
        if lowered == "source_path":
            frame.insert(
                frame.columns.get_loc(column),
                "source_relative_path",
                frame.apply(lambda row: relative_source_id(row, column), axis=1),
            )
            frame.drop(columns=[column], inplace=True)
        elif lowered == "retained_source_path":
            frame.insert(
                frame.columns.get_loc(column),
                "retained_source_filename",
                frame[column].map(source_filename),
            )
            frame.drop(columns=[column], inplace=True)

    for column in frame.columns:
        if not (
            pd.api.types.is_object_dtype(frame[column].dtype)
            or pd.api.types.is_string_dtype(frame[column].dtype)
        ):
            continue
        values = frame[column].fillna("").astype(str)
        if values.map(lambda value: bool(ABSOLUTE_PATH.search(value))).any():
            frame[column] = values.map(
                lambda value: f"local_file/{source_filename(value)}" if ABSOLUTE_PATH.search(value) else value
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")


def sanitize_json_value(value: object, key: str = "") -> object:
    if isinstance(value, dict):
        return {name: sanitize_json_value(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item, key) for item in value]
    if isinstance(value, str) and ABSOLUTE_PATH.search(value):
        if "path" in key.lower():
            return f"local_file/{source_filename(value)}"
        return sanitize_text(value)
    return value


def sanitize_text(text: str) -> str:
    sanitized = text
    for pattern in PRIVATE_ROOTS:
        sanitized = pattern.sub("<project-root>", sanitized)
    return sanitized


def write_json(source: Path, destination: Path) -> None:
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(sanitize_json_value(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(sanitize_text(source.read_text(encoding="utf-8-sig")), encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_release_documents() -> None:
    readme = """# Leakage-aware pediatric pneumonia classification: code and derived data

This repository accompanies the manuscript *Leakage-aware evaluation of deep learning for pediatric pneumonia classification on chest radiographs* by Lei Tang and Bizhi Wei.

Public repository: {PUBLIC_REPOSITORY_URL}

## Scope

The release contains analysis code, audit scripts, de-identified derived manifests, SHA-256 hashes, saved predictions, bootstrap outputs, tables and non-image statistical figures. It supports the leakage-aware internal evaluation and the post hoc held-out-source-matched same-source PneumoniaMNIST preprocessing stress test.

PneumoniaMNIST is derived from the same Kermany pediatric chest-radiograph source collection and is not an independent external clinical cohort.

## Data exclusions

The release does not redistribute original Kermany radiographs, PneumoniaMNIST arrays, model checkpoints, Grad-CAM radiograph panels or source-mapping contact sheets. Those items remain subject to their original licences or contain source-image pixels. Local absolute paths were replaced with repository-relative identifiers.

## Reproduction

Install `code/requirements.txt`, inspect the script headers for inputs, and run `pytest -q` for consistency checks. Thresholds and temperature-scaling parameters are loaded from internal-validation outputs; they must not be refitted on PneumoniaMNIST.

## Licence status

No reuse licence is granted by this release until the authors select and add an explicit code/data licence. Original datasets remain governed by their providers' terms.
"""
    citation = """cff-version: 1.2.0
message: "If you use this code or derived data, please cite the accompanying manuscript."
title: "Leakage-aware evaluation of deep learning for pediatric pneumonia classification on chest radiographs: code and derived data"
type: software
authors:
  - family-names: Tang
    given-names: Lei
  - family-names: Wei
    given-names: Bizhi
    orcid: "https://orcid.org/0009-0008-9481-3024"
version: 1.0.0
date-released: 2026-08-05
repository-code: "{PUBLIC_REPOSITORY_URL}"
url: "{PUBLIC_REPOSITORY_URL}"
keywords:
  - pediatric pneumonia
  - chest radiograph
  - data leakage
  - provenance audit
  - calibration
  - preprocessing shift
"""
    gitignore = """__pycache__/
.pytest_cache/
*.py[cod]
*.pt
*.pth
*.npz
*.jpeg
*.jpg
*.dicom
"""
    (RELEASE / "README.md").write_text(readme.format(PUBLIC_REPOSITORY_URL=PUBLIC_REPOSITORY_URL), encoding="utf-8")
    (RELEASE / "CITATION.cff").write_text(citation.format(PUBLIC_REPOSITORY_URL=PUBLIC_REPOSITORY_URL), encoding="utf-8")
    (RELEASE / ".gitignore").write_text(gitignore, encoding="utf-8")


def write_file_manifest() -> None:
    rows = []
    for path in sorted(item for item in RELEASE.rglob("*") if item.is_file()):
        if path.name == "release_file_manifest.csv":
            continue
        payload = path.read_bytes()
        rows.append(
            {
                "relative_path": path.relative_to(RELEASE).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(RELEASE / "release_file_manifest.csv", index=False, encoding="utf-8-sig")


def validate_release() -> None:
    prohibited_suffixes = {".jpeg", ".jpg", ".dcm", ".dicom", ".npz", ".pt", ".pth"}
    errors = []
    for path in sorted(item for item in RELEASE.rglob("*") if item.is_file()):
        if path.suffix.lower() in prohibited_suffixes:
            errors.append(f"Prohibited binary/data file: {path.relative_to(RELEASE)}")
        if path.suffix.lower() in {".csv", ".json", ".md", ".py", ".txt", ".cff"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for pattern in PRIVATE_ROOTS:
                if pattern.search(text):
                    errors.append(f"Unsanitized private root in {path.relative_to(RELEASE)}")
    if errors:
        raise RuntimeError("Release validation failed:\n" + "\n".join(errors))


def main() -> None:
    if not PUBLIC_REPOSITORY_URL.startswith("https://github.com/"):
        raise ValueError("Set PUBLIC_REPOSITORY_URL to the verified public GitHub repository URL.")
    ensure_empty_release()
    for destination, source in CSV_FILES.items():
        sanitize_csv(ROOT / source, RELEASE / destination)
    for destination, source in JSON_FILES.items():
        write_json(ROOT / source, RELEASE / destination)
    for destination, source in TEXT_FILES.items():
        write_text(ROOT / source, RELEASE / destination)
    for filename in SCRIPT_FILES:
        copy_file(ROOT / "02_scripts" / filename, RELEASE / "code" / filename)
    for filename in TEST_FILES:
        copy_file(ROOT / "tests" / filename, RELEASE / "tests" / filename)
    for stem in FIGURE_FILES:
        for suffix in (".png", ".pdf", ".svg"):
            copy_file(
                ROOT / "04_reports/figures_submission_v2" / f"{stem}{suffix}",
                RELEASE / "figures" / f"{stem}{suffix}",
            )
    write_release_documents()
    validate_release()
    write_file_manifest()
    validate_release()
    files = sum(1 for path in RELEASE.rglob("*") if path.is_file())
    size = sum(path.stat().st_size for path in RELEASE.rglob("*") if path.is_file())
    print(f"PASS: public release prepared at {RELEASE} ({files} files, {size} bytes).")


if __name__ == "__main__":
    main()
