from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    PROJECT_ROOT
    if (PROJECT_ROOT / "data").is_dir() and (PROJECT_ROOT / "results").is_dir()
    else PROJECT_ROOT / "04_reports/public_release/github_release_20260805_v2"
)
REPOSITORY_URL = "https://github.com/Bizhi-Wei/chest-radiographs"


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(RELEASE / path, encoding="utf-8-sig")


def test_release_identity_and_repository_url() -> None:
    assert RELEASE.is_dir()
    readme = (RELEASE / "README.md").read_text(encoding="utf-8")
    citation = (RELEASE / "CITATION.cff").read_text(encoding="utf-8")
    manuscript = (RELEASE / "manuscript/Leakage_aware_pneumonia_JXST_submission_v4.md").read_text(
        encoding="utf-8"
    )
    for text in (readme, citation, manuscript):
        assert REPOSITORY_URL in text
        assert "Lei Tang" not in text
        assert "0009-0006-7762-8105" not in text
    assert citation.count("family-names:") == 1
    assert "family-names: Wei" in citation


def test_release_has_explicit_code_and_data_licences() -> None:
    readme = (RELEASE / "README.md").read_text(encoding="utf-8")
    mit = (RELEASE / "LICENSE").read_text(encoding="utf-8")
    data_licence = (RELEASE / "LICENSE-DATA.md").read_text(encoding="utf-8")
    assert "MIT License" in readme
    assert "Creative Commons Attribution 4.0 International" in readme
    assert "Permission is hereby granted" in mit
    assert "creativecommons.org/licenses/by/4.0/legalcode" in data_licence


def test_release_contains_no_prohibited_data_or_local_paths() -> None:
    prohibited_suffixes = {".jpeg", ".jpg", ".dcm", ".dicom", ".npz", ".pt", ".pth"}
    text_suffixes = {".csv", ".json", ".md", ".py", ".txt", ".cff"}
    local_paths = (
        re.compile(r"/mnt/[a-z]/"),
        re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\"),
    )
    for path in RELEASE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in prohibited_suffixes, path
        if path.suffix.lower() in text_suffixes:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for pattern in local_paths:
                assert not pattern.search(text), path


def test_release_manifest_is_complete_and_hashes_match() -> None:
    manifest_path = RELEASE / "release_file_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    listed = {row["relative_path"] for row in rows}
    actual = {
        path.relative_to(RELEASE).as_posix()
        for path in RELEASE.rglob("*")
        if path.is_file()
        and path.name != manifest_path.name
        and ".git" not in path.parts
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    }
    assert listed == actual
    for row in rows:
        path = RELEASE / row["relative_path"]
        payload = path.read_bytes()
        assert len(payload) == int(row["size_bytes"]), path
        assert hashlib.sha256(payload).hexdigest() == row["sha256"], path


def test_clean_manifest_has_no_group_or_hash_cross_split_leakage() -> None:
    manifest = read_csv("data/patient_split_manifest.csv")
    assert len(manifest) == 5824
    assert manifest.groupby("patient_id")["split"].nunique().max() == 1
    assert manifest.groupby("sha256")["split"].nunique().max() == 1
    assert manifest["split"].value_counts().to_dict() == {"train": 4077, "val": 874, "test": 873}


def test_stress_subset_is_held_out_and_uses_frozen_parameters() -> None:
    manifest = read_csv("data/patient_split_manifest.csv")
    stress = read_csv("data/pneumoniamnist_source_matched_stress_manifest.csv")
    assert len(stress) == 108
    assert stress["mapping_status"].eq("confirmed_match").all()
    assert stress["internal_split"].eq("test").all()
    assert stress["threshold_source"].eq("internal_validation_only").all()
    assert stress["temperature_source"].eq("internal_validation_only").all()
    assert not stress["pneumoniamnist_threshold_optimization"].astype(bool).any()
    assert not stress["pneumoniamnist_temperature_refit"].astype(bool).any()
    train_val_groups = set(manifest.loc[manifest["split"].isin(["train", "val"]), "patient_id"])
    assert train_val_groups.isdisjoint(set(stress["internal_patient_id"]))


def test_provenance_counts_and_interpretation_boundary() -> None:
    summary = json.loads((RELEASE / "results/pneumoniamnist_overlap_summary.json").read_text(encoding="utf-8"))
    assert summary["independent_external_cohort"] is False
    assert summary["pneumoniamnist_test_n"] == 624
    assert summary["mapping_status_counts"] == {
        "confirmed_match": 549,
        "ambiguous_match": 48,
        "unmatched": 26,
        "probable_match": 1,
    }
    assert summary["accepted_matches_by_internal_split"] == {"train": 366, "test": 109, "val": 75}
    assert summary["source_disjoint_subset_n"] == 108


def test_stress_metrics_match_saved_predictions() -> None:
    metrics = read_csv("results/pneumoniamnist_stress_metrics.csv")
    predictions = read_csv("results/pneumoniamnist_stress_predictions.csv")
    prediction_columns = {
        "default_0.5": "prediction_default_0_5",
        "internal_validation_selected": "prediction_internal_validation_threshold",
    }
    for row in metrics.itertuples(index=False):
        subset = predictions[predictions["model"].eq(row.model)]
        y_true = subset["true_label"].to_numpy()
        y_pred = subset[prediction_columns[row.threshold_name]].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        assert (tn, fp, fn, tp) == (row.tn, row.fp, row.fn, row.tp)
        assert np.isclose(accuracy_score(y_true, y_pred), row.accuracy)
        assert np.isclose(roc_auc_score(y_true, subset["raw_probability"]), row.roc_auc)


def test_manuscript_uses_same_source_stress_test_language() -> None:
    manuscript = (RELEASE / "manuscript/Leakage_aware_pneumonia_JXST_submission_v4.md").read_text(
        encoding="utf-8"
    ).lower()
    assert "same-source preprocessing" in manuscript
    assert any(
        boundary in manuscript
        for boundary in (
            "no independent external clinical cohort",
            "not an independent external cohort",
            "no independent institutional cohort",
        )
    )
    for prohibited in (
        "pneumoniamnist independent external validation",
        "zero-shot external validation",
        "external clinical cohort using pneumoniamnist",
    ):
        assert prohibited not in manuscript
