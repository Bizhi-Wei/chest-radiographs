# Leakage-aware pediatric pneumonia classification: code and derived data

This repository accompanies the manuscript *Leakage-aware evaluation of deep learning for pediatric pneumonia classification on chest radiographs* by Lei Tang and Bizhi Wei.

Public repository: https://github.com/Bizhi-Wei/chest-radiographs

## Scope

The release contains analysis code, audit scripts, de-identified derived manifests, SHA-256 hashes, saved predictions, bootstrap outputs, tables and non-image statistical figures. It supports the leakage-aware internal evaluation and the post hoc held-out-source-matched same-source PneumoniaMNIST preprocessing stress test.

PneumoniaMNIST is derived from the same Kermany pediatric chest-radiograph source collection and is not an independent external clinical cohort.

## Data exclusions

The release does not redistribute original Kermany radiographs, PneumoniaMNIST arrays, model checkpoints, Grad-CAM radiograph panels or source-mapping contact sheets. Those items remain subject to their original licences or contain source-image pixels. Local absolute paths were replaced with repository-relative identifiers.

## Reproduction

Install `code/requirements.txt`, inspect the script headers for inputs, and run `pytest -q` for consistency checks. Thresholds and temperature-scaling parameters are loaded from internal-validation outputs; they must not be refitted on PneumoniaMNIST.

## Licences

- Source code is licensed under the MIT License; see `LICENSE`.
- Derived non-code materials, including tables, manifests, statistical outputs, manuscript text, reports and figures, are licensed under the Creative Commons Attribution 4.0 International License; see `LICENSE-DATA.md`.
- Original radiographs, PneumoniaMNIST arrays and other third-party materials are not redistributed and remain governed by their providers' terms.
