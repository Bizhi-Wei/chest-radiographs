# Data contamination notice (2026-07-06)

## Do not use the previous 6,570-image main analysis for publication

The previous datasets `data/chest_xray_patient_split` and
`data/chest_xray_patient_split_dedup` contain 746 synthetic RGB noise images
named `NORMAL_####.jpeg` or `PNEUMONIA_####.jpeg`.

- NORMAL placeholders contain near-uniform coloured noise.
- PNEUMONIA placeholders contain coloured noise with a bright central region.
- The placeholders are label-encoded and can be classified trivially.
- All model weights, thresholds, calibration results, Grad-CAM outputs and
  manuscript values derived from the 6,570-image split are contaminated.

Affected outputs include:

- `output_patient_split`
- `output_patient_split_baselines`
- `output_patient_split_dedup`
- `output_patient_split_dedup_baselines`
- `reports/final_tables/dedup_*`
- `reports/manuscript_polished_20260618.docx`
- `reports/latex_nature/main.tex`

These files are retained only as an audit trail and must not be reported as
current results.

## Clean primary dataset

The clean rebuild excludes all 746 placeholders before exact deduplication:

- Real source images: 5,856
- Exact duplicate images removed: 32
- Retained images: 5,824
- Train: 4,077 (NORMAL 1,105; PNEUMONIA 2,972)
- Validation: 874 (NORMAL 237; PNEUMONIA 637)
- Test: 873 (NORMAL 237; PNEUMONIA 636)
- Patient cross-split leakage: 0
- SHA-256 cross-split leakage: 0

Canonical clean dataset:

`<project-root>\data\chest_xray_patient_split_dedup`

All primary models must be retrained from random optimizer state on this clean
split. Previous checkpoints must not be reused.
