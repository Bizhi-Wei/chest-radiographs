# PneumoniaMNIST provenance and source-overlap audit

## Immediate correction

PneumoniaMNIST was derived from the same underlying Kermany pediatric chest radiograph collection and therefore does not constitute an independent external cohort.

The analysis is termed a **same-source preprocessing and resolution-shift stress test**. It does not test cross-institutional or independent-patient clinical generalization.

## Evidence base

- Official MedMNIST project: https://github.com/MedMNIST/MedMNIST
- Official source-ID mapping folder: https://drive.google.com/drive/folders/1A_99qH_c-J0p_SatwSiaP_i1CvLUOzVo?usp=sharing
- The official metadata states that PneumoniaMNIST is based on the 5,856-image Kermany pediatric chest radiograph collection, center-cropped and resized to 28x28.
- The official mapping contains 5,856 rows and 624 test rows. The test subtype counts are 234 NORMAL, 242 BACTERIA, and 148 VIRUS, exactly matching the local Kermany source test pool.
- Local filenames differ from the official MedMNIST image IDs, so mapping to local files was independently checked with multiple documented 28x28 reconstructions, pixel distances, NCC, SSIM, aHash, pHash, dHash, and bidirectional top-1 consistency.

## Mapping thresholds

```json
{
  "confirmed": {
    "mutual_top1": true,
    "subtype_match": true,
    "max_mse": 100.0,
    "max_mae": 8.0,
    "min_ncc": 0.985,
    "min_ssim": 0.95,
    "max_phash_distance": 12,
    "max_dhash_distance": 14,
    "min_second_to_first_mse_ratio": 1.2
  },
  "probable": {
    "subtype_match": true,
    "max_mse": 400.0,
    "max_mae": 15.0,
    "min_ncc": 0.95,
    "min_ssim": 0.85,
    "max_phash_distance": 18,
    "max_dhash_distance": 20,
    "min_second_to_first_mse_ratio": 1.1,
    "requires_mutual_or_strong_margin": true
  },
  "ambiguous": {
    "min_ncc": 0.8,
    "min_ssim": 0.65
  }
}
```

## PneumoniaMNIST test mapping

- Total: 624
- Confirmed: 549
- Probable: 1
- Ambiguous: 48
- Unmatched: 26
- Label conflicts: 0
- One source image mapped to multiple accepted PneumoniaMNIST samples: 0

Accepted confirmed/probable source-image matches by internal clean split:

- Internal train: 366 images, 243 patient/groups
- Internal validation: 75 images, 51 patient/groups
- Internal test: 109 images, 68 patient/groups
- Not retained after exact deduplication or unresolved: 0

No-match warning: not finding a high-confidence local mapping does not prove that overlap is absent.

## Source-disjoint stress-test decision

Decision: **Scheme A**.

A conservative source-disjoint stress-test subset was created with 108 samples. Every retained sample is a confirmed local source match, belongs to the internal held-out test split, has no patient/group in internal train or validation, and did not participate in checkpoint or operating-threshold selection. Thresholds and temperature parameters remain frozen from internal validation.

Manifest: `<project-root>/01_data/manifests/pneumoniamnist_source_disjoint_stress_test.csv`

## Interpretation boundary

- This audit can establish same-source provenance and identify conservative transformed counterparts of internally held-out images.
- It cannot convert PneumoniaMNIST into an independent external cohort.
- It cannot validate cross-hospital performance or clinical portability.
- Independent external clinical validation remains necessary.
