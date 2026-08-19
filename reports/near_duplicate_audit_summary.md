# Near-duplicate audit

Dataset: `<project-root>/data/chest_xray_patient_split_dedup`
Images scanned: 5824

## Methods

- Computed aHash, dHash and pHash for each image.
- Generated candidate pairs using perceptual-hash Hamming thresholds and ResNet18 embedding nearest neighbours.
- Computed SSIM for the strongest candidate pairs.
- Generated contact sheets for manual review.

## Summary counts

- Candidate pairs reviewed: 21738
- Cross-split candidate pairs: 9712
- Pairs with SSIM >= 0.95: 0
- Pairs with embedding cosine >= 0.99: 0
- Cross-split high-similarity pairs needing manual review: 0

## Manual review images

- `reports/near_duplicate_audit/clean_20260706_manual_review_top_cross_split_pairs.png`
- `reports/near_duplicate_audit/clean_20260706_manual_review_top_overall_pairs.png`

## Interpretation

These files are an audit queue, not automatic proof of leakage. Top cross-split pairs should be manually inspected and, if judged to be near duplicates, handled by excluding one image or grouping the pair before final modelling.