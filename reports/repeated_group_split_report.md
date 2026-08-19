# Clean repeated patient/group split ResNet18 results

- Target root: `<project-root>/data/clean_repeated_group_splits`
- Output dir: `<project-root>/output_clean_20260707/repeated_splits_resnet18`
- Device: `cuda`
- Repeats: 5
- Epoch cap: 15
- Threshold policy: selected on validation set by maximum F1, then frozen for test evaluation.
- Confidence intervals: patient/group bootstrap on test groups.

## Summary

| Metric | Mean | SD | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.9668 | 0.0026 | 0.9634 | 0.9703 |
| f1 | 0.9774 | 0.0016 | 0.9753 | 0.9796 |
| roc_auc | 0.9932 | 0.0016 | 0.9907 | 0.9949 |
| pr_auc | 0.9974 | 0.0007 | 0.9963 | 0.9981 |
| sensitivity | 0.9846 | 0.0072 | 0.9749 | 0.9937 |
| specificity | 0.9190 | 0.0245 | 0.8819 | 0.9409 |