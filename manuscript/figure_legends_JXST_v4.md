# Figure legends

## Figure 1
Data provenance and evaluation workflow. (a) A contaminated prior local working copy contained 746 label-encoded synthetic placeholders introduced during an earlier local data-processing workflow. Those files were not part of the original Kermany source collection; prior checkpoints, thresholds, calibration parameters and manuscript results were discarded. The formal workflow audited 5,856 real Kermany source images, removed 32 redundant exact duplicates and assigned 5,824 retained images by inferred patient/group. (b) The post hoc PneumoniaMNIST provenance audit identified a held-out-source-matched same-source stress subset of 108 transformed images from 68 groups. This branch evaluates preprocessing and threshold stability, not independent clinical generalization.

## Figure 2
Internal held-out-test discrimination and calibration. (a) ROC and (b) precision–recall curves were calculated from saved frozen-model predictions. (c,d) Reliability diagrams compare raw and validation-temperature-scaled probabilities using 10 equal-width bins; the diagonal denotes perfect calibration. Temperature was fitted only on internal validation logits.

## Figure 3
Paired probability shift after the PneumoniaMNIST 28×28 transformation. Each point links a confirmed internally held-out source radiograph to its transformed counterpart. Blue denotes NORMAL (n=32) and red denotes PNEUMONIA (n=76). The diagonal denotes no probability shift; purple vertical and horizontal lines denote the frozen internal-validation thresholds (ResNet18, 0.450; MobileNetV3-Small, 0.345). Crossing a purple line changes the fixed-threshold classification. Jitter was applied only for display.

## Figure 4
ResNet18 performance across five repeated stratified patient/group splits. Each blue point is an independently initialized repeat; red diamonds and bars show the mean and sample standard deviation. Every repeat reconstructed train, validation and test groups and selected its operating threshold on validation predictions only.

## Supplementary Figure S1
Stratified confirmed source-mapping examples. Within each official subtype (NORMAL, BACTERIA and VIRUS), one confirmed top-1 match was selected from the higher and one from the lower end of the SSIM distribution. Columns show PneumoniaMNIST 28×28 input, matched Kermany source image and the best locally reconstructed 28×28 image. Labels identify internal split, group ID, SSIM and pixel MAE.

## Supplementary Figure S2
Paired logit shifts after 28×28 transformation. Positive values indicate movement toward PNEUMONIA. Both model panels use the same y-axis range; boxplots are overlaid with all observations, and the dashed line denotes no change.

## Supplementary Figure S3
Crossings of the frozen internal-validation operating thresholds. Each model/class appears once. Labels give numerator, denominator, percentage and direction. NORMAL images crossed from below to above the pneumonia threshold in 30/32 cases (93.75%) for both models.

## Supplementary Figure S4
Class-stratified empirical cumulative distributions of pneumonia probabilities before and after 28×28 transformation. ECDFs use proportions rather than raw counts, and dotted lines denote frozen internal-validation thresholds.
