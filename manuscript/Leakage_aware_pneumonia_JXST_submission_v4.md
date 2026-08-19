# Leakage-aware evaluation of deep learning for pediatric pneumonia classification on chest radiographs



Lei Tang<sup>1</sup>, Bizhi Wei<sup>2,*</sup> ([ORCID](https://orcid.org/0009-0008-9481-3024))



<sup>1</sup> School of Economics and Management, Kunming University of Science and Technology Oxbridge College, No. 1369 Yunqiao Street, Guandu District, Kunming 650211, Yunnan Province, China



<sup>2</sup> Puai Medical College, Shaoyang University, Shaoyang 422000, Hunan Province, China



<sup>*</sup>Correspondence: Bizhi Wei, Puai Medical College, Shaoyang University, Shaoyang 422000, Hunan Province, China. Email: 15619056250wbz@gmail.com



## Abstract

**Objective:** To evaluate pediatric pneumonia classification under leakage-controlled internal validation and same-source preprocessing and resolution shift, and to determine whether validation-derived probability calibration and decision thresholds remain stable after 28×28 transformation.

**Methods:** We audited 5,856 Kermany radiographs, removed 32 exact duplicates, and split 5,824 images by inferred patient/group. ResNet18 and MobileNetV3-Small used operating thresholds and temperature scaling fitted only on internal validation data. Five repeated stratified group splits assessed robustness. Group-bootstrap resampling provided 95% confidence intervals. Following provenance mapping, 108 PneumoniaMNIST images from 68 held-out groups formed a post hoc same-source 28×28 preprocessing stress subset.

**Results:** On 873 internally held-out images, ResNet18 achieved accuracy 0.9737 (95% confidence interval [CI], 0.9628–0.9839), F1 0.9818 (0.9739–0.9889), receiver operating characteristic area under the curve 0.9968 (0.9948–0.9985), and precision-recall area under the curve 0.9988 (0.9980–0.9995). MobileNetV3-Small also showed high internal discrimination, and repeated splits supported robustness. Under the 28×28 transformation, both frozen models predicted all 108 images as pneumonia at default and validation-selected thresholds, yielding zero specificity. Normal-image probabilities shifted disproportionately toward pneumonia.

**Conclusion:** Leakage-aware internal evaluation produced high discrimination, whereas same-source preprocessing stress testing revealed instability in probability scale and fixed thresholds. PneumoniaMNIST was not an independent external cohort; external clinical validation remains necessary.



**Keywords:** pediatric pneumonia; chest radiograph; data leakage; patient/group split; calibration; provenance audit; preprocessing shift



## Introduction

Chest radiographs are widely used in the assessment of suspected pediatric pneumonia, and public datasets have enabled rapid development of image classifiers [1,2]. Yet an apparently independent image-level test set can remain correlated with training data through repeated patients, exact or near-duplicate images and processing artefacts [8,9,16]. These routes can inflate performance or encourage shortcut learning that is unrelated to pathology [10,14].

A leakage-aware study must align the resampling unit, data split and uncertainty estimate with the independent clinical unit. It must also keep model selection, threshold selection and calibration fitting within the training/validation workflow rather than treating the final test set as a tuning resource. Reporting guidance for medical-imaging and prediction-model studies likewise emphasizes transparent data provenance, partitioning, model selection and validation [12,13].

High discrimination on one internal distribution does not establish transportability. Institution, acquisition and preprocessing changes can alter score distributions even when rank ordering remains partly preserved [7,11,15]. A fixed operating threshold can therefore fail despite a respectable ROC AUC. Such stress testing is scientifically distinct from external validation, which requires appropriately independent patients and settings.

PneumoniaMNIST is a 28×28 transformed benchmark derived from the same underlying Kermany pediatric chest-radiograph collection [1,2] and does not constitute an independent external cohort. We first rebuilt the internal workflow from a clean source inventory, exact-deduplicated patient/group splits and frozen validation-derived operating parameters. After identifying PneumoniaMNIST's same-source provenance, we performed a post hoc source-overlap audit and evaluated only confirmed transformed counterparts of internally held-out radiographs. Our objectives were to quantify trustworthy internal discrimination and calibration, assess robustness across repeated group splits, and determine how severe resolution reduction changes probabilities and threshold decisions without implying cross-institutional generalization.



## Results



### Data audit and leakage-controlled cohort

A prior local working copy contained 746 label-encoded synthetic placeholder images introduced during an earlier local data-processing workflow. These files were not part of the original Kermany source collection and all results derived from that contaminated working copy were discarded, including checkpoints, thresholds, calibration outputs and manuscript values. The formal source inventory contained 5,856 real radiographs. SHA-256 hashing identified 32 redundant exact duplicates; 5,824 retained images were repartitioned by inferred patient/group (Figure 1 and Table 1).

The primary split contained 4,077 training images (1,105 NORMAL and 2,972 PNEUMONIA), 874 validation images (237 and 637) and 873 test images (237 and 636). The corresponding group counts were 2,188, 468 and 461. No inferred patient/group identifier and no SHA-256 hash crossed split boundaries. Near-duplicate screening generated 21,738 algorithmic candidate pairs, including 9,712 cross-split candidates. SSIM was calculated for the strongest 4,000 candidates after 256×256 grayscale LANCZOS resizing. No pair reached SSIM ≥0.95 or ImageNet-ResNet18 embedding cosine similarity ≥0.99. Contact sheets were generated for the top 40 cross-split and top 40 overall candidates; the project record does not establish that a separate formal human adjudication was completed.



![Figure 1](../04_reports/figures_submission_v2/main/Figure_1_workflow.png)



**Figure 1.** Data provenance and evaluation workflow. (a) A contaminated prior local working copy contained 746 label-encoded synthetic placeholders introduced during an earlier local data-processing workflow. Those files were not part of the original Kermany source collection; prior checkpoints, thresholds, calibration parameters and manuscript results were discarded. The formal workflow audited 5,856 real Kermany source images, removed 32 redundant exact duplicates and assigned 5,824 retained images by inferred patient/group. (b) The post hoc PneumoniaMNIST provenance audit identified a held-out-source-matched same-source stress subset of 108 transformed images from 68 groups. This branch evaluates preprocessing and threshold stability, not independent clinical generalization.



**Table 1. Clean data audit and primary patient/group split.**

| Audit item | Images | Detail |
| --- | --- | --- |
| Prior local synthetic placeholders excluded | 746 | Not part of original Kermany source |
| Verified real source images | 5,856 | Formal source inventory |
| Redundant exact duplicates removed | 32 | SHA-256 |
| Retained real images | 5,824 | Exact-deduplicated cohort |
| Train | 4,077 | 1,105 NORMAL; 2,972 PNEUMONIA; 2,188 groups |
| Validation | 874 | 237 NORMAL; 637 PNEUMONIA; 468 groups |
| Test | 873 | 237 NORMAL; 636 PNEUMONIA; 461 groups |
| Cross-split group/hash leakage | 0/0 | Automated PASS |



### Internal held-out-test performance

At the validation-selected threshold of 0.450, ResNet18 achieved accuracy 0.9737 (0.9628–0.9839), F1 0.9818 (0.9739–0.9889), sensitivity 0.9764 (0.9640–0.9876), specificity 0.9662 (0.9417–0.9875), ROC AUC 0.9968 (0.9948–0.9985) and PR AUC 0.9988 (0.9980–0.9995) on the internal test set. MobileNetV3-Small at its validation-selected threshold of 0.345 achieved accuracy 0.9622 (0.9493–0.9741), F1 0.9741 (0.9645–0.9825), sensitivity 0.9764 (0.9634–0.9880), specificity 0.9241 (0.8907–0.9567), ROC AUC 0.9932 (0.9895–0.9963) and PR AUC 0.9976 (0.9961–0.9987) (Table 2; Figure 2a,b). All intervals used 1,000 patient/group bootstrap resamples, and all 1,000 resamples were valid.

Temperature scaling fitted to internal validation logits improved internal calibration. For ResNet18, Brier score/NLL/ECE changed from 0.0225/0.0817/0.0310 to 0.0215/0.0720/0.0148 at T=0.4894367690. For MobileNetV3-Small, the corresponding values changed from 0.0453/0.1691/0.0875 to 0.0388/0.1348/0.0509 at T=0.4773525321 (Table 3; Figure 2c,d).



**Table 2. Internal held-out-test performance.**

| Model | Threshold source | Threshold | Accuracy (95% CI) | F1 (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | ROC AUC (95% CI) | PR AUC (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | Default | 0.5 | 0.9725 (0.9617-0.9826) | 0.9810 (0.9731-0.9879) | 0.9733 (0.9599-0.9855) | 0.9705 (0.9471-0.9885) | 0.9968 (0.9948-0.9985) | 0.9988 (0.9980-0.9995) |
| ResNet18 | Internal validation best F1 | 0.45 | 0.9737 (0.9628-0.9839) | 0.9818 (0.9739-0.9889) | 0.9764 (0.9640-0.9876) | 0.9662 (0.9417-0.9875) | 0.9968 (0.9948-0.9985) | 0.9988 (0.9980-0.9995) |
| MobileNetV3-Small | Default | 0.5 | 0.9450 (0.9290-0.9601) | 0.9610 (0.9483-0.9722) | 0.9308 (0.9078-0.9505) | 0.9831 (0.9662-0.9961) | 0.9932 (0.9895-0.9963) | 0.9976 (0.9961-0.9987) |
| MobileNetV3-Small | Internal validation best F1 | 0.345 | 0.9622 (0.9493-0.9741) | 0.9741 (0.9645-0.9825) | 0.9764 (0.9634-0.9880) | 0.9241 (0.8907-0.9567) | 0.9932 (0.9895-0.9963) | 0.9976 (0.9961-0.9987) |



![Figure 2](../04_reports/figures_submission_v2/main/Figure_2_internal_performance.png)



**Figure 2.** Internal held-out-test discrimination and calibration. (a) ROC and (b) precision–recall curves were calculated from saved frozen-model predictions. (c,d) Reliability diagrams compare raw and validation-temperature-scaled probabilities using 10 equal-width bins; the diagonal denotes perfect calibration. Temperature was fitted only on internal validation logits.



**Table 3. Internal calibration before and after validation-fitted temperature scaling.**

| Model | Probability type | Temperature | Brier score | NLL | ECE |
| --- | --- | --- | --- | --- | --- |
| ResNet18 | raw | 1.0000 | 0.0225 | 0.0817 | 0.0310 |
| ResNet18 | temperature_scaled | 0.4894 | 0.0215 | 0.0720 | 0.0148 |
| MobileNetV3-Small | raw | 1.0000 | 0.0453 | 0.1691 | 0.0875 |
| MobileNetV3-Small | temperature_scaled | 0.4774 | 0.0388 | 0.1348 | 0.0509 |



### Post hoc paired preprocessing-shift analysis

The PneumoniaMNIST provenance audit found 549 confirmed, 1 probable, 48 ambiguous and 26 unmatched mappings among 624 test images. Accepted mappings corresponded to 366 internal training images, 75 validation images and 109 test images. The post hoc held-out-source-matched same-source preprocessing stress test retained 108 confirmed internal-test mappings from 68 groups: 32 NORMAL and 76 PNEUMONIA. No model, threshold or calibration parameter was selected or fitted on this subset (Figure 1b). Failure to map an image was not treated as proof that source overlap was absent.

The transformation shifted ResNet18 probabilities by a mean of +0.3013 (95% CI +0.2041 to +0.4272), with median +0.0078 (IQR -0.0027 to +0.7785) and a prediction-flip rate of 0.3241 (95% CI 0.2114–0.4688). MobileNetV3-Small mean shift was +0.1233 (95% CI +0.0425 to +0.2269), median -0.0595 (IQR -0.1411 to +0.4454), and flip rate 0.2870 (95% CI 0.1869–0.4187). The transformation disproportionately shifted normal-image scores toward the pneumonia class: mean NORMAL shifts were +0.8619 and +0.5719, and 30/32 NORMAL images (93.75%) crossed each frozen threshold (Figure 3; Supplementary Figures S2–S4).



![Figure 3](../04_reports/figures_submission_v2/main/Figure_3_paired_probability_shift.png)



**Figure 3.** Paired probability shift after the PneumoniaMNIST 28×28 transformation. Each point links a confirmed internally held-out source radiograph to its transformed counterpart. Blue denotes NORMAL (n=32) and red denotes PNEUMONIA (n=76). The diagonal denotes no probability shift; purple vertical and horizontal lines denote the frozen internal-validation thresholds (ResNet18, 0.450; MobileNetV3-Small, 0.345). Crossing a purple line changes the fixed-threshold classification. Jitter was applied only for display.



### Repeated stratified group splits

Across five repeated stratified patient/group splits, each repeat rebuilt the partitions and reinitialized ResNet18 from ImageNet weights before training. Mean (SD) validation-threshold test performance was accuracy 0.9668 (0.0026), F1 0.9774 (0.0016), sensitivity 0.9846 (0.0072), specificity 0.9190 (0.0245), ROC AUC 0.9932 (0.0016) and PR AUC 0.9974 (0.0007) (Table 4; Figure 4). Specificity varied more than ranking metrics, highlighting operating-point sensitivity even within the source dataset.



**Table 4. Five repeated stratified patient/group splits.**

| Metric | Mean | SD | Minimum | Maximum |
| --- | --- | --- | --- | --- |
| Accuracy | 0.9668 | 0.0026 | 0.9634 | 0.9703 |
| F1 | 0.9774 | 0.0016 | 0.9753 | 0.9796 |
| Sensitivity | 0.9846 | 0.0072 | 0.9749 | 0.9937 |
| Specificity | 0.9190 | 0.0245 | 0.8819 | 0.9409 |
| ROC AUC | 0.9932 | 0.0016 | 0.9907 | 0.9949 |
| PR AUC | 0.9974 | 0.0007 | 0.9963 | 0.9981 |



![Figure 4](../04_reports/figures_submission_v2/main/Figure_4_repeated_split_performance.png)



**Figure 4.** ResNet18 performance across five repeated stratified patient/group splits. Each blue point is an independently initialized repeat; red diamonds and bars show the mean and sample standard deviation. Every repeat reconstructed train, validation and test groups and selected its operating threshold on validation predictions only.



### Aggregate same-source stress-test performance

At both 0.5 and the frozen internal-validation thresholds, both models labelled all 108 images PNEUMONIA. Thus, at the validation-derived thresholds, accuracy was 0.7037 (0.5728–0.8063), balanced accuracy 0.5000 (0.5000–0.5000), F1 0.8261 (0.7284–0.8928), sensitivity 1.0000 (1.0000–1.0000), specificity 0.0000 (0.0000–0.0000) and PPV 0.7037 (0.5728–0.8063) for both models. NPV was undefined because neither model produced a negative prediction. ResNet18 ROC AUC was 0.8553 (0.7619–0.9277) and PR AUC 0.9259 (0.8387–0.9750); MobileNetV3-Small ROC AUC was 0.9116 (0.8320–0.9643) and PR AUC 0.9644 (0.9157–0.9878) (Table 5).

Internal-validation temperature scaling did not transfer to the transformed distribution. ResNet18 raw Brier/NLL/ECE values of 0.2764/1.1573/0.2806 changed to 0.2946/2.3326/0.2953 after applying the frozen temperature. MobileNetV3-Small raw values of 0.1794/0.5259/0.1988 changed to 0.2266/0.6658/0.2216. Temperature was not refitted on PneumoniaMNIST. These results indicate that calibration parameters are distribution-dependent; they do not show that temperature scaling is intrinsically ineffective.



**Table 5. Performance on the held-out-source-matched same-source PneumoniaMNIST preprocessing stress subset.**

| Model | Threshold source | Threshold | Accuracy (95% CI) | Balanced accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | ROC AUC (95% CI) | PR AUC (95% CI) | Brier (95% CI) | ECE (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | Default | 0.5 | 0.7037 (0.5728-0.8063) | 0.5000 (0.5000-0.5000) | 1.0000 (1.0000-1.0000) | 0.0000 (0.0000-0.0000) | 0.8553 (0.7619-0.9277) | 0.9259 (0.8387-0.9750) | 0.2764 (0.1807-0.3983) | 0.2806 (0.1813-0.4089) |
| ResNet18 | Internal validation best F1 | 0.45 | 0.7037 (0.5728-0.8063) | 0.5000 (0.5000-0.5000) | 1.0000 (1.0000-1.0000) | 0.0000 (0.0000-0.0000) | 0.8553 (0.7619-0.9277) | 0.9259 (0.8387-0.9750) | 0.2764 (0.1807-0.3983) | 0.2806 (0.1813-0.4089) |
| MobileNetV3-Small | Default | 0.5 | 0.7037 (0.5728-0.8063) | 0.5000 (0.5000-0.5000) | 1.0000 (1.0000-1.0000) | 0.0000 (0.0000-0.0000) | 0.9116 (0.8320-0.9643) | 0.9644 (0.9157-0.9878) | 0.1794 (0.1296-0.2468) | 0.1988 (0.1276-0.2897) |
| MobileNetV3-Small | Internal validation best F1 | 0.345 | 0.7037 (0.5728-0.8063) | 0.5000 (0.5000-0.5000) | 1.0000 (1.0000-1.0000) | 0.0000 (0.0000-0.0000) | 0.9116 (0.8320-0.9643) | 0.9644 (0.9157-0.9878) | 0.1794 (0.1296-0.2468) | 0.1988 (0.1276-0.2897) |



*Notes:* N=108 images from 68 inferred patient/groups (32 NORMAL; 76 PNEUMONIA). Thresholds were frozen internally. No PneumoniaMNIST threshold optimization or recalibration was performed. NPV is undefined because no negative predictions were produced. CIs used 1,000 patient/group bootstrap resamples.



## Discussion

Leakage-aware internal evaluation retained high discrimination after removing a contaminated local working copy, exact duplicates and patient/group overlap. Confidence intervals based on patient/group rather than image resampling were narrow for ranking metrics, and repeated group splits showed limited variation in ResNet18 ROC and PR AUC. These findings support the reproducibility of internal discrimination within this source collection, not clinical transportability.

The placeholder correction is central to that interpretation. The 746 label-encoded synthetic images were introduced in an earlier local processing workflow and were not part of the public Kermany data. Had they remained, a model could have inherited a label-encoded shortcut; no formal contamination-versus-clean ablation was used to make a stronger causal claim. Discarding all dependent checkpoints and results prevented the previous working copy from contributing to the formal analysis.

PneumoniaMNIST cannot serve as an independent external cohort because it is transformed from the same Kermany source radiographs. The source audit further showed direct counterparts in all three internal partitions. Restricting the exploratory analysis to confirmed internal-test counterparts prevents train/validation source-image and group overlap for this analysis, but does not create institutional independence. The result therefore addresses a narrower question: how stable are model ranking, probability scale and frozen operating thresholds when the same underlying source radiographs undergo severe resolution reduction and preprocessing changes?

The paired analysis gives a more informative answer than aggregate AUC alone. Normal-image probabilities moved sharply toward PNEUMONIA, causing 93.75% of normal images to cross each model's frozen threshold. This produced zero specificity while ROC and PR AUC remained above chance. Applying validation-fitted temperatures also worsened transformed-distribution calibration. Together, these findings illustrate why discrimination, calibration and operating-point transport should be reported separately [7,11].

This study has limitations. Patient/group identifiers were inferred from filenames rather than verified clinical identifiers. The provenance audit left 48 mappings ambiguous and 26 unmatched; absence of a confident local match does not establish absence of overlap. The near-duplicate pipeline generated review contact sheets, but the formal record does not establish completed independent human adjudication. The exact training-time package versions and the terminal epoch of the formal ResNet18 run were not serialized, and the primary ResNet18 script declared but did not apply a complete random-seeding routine. Grad-CAM is qualitative and cannot establish causal image evidence [6]. Finally, no independent institutional cohort, prospective workflow or reader study was evaluated. Independent external clinical validation remains necessary.



## Methods



### Study design and formal result boundary

This retrospective computational evaluation used only formal clean assets from clean_20260706, clean_20260707 and the provenance-corrected outputs. Results derived from prior contaminated outputs, checkpoints or manuscripts were excluded. The primary analysis comprised a single leakage-controlled internal patient/group split, frozen-model evaluation, calibration and five repeated stratified group splits. The PneumoniaMNIST audit and 108-image held-out-source-matched same-source preprocessing stress test were post hoc exploratory analyses initiated after same-source provenance was recognized. No PneumoniaMNIST observation contributed to model selection, threshold selection or temperature fitting.



### Data sources, placeholder exclusion and patient/group inference

The Kermany pediatric chest-radiograph collection was assembled from the available source train/validation/test folders [1]. The original folder split was not retained. The formal inventory identified 5,856 real source radiographs. A prior local working copy additionally contained 746 label-encoded synthetic placeholder images introduced by an earlier local data-processing workflow; they were not part of the original Kermany collection and were excluded before formal splitting or training. PNEUMONIA filenames were grouped using person identifiers, whereas NORMAL filenames used IM or NORMAL2-IM identifiers; unresolved names would fall back to the file stem and be logged.



### Exact duplicates, group splitting and leakage checks

SHA-256 was calculated for every real source image. Thirty-two redundant exact-duplicate files were removed, leaving 5,824 images. Stratified group splitting assigned every inferred patient/group to exactly one of train, validation and test while approximately preserving class prevalence. Automated checks required zero cross-split patient/group identifiers and zero cross-split SHA-256 hashes. The primary split seed was 42.



### Image preprocessing and augmentation

PIL-loaded images were converted to RGB by torchvision ImageFolder. Training inputs were resized to 256×256, randomly cropped to 224×224, horizontally flipped with probability 0.5, rotated within ±10°, and color-jittered with brightness and contrast factors of 0.2. Validation and test inputs were deterministically resized to 224×224 without random augmentation. All inputs were converted to tensors and normalized with ImageNet channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225).



### Models and optimization

ResNet18 was the primary architecture [3], and MobileNetV3-Small was the lightweight comparator [4]. Under the verified current torchvision 0.21.0 environment, each used the DEFAULT ImageNet weights corresponding to IMAGENET1K_V1. The final classifier was replaced and initialized for two classes. ResNet18 had no added dropout; MobileNetV3-Small retained its default classifier dropout (p=0.2). The formal ResNet18 checkpoint came from the no-MixUp stage, so MixUp did not contribute to reported results. Training used focal loss (α=0.25, γ=2), inverse-frequency WeightedRandomSampler sampling with replacement, AdamW (learning rate 0.0001; weight decay 0.0001) and cosine annealing. ResNet18 used T_max=25, a 25-epoch cap and early stopping after 7 epochs without ≥0.001 improvement in validation accuracy. MobileNetV3-Small used T_max=20, a 20-epoch cap and patience 5 with the same minimum improvement. Batch size was 32 and DataLoader workers were 2. Checkpoints were selected by maximum validation accuracy. MobileNetV3-Small reached its best validation accuracy at epoch 6 and stopped after epoch 11; the formal record does not recover the exact terminal epoch of the ResNet18 run.



### Randomness and repeated splits

The main split used seed 42. MobileNetV3-Small explicitly seeded Python, NumPy, PyTorch and CUDA with 42. The ResNet18 script declared SEED=42 but did not invoke a complete seeding routine; cuDNN benchmark mode was enabled and deterministic algorithms were not enabled. No explicit DataLoader worker-initialization function was recorded. Repeated split construction used seeds 42, 143, 244, 345 and 446. Each repeat reconstructed the split and reinitialized the complete ResNet18 model from ImageNet weights with a new two-class head; weights were not carried between repeats. A global training seed of 42 was set before the repeated loop, but a repeat-specific training seed was not reset, so exact bitwise repeatability is not claimed. Repeated training used batch size 32, 2 workers, a 15-epoch cap and early-stopping patience 5.



### Validation-only threshold selection

Operating thresholds were selected from validation predictions and frozen before test evaluation. In the primary workflow, candidates ranged from 0.010 to 0.995 in increments of 0.005. The maximum F1 candidate was chosen; ties were resolved by higher balanced accuracy and then lower threshold. This produced 0.450 for ResNet18 and 0.345 for MobileNetV3-Small. The repeated-split workflow searched 201 values from 0 to 1 in increments of 0.005 and retained the first maximum-F1 threshold, equivalent to the lower threshold on a tie. The test set and PneumoniaMNIST were never used to optimize thresholds.



### Metrics and patient/group bootstrap

Metrics included accuracy, balanced accuracy, F1, sensitivity, specificity, PPV, NPV, ROC AUC, PR AUC, Brier score, negative log likelihood (NLL), expected calibration error (ECE) and confusion-matrix counts. PNEUMONIA was the positive class. Ninety-five per cent confidence intervals were percentile intervals from 1,000 bootstrap resamples of patient/group identifiers with replacement using seed 20260805. All images belonging to a sampled group were retained together. Resamples that could not support a metric because only one class was present were excluded for that metric and the valid count recorded. All 1,000 primary and stress-test resamples were valid for reported two-class ranking metrics. NPV was left undefined when no negative prediction occurred.



### Calibration

Temperature scaling used internal validation logits only [5]. A scalar log-temperature was optimized against multinomial log loss/NLL by scipy.optimize.minimize with Nelder–Mead, an initial log-temperature of 0 and a maximum of 200 iterations. Temperature was constrained positive by exponentiation and floored at 0.001. Brier score used the mean squared error between pneumonia probability and the binary outcome; NLL used log loss. ECE used 10 equal-width probability bins and the prevalence-weighted absolute difference between mean confidence and event rate. Calibration uncertainty used 1,000 patient/group bootstrap resamples.



### Near-duplicate and Grad-CAM audits

The near-duplicate audit computed 64-bit average hash, difference hash and DCT perceptual hash. A pair entered the hash candidate set when dHash distance ≤8, pHash distance ≤10 or aHash distance ≤8. Up to the strongest 2,500 hash candidates were retained, and L2-normalized 512-dimensional ImageNet-ResNet18 embeddings supplied the top three neighbours per image. After combining candidate sources and removing duplicate pair identities, 21,738 algorithmic candidates remained. SSIM used 256×256 grayscale LANCZOS images, an 11×11 Gaussian window and σ=1.5 for the strongest 4,000 pairs. Contact sheets contained the top 40 cross-split and top 40 overall candidates; no completed formal human adjudication record was found. Grad-CAM [6] was generated for test true positives, true negatives, false positives and false negatives to inspect whether activation localized to lung fields or to boundaries, markers and artefacts. It was treated as qualitative error-audit evidence rather than causal explanation.



### PneumoniaMNIST provenance and source-overlap audit

PneumoniaMNIST train, validation and test arrays and official source metadata were inspected [2]. Because local Kaggle-style filenames differed from official MedMNIST source IDs, test images were compared only with label- and subtype-compatible Kermany candidates after six prespecified reconstructions: direct PIL bilinear resize; aspect-fill PIL bilinear; center-crop PIL bilinear; center-crop PIL bicubic; center-crop OpenCV area; and center-crop scikit-image linear interpolation. Diagnostic min–max normalization and inversion were recorded separately. Matching used pixel MAE/MSE, normalized cross-correlation, SSIM, aHash, pHash, dHash and bidirectional top-1 consistency. Confirmed matches required subtype agreement, mutual top-1, MSE≤100, MAE≤8, NCC≥0.985, SSIM≥0.95, pHash≤12, dHash≤14 and a second-to-first MSE ratio≥1.2. Probable and ambiguous thresholds were prespecified in the audit report. Matches were classified as confirmed, probable, ambiguous or unmatched; no match was forced.



### Held-out-source-matched same-source preprocessing stress test

The post hoc stress subset retained only confirmed PneumoniaMNIST test mappings whose Kermany source radiograph was in the internal held-out test split, whose inferred group occurred in neither internal training nor validation, and whose source image had not informed checkpoint, threshold or temperature selection. The subset contained 108 images from 68 groups. Frozen ResNet18 and MobileNetV3-Small checkpoints were evaluated at threshold 0.5 and at their internal-validation thresholds. Internal-validation temperatures were applied only as a transfer analysis and were not refitted. Paired source-versus-transformed probability and logit differences, prediction flips and threshold crossings were summarized overall and by class with paired patient/group percentile bootstrap intervals.



### Software and hardware

The currently verified reproducibility environment was Python 3.12.13, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, scikit-learn 1.9.0 and CUDA 12.4 on an NVIDIA GeForce RTX 4070 Laptop GPU. The host was Windows 11 Pro for Workstations (10.0.26200, build 26200) with WSL2 Linux kernel 6.18.33.2 and NVIDIA driver 610.74. Exact training-time package versions were not serialized; these values describe the verified environment used for the frozen-model recomputation.



## Conclusion

Leakage-aware internal evaluation produced high discrimination, while a same-source preprocessing stress test demonstrated substantial instability of probability scale and fixed operating thresholds. Independent external clinical validation remains necessary.



## Data availability

The Kermany radiographs and PneumoniaMNIST arrays are publicly available from their original providers and remain subject to the original licences and terms of use. The present study does not redistribute the original images or arrays. De-identified derived inventories, hashes, split manifests, provenance mappings, predictions, bootstrap outputs and figure-generation inputs are available at https://github.com/Bizhi-Wei/chest-radiographs.



## Code availability

The analysis code, provenance-audit scripts, evaluation scripts, consistency tests and derived manifests are publicly available at https://github.com/Bizhi-Wei/chest-radiographs. Formal project scripts include `02_scripts/26_audit_pneumoniamnist_provenance.py`, `02_scripts/27_evaluate_pneumoniamnist_source_disjoint_stress.py`, `02_scripts/28_prepare_submission_revision_v2.py` and `02_scripts/29_prepare_public_release.py`. The workflow reloads frozen clean checkpoints and regenerates predictions, group-bootstrap intervals, tables and figures without training.



## Ethics approval

This study used only publicly available, de-identified datasets and involved no direct interaction with human participants. Ethics approval and informed consent were therefore not required.



## Funding

This research received no external funding.



## Acknowledgements

None.



## Author contributions

Lei Tang: Conceptualization, Methodology, Supervision, Writing – original draft, and Writing – review & editing. Bizhi Wei: Conceptualization, Data curation, Software, Formal analysis, Investigation, Validation, Visualization, Project administration, Writing – original draft, and Writing – review & editing. All authors read and approved the final manuscript.



## Competing interests

The authors declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.



## References

1. Kermany DS, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell. 2018;172:1122–1131.e9. doi:10.1016/j.cell.2018.02.010.

2. Yang J, et al. MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification. Sci Data. 2023;10:41. doi:10.1038/s41597-022-01721-8.

3. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition; 2016. p. 770–778. doi:10.1109/CVPR.2016.90.

4. Howard A, et al. Searching for MobileNetV3. In: Proceedings of the IEEE/CVF International Conference on Computer Vision; 2019. p. 1314–1324. doi:10.1109/ICCV.2019.00140.

5. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. Proc Mach Learn Res. 2017;70:1321–1330.

6. Selvaraju RR, et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision; 2017. p. 618–626. doi:10.1109/ICCV.2017.74.

7. Zech JR, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS Med. 2018;15:e1002683. doi:10.1371/journal.pmed.1002683.

8. Tampu IE, Eklund A, Haj-Hosseini N. Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images. Sci Data. 2022;9:580. doi:10.1038/s41597-022-01618-6.

9. Roberts M, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nat Mach Intell. 2021;3:199–217. doi:10.1038/s42256-021-00307-0.

10. Geirhos R, et al. Shortcut learning in deep neural networks. Nat Mach Intell. 2020;2:665–673. doi:10.1038/s42256-020-00257-z.

11. Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230. doi:10.1186/s12916-019-1466-7.

12. Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2:e200029. doi:10.1148/ryai.2020200029.

13. Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.

14. Oakden-Rayner L, et al. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. In: Proceedings of the ACM Conference on Health, Inference, and Learning; 2020. p. 151–159. doi:10.1145/3368555.3384468.

15. Debray TPA, et al. A framework for developing, implementing, and evaluating clinical prediction models in an individual participant data meta-analysis. J Clin Epidemiol. 2015;68:279–289. doi:10.1016/j.jclinepi.2014.06.018.

16. Maguolo G, Nanni L. A critic evaluation of methods for COVID-19 automatic detection from X-ray images. Inf Fusion. 2021;76:1–7. doi:10.1016/j.inffus.2021.04.008.
