# Grad-CAM error analysis

Dataset: `<project-root>/data/chest_xray_patient_split_dedup`
Model: `ResNet18`
Checkpoint: `output_clean_20260706/resnet18/best_resnet18.pth`
Operating threshold: 0.450 (Best F1 on validation)

## Confusion matrix at the analyzed operating point

- TN: 229
- FP: 8
- FN: 15
- TP: 621

## Audit design

For each category, the most confident cases were selected: high pneumonia probability for TP/FP and low pneumonia probability for TN/FN. Grad-CAM was generated for the predicted class, because the goal is to inspect the evidence that drove the model decision.

Artifact-risk flags are heuristic screeners, not final judgments. A case is flagged when Grad-CAM heat is unusually concentrated near image borders or corners. Flagged cases should be manually reviewed for edge artifacts, labels, markers, cropping, diaphragm borders or non-lung cues.

## Selected case counts

- FP: total=8, reviewed=8, artifact-risk flags=7
- FN: total=15, reviewed=8, artifact-risk flags=0
- TP: total=621, reviewed=8, artifact-risk flags=0
- TN: total=229, reviewed=8, artifact-risk flags=0

## Outputs

- Case-level audit table: `reports/gradcam_error_analysis/clean_20260706_resnet18_gradcam_error_cases.csv`
- FP contact sheet: `reports/gradcam_error_analysis/clean_20260706_resnet18_gradcam_fp_top_cases.png`
- FN contact sheet: `reports/gradcam_error_analysis/clean_20260706_resnet18_gradcam_fn_top_cases.png`
- TP contact sheet: `reports/gradcam_error_analysis/clean_20260706_resnet18_gradcam_tp_top_cases.png`
- TN contact sheet: `reports/gradcam_error_analysis/clean_20260706_resnet18_gradcam_tn_top_cases.png`

## Suggested manuscript wording

Grad-CAM was used as an error-analysis tool rather than only as a representative visualization. False-positive, false-negative, true-positive and true-negative test cases were stratified at the validation-selected operating threshold, and the most confident cases in each category were reviewed. Heatmap concentration near image borders and corners was quantified to screen for possible attention to non-pulmonary artifacts, labels, markers or cropping effects. These qualitative maps were interpreted as hypothesis-generating model-audit evidence rather than proof of causal localization.