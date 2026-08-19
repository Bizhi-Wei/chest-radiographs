# Same-source PneumoniaMNIST preprocessing and resolution-shift stress test

> PneumoniaMNIST is derived from the same underlying Kermany pediatric chest radiograph collection. This is not independent external validation.

- Conservative source-disjoint transformed subset: 108 images from 68 patient/groups
- Source requirement: confirmed local mapping to the internal held-out test split only
- Model weights: frozen clean checkpoints
- Operating thresholds: frozen from internal validation (ResNet18 0.450; MobileNetV3-Small 0.345)
- Temperature scaling: frozen from internal validation
- Uncertainty: 1,000-sample patient/group bootstrap
- No PneumoniaMNIST threshold optimization or temperature refitting was performed

## Metrics

model,stress_test_role,threshold_name,threshold,threshold_source,temperature,temperature_source,n,normal_n,pneumonia_n,accuracy,balanced_accuracy,f1,sensitivity,specificity,ppv,npv,roc_auc,pr_auc,brier,nll,ece,predicted_positive_proportion,tn,fp,fn,tp,brier_raw,nll_raw,ece_raw,brier_temperature_scaled,nll_temperature_scaled,ece_temperature_scaled,ci_method,bootstrap_groups,bootstrap_samples,accuracy_95ci_low,accuracy_95ci_high,balanced_accuracy_95ci_low,balanced_accuracy_95ci_high,f1_95ci_low,f1_95ci_high,sensitivity_95ci_low,sensitivity_95ci_high,specificity_95ci_low,specificity_95ci_high,ppv_95ci_low,ppv_95ci_high,npv_95ci_low,npv_95ci_high,roc_auc_95ci_low,roc_auc_95ci_high,pr_auc_95ci_low,pr_auc_95ci_high,brier_95ci_low,brier_95ci_high,nll_95ci_low,nll_95ci_high,ece_95ci_low,ece_95ci_high,predicted_positive_proportion_95ci_low,predicted_positive_proportion_95ci_high
ResNet18,same-source preprocessing and resolution-shift stress test,default_0.5,0.5,fixed_default,0.4894367690114362,internal_validation_only,108,32,76,0.7037037037037037,0.5,0.8260869565217391,1.0,0.0,0.7037037037037037,,0.8552631578947367,0.925926876371846,0.2763843536376953,1.1573327779769895,0.2806366229498828,1.0,0,32,0,76,0.2763843536376953,1.1573327779769895,0.2806366229498828,0.2946290671825409,2.332592725753784,0.2953330852367259,patient_group_bootstrap,68,1000,0.57284375,0.8063019860804617,0.5,0.5,0.7284177669042898,0.8927652983524851,1.0,1.0,0.0,0.0,0.57284375,0.8063019860804617,,,0.7619259699746945,0.9277214203783632,0.8387460969634637,0.9749817203241998,0.1807429332286119,0.3982848927378654,0.7573272317647934,1.6968329727649687,0.1813125241773193,0.4089119267612695,1.0,1.0
ResNet18,same-source preprocessing and resolution-shift stress test,internal_validation_selected,0.45,internal_validation_only,0.4894367690114362,internal_validation_only,108,32,76,0.7037037037037037,0.5,0.8260869565217391,1.0,0.0,0.7037037037037037,,0.8552631578947367,0.925926876371846,0.2763843536376953,1.1573327779769895,0.2806366229498828,1.0,0,32,0,76,0.2763843536376953,1.1573327779769895,0.2806366229498828,0.2946290671825409,2.332592725753784,0.2953330852367259,patient_group_bootstrap,68,1000,0.57284375,0.8063019860804617,0.5,0.5,0.7284177669042898,0.8927652983524851,1.0,1.0,0.0,0.0,0.57284375,0.8063019860804617,,,0.7619259699746945,0.9277214203783632,0.8387460969634637,0.9749817203241998,0.1807429332286119,0.3982848927378654,0.7573272317647934,1.6968329727649687,0.1813125241773193,0.4089119267612695,1.0,1.0
MobileNetV3-Small,same-source preprocessing and resolution-shift stress test,default_0.5,0.5,fixed_default,0.4773525321279752,internal_validation_only,108,32,76,0.7037037037037037,0.5,0.8260869565217391,1.0,0.0,0.7037037037037037,,0.911595394736842,0.9643848281615144,0.1793762594461441,0.5258530378341675,0.1988019976350996,1.0,0,32,0,76,0.1793762594461441,0.5258530378341675,0.1988019976350996,0.226577028632164,0.6658322811126709,0.2216421001487308,patient_group_bootstrap,68,1000,0.57284375,0.8063019860804617,0.5,0.5,0.7284177669042898,0.8927652983524851,1.0,1.0,0.0,0.0,0.57284375,0.8063019860804617,,,0.8320493278833272,0.964323812947111,0.9157030499753276,0.9877628252339262,0.129617154225707,0.2467801198363303,0.4149690769612789,0.6783855438232422,0.1275594829698481,0.289744776694142,1.0,1.0
MobileNetV3-Small,same-source preprocessing and resolution-shift stress test,internal_validation_selected,0.345,internal_validation_only,0.4773525321279752,internal_validation_only,108,32,76,0.7037037037037037,0.5,0.8260869565217391,1.0,0.0,0.7037037037037037,,0.911595394736842,0.9643848281615144,0.1793762594461441,0.5258530378341675,0.1988019976350996,1.0,0,32,0,76,0.1793762594461441,0.5258530378341675,0.1988019976350996,0.226577028632164,0.6658322811126709,0.2216421001487308,patient_group_bootstrap,68,1000,0.57284375,0.8063019860804617,0.5,0.5,0.7284177669042898,0.8927652983524851,1.0,1.0,0.0,0.0,0.57284375,0.8063019860804617,,,0.8320493278833272,0.964323812947111,0.9157030499753276,0.9877628252339262,0.129617154225707,0.2467801198363303,0.4149690769612789,0.6783855438232422,0.1275594829698481,0.289744776694142,1.0,1.0


## Paired preprocessing shift

### ResNet18

- Mean probability shift: 0.3013
- Median probability shift: 0.0078
- Prediction flip rate: 0.3241
- Internal-threshold crossing rate: 0.3241

### MobileNetV3-Small

- Mean probability shift: 0.1233
- Median probability shift: -0.0595
- Prediction flip rate: 0.2870
- Internal-threshold crossing rate: 0.2870

## Scientific interpretation

The compliant question is: How stable are model ranking, probability scale, and fixed operating thresholds when the same underlying source radiographs undergo severe resolution reduction and preprocessing changes?

No independent external clinical cohort was evaluated. These results cannot establish cross-hospital generalization or clinical portability.
