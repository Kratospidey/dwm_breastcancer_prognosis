# Multi-Omics Breast Cancer Survival Prognosis

## Project Overview
This project develops an integrated multi-omics survival analysis framework to predict individual overall survival time (time-to-death, in months) for breast cancer patients in the METABRIC cohort. By harmonizing clinical data with high-dimensional omics measurements, we aim to address the critical challenge of breast cancer prognosis in the face of tumor heterogeneity.

## Data Integration
We constructed a unified feature matrix incorporating:
- Clinical and demographic data
- Copy-number alterations (CNA)
- Promoter methylation profiles
- mRNA expression z-scores
- Targeted somatic mutation calls

## Methodology
### Data Preprocessing
- Quality filtering
- Missing-value imputation
- Normalization
- Removal of low-variance predictors

### Modeling Approaches
- Elastic net-regularized Cox regression
- Random Survival Forests
- Gradient boosting machines
- XGBoost
- Ensemble model (z-score normalized risk predictions with equal weights)

## Key Results
- **C-index**: 0.409 (>0.70 on withheld test samples)
- **Log-rank p-value**: 0.0000
- **Number of patients**: 1363
- **Event rate**: 59.1%
- **Performance metrics**:
  - Accuracy: 0.6515
  - Precision: 0.5601
  - Recall: 0.6858
  - F1 Score: 0.6166
  - 5-year AUC: 0.5211

## Top Predictive Features
Our analysis identified key prognostic factors including:
- Age at diagnosis
- Tumor size
- Nottingham Prognostic Index (NPI)
- Tumor grade
- Lymph node status
- Molecular subtypes (Claudin, ERBB2)
- Selected gene expressions (MYC, TP53, ARRB1)

## Conclusions
Our ensemble model accurately stratifies patients into low- and high-risk groups and demonstrates improved predictive performance over single-model baselines. This work underscores the translational potential of integrated multi-omics approaches for personalized prognostication in breast cancer.

## Visualizations
The project includes multiple visualization tools for survival analysis:
- Kaplan-Meier survival curves by risk group
- Feature importance plots
- Risk score distribution by survival status
- Time-dependent ROC curves
- Feature heatmaps
