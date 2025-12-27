# Predicting Entrepreneurial Activity Using Supervised Machine Learning

This repository presents a complete, end-to-end workflow for predicting entrepreneurial activity using supervised machine learning techniques. The project integrates dataset construction in Stata, model development in R, and sampling methods to address class imbalance, with a specific focus on Opportunity-Motivated Entrepreneurship (OME) and Necessity-Motivated Entrepreneurship (NME).

Datasets are in this drive link: https://drive.google.com/drive/folders/14dzzPFUuVTIxWjBv3t2_P01-zi2smbGm?usp=drive_link

## 1. Machine Learning Models Used

The study implements a diverse set of classification algorithms:

- Lasso Regression
- Ridge Regression
- Elastic Net
- Logistic Regression
- Random Forest
- Decision Trees
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN)
- XGBoost
- CatBoost
- Naive Bayes
- H2O Gradient Boosting Machine (GBM)

## 2. Sampling Technique for Class Imbalance

Due to severe class imbalance in OME and NME outcomes, SMOTE (Synthetic Minority Over-sampling Technique) is applied during preprocessing in every R script to ensure adequate minority class representation.

## 3. Workflow

### Step 1: Dataset Generation in Stata
- Run yearly Stata .do files for 2015–2020.
- Merge yearly outputs into a combined dataset.

**Target Variables**
- OME: TEAyyOPP
- NME: TEAyyNEC

**Mandatory Adjustments (2019–2020, NME)**
- teayymot1yes → teayymot3yes
- teayymot2yes → teayymot4yes

### Step 2: Machine Learning in R

Load dataset:
```
library(haven)
df <- read_dta("combined_dataset(2015-2020).dta")
```

Apply SMOTE:
```
library(DMwR)
df_balanced <- SMOTE(target_variable ~ ., data = df)
```

Run the provided scripts for:
- Global OME & NME
- India-only OME & NME
- Pooled and year-wise analysis

### Step 3: India-Specific Filtering
```
df_india <- subset(df, country == 91)
```

## 4. Outputs
- Confusion matrices
- ROC curves and AUC
- Accuracy, Precision, Recall, F1-score
- Year-wise Results (2015–2020)
- Pooled results (2015-2020)
- Comparative model performance tables

## 5. Notes
- Ensure Stata and R are properly installed.
- Verify all package dependencies.
- Year-specific variable adjustments are mandatory for NME (2019–2020).
- Validate datasets before merging.

