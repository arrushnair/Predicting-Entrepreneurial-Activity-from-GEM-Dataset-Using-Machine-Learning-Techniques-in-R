This document provides a structured overview of the workflow for predicting entrepreneurial activity using supervised machine learning techniques. The process combines dataset generation in Stata, model development in R, and the use of sampling methods to mitigate class imbalance in OME and NME classifications.

1. Machine Learning Models Used
The analysis incorporates eleven commonly used classification algorithms:

Lasso Regression ,Ridge Regression,Elastic Net, Logistic Regression
Random Forest,Artificial Neural Network (ANN),K-Nearest Neighbors (KNN), Decision Trees, XGBoost, CatBoost, Logistic Regression, Naive Bayes, H2O GBM

These models were selected to cover a broad spectrum of linear, probabilistic, tree-based, and ensemble approaches.

2. Sampling Techniques Applied
Because entrepreneurial activity—particularly OME and NME—is highly imbalanced in the dataset, SMOTE (Synthetic Minority Over-sampling Technique) is applied during preprocessing in every R file. This technique helps ensure that minority classes are adequately represented during training.

3. Step-by-Step Workflow
Step 1: Dataset Generation in Stata
Execute the provided .do files for each year from 2015 to 2020 to generate the annual datasets.
Merge all yearly datasets into a single unified dataset for analysis.

Target Variable Configuration:  
For OME (Opportunity-Motivated Entrepreneurship):Use TEAyyOPP as the dependent variable.

For NME (Necessity-Motivated Entrepreneurship):  Use TEAyyNEC as the dependent variable.

Important Adjustments for 2019 and 2020
For NME processing in 2019–2020, replace OME motivator variables with NME equivalents:
teayymot1yes → teayymot3yes
teayymot2yes → teayymot4yes

These changes must be implemented directly in the corresponding .do files.


Step 2: Running Machine Learning Models in R

After generating and consolidating the dataset in Stata, the next phase involves training and evaluating machine learning models in R.

Procedure:
Load the Dataset
Import the combined dataset into R:
library(haven)
df <- read_dta("combined_dataset(2015-2020.dta")
Apply SMOTE Sampling Technique
library(DMwR)
df_balanced <- SMOTE(target_variable ~ ., data = df)

Train the Machine Learning Models
Execute the R scripts provided to run all twelve machine learning models:
NME (Necessity-Motivated Entrepreneurship)
nme_india.R

nme_india_yearwise.R

nme_world.R

nme_world_yearwise.R

OME (Opportunity-Motivated Entrepreneurship)

ome_india.R

ome_india_yearwise.R

ome_world.R

ome_world_yearwise.R

Each model generates:
Accuracy
Precision, Recall, and F1-score
Confusion matrices
ROC curves and AUC values
Store and Organize Outputs
The pooled data R files print out the outputs while the yearwise ones print and generate .xlsx files


How Filtering the Dataset for the Indian Context was done in R:

Apply a filter based on the country code:

df_india <- subset(df, country == 91)


Repeat the same sampling and model training procedures used on the global dataset.

Export the results separately under an India-specific directory, ensuring that OME and NME outputs are stored independently.

4. Outputs Generated

For both Opportunity-Motivated Entrepreneurship (OME) and Necessity-Motivated Entrepreneurship (NME), the workflow produces the following outputs for:

Global dataset

India-only dataset

Generated Outputs

Confusion matrices for each model

ROC curves and AUC scores

Model accuracy and sensitivity metrics

Year-wise results xlsx files (2015–2020)

Pooled results (2015–2020 combined)

Comparative performance tables across models

These outputs allow detailed evaluation of model behaviour across different years, sampling methods, and country contexts.

5. Important Notes

Ensure that Stata is installed and correctly configured before attempting to run dataset generation scripts.

Verify that R and all required libraries (caret, DMwR, xgboost, catboost, nnet, etc.) are installed.

Double-check dataset paths and variable names in both Stata .do files and R scripts.

When performing NME analysis for 2019–2020, the motivator variable replacements are mandatory.

It is recommended to validate consistency across yearly datasets before merging.
