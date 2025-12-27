# ==========================================
# 1. SETUP AND LIBRARIES
# ==========================================
library(haven)
library(glmnet)
library(randomForest)
library(nnet)
library(class)
library(xgboost)
library(rpart)
library(e1071)
library(caret)
library(pROC)
library(MLmetrics)
library(smotefamily)
library(catboost)
library(h2o)
# 
# Initialize H2O at the start
h2o.init(nthreads = -1)
# 
# ==========================================
# 2. DATA LOADING AND PREPROCESSING
# ==========================================

# NOTE: Ensure this path is correct for your machine
data_path <- "C:/Users/ady27/Desktop/combined2/combined(2015-2020)OME.dta"
data <- read_dta(data_path)
data <- data[complete.cases(data), ]


# Convert target variable to factor
data$TEAyyOPP_factor <- factor(data$TEAyyOPP,
                               levels = c(0, 1),
                               labels = c("No", "Yes"))

# Convert character variables to factors
data <- data.frame(lapply(data, function(x) {
  if (is.character(x)) return(as.factor(x))
  return(x)
}))

# Remove non-numeric and non-factor columns
valid_columns <- sapply(data, function(x) is.numeric(x) || is.factor(x))
data <- data[, valid_columns]

# Train-test split
set.seed(1234)
pd <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[pd == 1, ]
test <- data[pd == 2, ]

# Apply SMOTE
train_x <- train[, !names(train) %in% c("TEAyyOPP_factor", "TEAyyOPP")]
train_y <- as.numeric(train$TEAyyOPP_factor) - 1
set.seed(1234)
smote_output <- SMOTE(X = train_x, target = train_y, K = 5, dup_size = 8)

over <- data.frame(smote_output$data)
names(over)[ncol(over)] <- "class"
over$TEAyyOPP_factor <- factor(over$class,
                               levels = c(0, 1),
                               labels = c("No", "Yes"))
over$class <- NULL

# One-hot encoding (Matrix format for GLMNET, XGBOOST, ANN, KNN)
over_matrix <- model.matrix(TEAyyOPP_factor ~ . - 1, data = over)
test_matrix <- model.matrix(TEAyyOPP_factor ~ . - 1, data = test)
# Align columns
test_matrix <- test_matrix[, colnames(over_matrix), drop = FALSE]
test_matrix[is.na(test_matrix)] <- 0

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

calculate_confusion_stats <- function(true, pred) {
  tp <- sum(true == "Yes" & pred == "Yes")
  tn <- sum(true == "No" & pred == "No")
  fp <- sum(true == "No" & pred == "Yes")
  fn <- sum(true == "Yes" & pred == "No")
  
  return(data.frame(
    Class = c("Yes", "No"),
    TP = c(tp, tn), FP = c(fp, fn),
    TN = c(tn, tp), FN = c(fn, fp)
  ))
}

evaluate_model <- function(pred, pred_prob, true_labels, model_name) {
  # CRITICAL FIX: explicitly set positive class to "Yes"
  confusion <- confusionMatrix(pred, true_labels, mode = "everything", positive = "Yes")
  
  # ROC Curve
  roc_curve <- roc(true_labels, as.numeric(pred_prob), levels = c("No", "Yes"), direction = "<")
  auc_value <- auc(roc_curve)
  
  metrics <- data.frame(
    Model = model_name,
    Accuracy = confusion$overall["Accuracy"],
    Recall = confusion$byClass["Sensitivity"],  # Now correctly tracks "Yes"
    Precision = confusion$byClass["Precision"], # Now correctly tracks "Yes"
    F1_Score = confusion$byClass["F1"],
    AUC = auc_value
  )
  
  confusion_stats <- calculate_confusion_stats(true_labels, pred)
  return(list(metrics = metrics, confusion_stats = confusion_stats, roc_curve = roc_curve))
}

# Initialize storage
all_metrics <- list()
confusion_tables <- list()
roc_curves <- list()

# ==========================================
# 4. MODEL TRAINING
# ==========================================

# --- 1. Lasso ---
lasso_model <- glmnet(over_matrix, as.numeric(over$TEAyyOPP_factor) - 1, alpha = 1, family = "binomial")
pred_prob <- predict(lasso_model, newx = test_matrix, type = "response", s = 0.01)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "Lasso")
all_metrics[[1]] <- result$metrics; confusion_tables[[1]] <- data.frame(Model = "Lasso", result$confusion_stats); roc_curves[[1]] <- result$roc_curve

# --- 2. Ridge ---
ridge_model <- glmnet(over_matrix, as.numeric(over$TEAyyOPP_factor) - 1, alpha = 0, family = "binomial")
pred_prob <- predict(ridge_model, newx = test_matrix, type = "response", s = 0.01)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "Ridge")
all_metrics[[2]] <- result$metrics; confusion_tables[[2]] <- data.frame(Model = "Ridge", result$confusion_stats); roc_curves[[2]] <- result$roc_curve

# --- 3. Elastic Net ---
enet_model <- glmnet(over_matrix, as.numeric(over$TEAyyOPP_factor) - 1, alpha = 0.5, family = "binomial")
pred_prob <- predict(enet_model, newx = test_matrix, type = "response", s = 0.01)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "Elastic Net")
all_metrics[[3]] <- result$metrics; confusion_tables[[3]] <- data.frame(Model = "Elastic Net", result$confusion_stats); roc_curves[[3]] <- result$roc_curve

# --- 4. Random Forest ---
rf_model <- randomForest(over_matrix, as.factor(over$TEAyyOPP_factor), ntree = 100)
pred_prob <- predict(rf_model, newdata = test_matrix, type = "prob")[, "Yes"]
pred <- predict(rf_model, newdata = test_matrix)
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "Random Forest")
all_metrics[[4]] <- result$metrics; confusion_tables[[4]] <- data.frame(Model = "Random Forest", result$confusion_stats); roc_curves[[4]] <- result$roc_curve

# --- 5. ANN ---
ann_model <- nnet(over_matrix, as.numeric(over$TEAyyOPP_factor) - 1, size = 5, decay = 0.1, maxit = 100, trace = FALSE)
pred_prob <- predict(ann_model, newdata = test_matrix)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "ANN")
all_metrics[[5]] <- result$metrics; confusion_tables[[5]] <- data.frame(Model = "ANN", result$confusion_stats); roc_curves[[5]] <- result$roc_curve

# --- 6. kNN ---
pred_knn <- knn(train = over_matrix, test = test_matrix, cl = over$TEAyyOPP_factor, k = 5, prob = TRUE)
pred_prob <- attr(pred_knn, "prob")
pred_indices <- ifelse(pred_knn == "Yes", pred_prob, 1 - pred_prob) # Adjust prob for class
result <- evaluate_model(factor(pred_knn, levels = c("No", "Yes")), pred_indices, test$TEAyyOPP_factor, "kNN")
all_metrics[[6]] <- result$metrics; confusion_tables[[6]] <- data.frame(Model = "kNN", result$confusion_stats); roc_curves[[6]] <- result$roc_curve

# --- 7. XGBoost ---
xgb_model <- xgboost(data = over_matrix, label = as.numeric(over$TEAyyOPP_factor) - 1, objective = "binary:logistic", nrounds = 100, verbose = 0)
pred_prob <- predict(xgb_model, newdata = test_matrix)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "XGBoost")
all_metrics[[7]] <- result$metrics; confusion_tables[[7]] <- data.frame(Model = "XGBoost", result$confusion_stats); roc_curves[[7]] <- result$roc_curve

# --- 8. Decision Tree ---
dt_model <- rpart(TEAyyOPP_factor ~ ., data = over)
pred_prob <- predict(dt_model, newdata = test, type = "prob")[, "Yes"]
pred <- predict(dt_model, newdata = test, type = "class")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "Decision Tree")
all_metrics[[8]] <- result$metrics; confusion_tables[[8]] <- data.frame(Model = "Decision Tree", result$confusion_stats); roc_curves[[8]] <- result$roc_curve

# --- 9. Logistic Regression ---
logit_model <- glm(TEAyyOPP_factor ~ ., data = over, family = "binomial")
pred_prob <- predict(logit_model, newdata = test, type = "response")
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "Logistic Regression")
all_metrics[[9]] <- result$metrics; confusion_tables[[9]] <- data.frame(Model = "Logistic Regression", result$confusion_stats); roc_curves[[9]] <- result$roc_curve

# --- 10. Naive Bayes ---
naive_bayes_model <- naiveBayes(TEAyyOPP_factor ~ ., data = over)
pred_prob <- predict(naive_bayes_model, newdata = test, type = "raw")[, "Yes"]
pred <- predict(naive_bayes_model, newdata = test)
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "Naive Bayes")
all_metrics[[10]] <- result$metrics; confusion_tables[[10]] <- data.frame(Model = "Naive Bayes", result$confusion_stats); roc_curves[[10]] <- result$roc_curve

# --- 11. CatBoost ---
catboost_data <- catboost.load_pool(data = over_matrix, label = as.numeric(over$TEAyyOPP_factor) - 1)
catboost_model <- catboost.train(catboost_data, params = list(iterations = 100, learning_rate = 0.1, depth = 6, logging_level = 'Silent'))
catboost_test_data <- catboost.load_pool(data = test_matrix)
pred_prob <- catboost.predict(catboost_model, catboost_test_data)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "CatBoost")
all_metrics[[11]] <- result$metrics; confusion_tables[[11]] <- data.frame(Model = "CatBoost", result$confusion_stats); roc_curves[[11]] <- result$roc_curve

# --- 12. H2O GBM ---
# HELPER: Strip labels for H2O
clean_for_h2o <- function(df) {
  df[] <- lapply(df, function(x) {
    if (inherits(x, "haven_labelled")) return(as.numeric(x)) else return(x)
  })
  return(df)
}

over_clean <- clean_for_h2o(over)
test_clean <- clean_for_h2o(test)

train_h2o <- as.h2o(over_clean)
test_h2o <- as.h2o(test_clean)
y <- "TEAyyOPP_factor"
x <- setdiff(names(train_h2o), y)

h2o_gbm_model <- h2o.gbm(x = x, y = y, training_frame = train_h2o, ntrees = 100, learn_rate = 0.1, max_depth = 6, seed = 1234)
pred_h2o <- h2o.predict(h2o_gbm_model, newdata = test_h2o)
pred_h2o_df <- as.data.frame(pred_h2o)
pred_prob <- pred_h2o_df$Yes
pred <- pred_h2o_df$predict

result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyOPP_factor, "H2O GBM")
all_metrics[[12]] <- result$metrics; confusion_tables[[12]] <- data.frame(Model = "H2O GBM", result$confusion_stats); roc_curves[[12]] <- result$roc_curve

# ==========================================
# 5. RESULTS AND PLOTTING
# ==========================================

# Combine results
final_metrics <- do.call(rbind, all_metrics)
final_confusion <- do.call(rbind, confusion_tables)

# Plot ROC
plot(roc_curves[[1]], col = "red", lwd = 2, main = "ROC Curves (All Models)", xlab = "1 - Specificity", ylab = "Sensitivity")
colors_list <- rainbow(length(roc_curves))
for (i in 2:length(roc_curves)) {
  plot(roc_curves[[i]], col = colors_list[i], lwd = 2, add = TRUE)
}
legend("bottomright", legend = final_metrics$Model, col = colors_list, lwd = 2, cex = 0.6)

# Print Final Outputs
print("Final Metrics (Positive Class: YES):")
print(final_metrics)

print("Confusion Matrices:")
print(final_confusion)