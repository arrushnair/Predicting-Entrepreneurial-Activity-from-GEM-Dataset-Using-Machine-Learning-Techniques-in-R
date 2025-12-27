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
library(h2o)  # Add H2O library

# Initialize H2O
h2o.init()

# Load and preprocess the data
data <- read_dta("C:/Users/ady27/Desktop/combined2/2018NME.dta")
data <- data[complete.cases(data), ]

# Filter for India
data <- subset(data, country == 91)
if (nrow(data) == 0) {
  stop("No rows found with country = 'India'. Exiting script.")
}

# Convert target variable to factor
data$TEAyyNEC_factor <- factor(data$TEAyyNEC, levels = c(0, 1), labels = c("No", "Yes"))

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
train_x <- train[, !names(train) %in% c("TEAyyNEC_factor", "TEAyyNEC")]
train_y <- as.numeric(train$TEAyyNEC_factor) - 1
set.seed(1234)
smote_output <- SMOTE(X = train_x, target = train_y, K = 5, dup_size = 8)
over <- data.frame(smote_output$data)
names(over)[ncol(over)] <- "class"
over$TEAyyNEC_factor <- factor(over$class, levels = c(0, 1), labels = c("No", "Yes"))
over$class <- NULL

# One-hot encoding
over_matrix <- model.matrix(TEAyyNEC_factor ~ . - 1, data = over)
test_matrix <- model.matrix(TEAyyNEC_factor ~ . - 1, data = test)
test_matrix <- test_matrix[, colnames(over_matrix), drop = FALSE]
test_matrix[is.na(test_matrix)] <- 0

# Function to calculate TP, FP, TN, FN
calculate_confusion_stats <- function(true, pred) {
  tp <- sum(true == "Yes" & pred == "Yes")
  tn <- sum(true == "No" & pred == "No")
  fp <- sum(true == "No" & pred == "Yes")
  fn <- sum(true == "Yes" & pred == "No")
  
  return(data.frame(
    Class = c("Yes", "No"),
    TP = c(tp, tn),
    FP = c(fp, fn),
    TN = c(tn, tp),
    FN = c(fn, fp)
  ))
}

evaluate_model <- function(pred, pred_prob, true_labels, model_name) {
  confusion <- confusionMatrix(pred, true_labels)
  roc_curve <- roc(true_labels, as.numeric(pred_prob))
  auc_value <- auc(roc_curve)
  metrics <- data.frame(
    Model = model_name,
    Accuracy = confusion$overall["Accuracy"],
    Recall = confusion$byClass["Sensitivity"],
    Precision = confusion$byClass["Positive Predictive Value"],
    F1_Score = F1_Score(y_true = true_labels, y_pred = pred, positive = "Yes"),
    AUC = auc_value
  )
  
  confusion_stats <- calculate_confusion_stats(true_labels, pred)
  return(list(metrics = metrics, confusion_stats = confusion_stats, roc_curve = roc_curve))
}

# Initialize storage for results
all_metrics <- list()
confusion_tables <- list()
roc_curves <- vector("list", 12)  # Increased to 12 for H2O GBM

# Lasso
lasso_model <- glmnet(over_matrix, as.numeric(over$TEAyyNEC_factor) - 1, alpha = 1, family = "binomial")
pred_prob <- predict(lasso_model, newx = test_matrix, type = "response", s = 0.01)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "Lasso")
all_metrics[[1]] <- result$metrics
confusion_tables[[1]] <- data.frame(Model = "Lasso", result$confusion_stats)
roc_curves[[1]] <- result$roc_curve

# Ridge
ridge_model <- glmnet(over_matrix, as.numeric(over$TEAyyNEC_factor) - 1, alpha = 0, family = "binomial")
pred_prob <- predict(ridge_model, newx = test_matrix, type = "response", s = 0.01)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "Ridge")
all_metrics[[2]] <- result$metrics
confusion_tables[[2]] <- data.frame(Model = "Ridge", result$confusion_stats)
roc_curves[[2]] <- result$roc_curve

# Elastic Net
enet_model <- glmnet(over_matrix, as.numeric(over$TEAyyNEC_factor) - 1, alpha = 0.5, family = "binomial")
pred_prob <- predict(enet_model, newx = test_matrix, type = "response", s = 0.01)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "Elastic Net")
all_metrics[[3]] <- result$metrics
confusion_tables[[3]] <- data.frame(Model = "Elastic Net", result$confusion_stats)
roc_curves[[3]] <- result$roc_curve

# Random Forest
rf_model <- randomForest(over_matrix, as.factor(over$TEAyyNEC_factor), ntree = 100)
pred_prob <- predict(rf_model, newdata = test_matrix, type = "prob")[, "Yes"]
pred <- predict(rf_model, newdata = test_matrix)
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "Random Forest")
all_metrics[[4]] <- result$metrics
confusion_tables[[4]] <- data.frame(Model = "Random Forest", result$confusion_stats)
roc_curves[[4]] <- result$roc_curve

# ANN
ann_model <- nnet(over_matrix, as.numeric(over$TEAyyNEC_factor) - 1, size = 5, decay = 0.1, maxit = 100)
pred_prob <- predict(ann_model, newdata = test_matrix)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "ANN")
all_metrics[[5]] <- result$metrics
confusion_tables[[5]] <- data.frame(Model = "ANN", result$confusion_stats)
roc_curves[[5]] <- result$roc_curve

# kNN
pred <- knn(train = over_matrix, test = test_matrix, cl = over$TEAyyNEC_factor, k = 5, prob = TRUE)
pred_prob <- attr(pred, "prob")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "kNN")
all_metrics[[6]] <- result$metrics
confusion_tables[[6]] <- data.frame(Model = "kNN", result$confusion_stats)
roc_curves[[6]] <- result$roc_curve

# XGBoost
xgb_model <- xgboost(data = over_matrix, label = as.numeric(over$TEAyyNEC_factor) - 1, objective = "binary:logistic", nrounds = 100, verbose = 0)
pred_prob <- predict(xgb_model, newdata = test_matrix)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "XGBoost")
all_metrics[[7]] <- result$metrics
confusion_tables[[7]] <- data.frame(Model = "XGBoost", result$confusion_stats)
roc_curves[[7]] <- result$roc_curve

# Decision Tree
dt_model <- rpart(TEAyyNEC_factor ~ ., data = over)
pred_prob <- predict(dt_model, newdata = test, type = "prob")[, "Yes"]
pred <- predict(dt_model, newdata = test, type = "class")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "Decision Tree")
all_metrics[[8]] <- result$metrics
confusion_tables[[8]] <- data.frame(Model = "Decision Tree", result$confusion_stats)
roc_curves[[8]] <- result$roc_curve

# Logistic Regression
logit_model <- glm(TEAyyNEC_factor ~ ., data = over, family = "binomial")
pred_prob <- predict(logit_model, newdata = test, type = "response")
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "Logistic Regression")
all_metrics[[9]] <- result$metrics
confusion_tables[[9]] <- data.frame(Model = "Logistic Regression", result$confusion_stats)
roc_curves[[9]] <- result$roc_curve

# Naive Bayes
naive_bayes_model <- naiveBayes(TEAyyNEC_factor ~ ., data = over)
pred_prob <- predict(naive_bayes_model, newdata = test, type = "raw")[, "Yes"]
pred <- predict(naive_bayes_model, newdata = test)
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "Naive Bayes")
all_metrics[[10]] <- result$metrics
confusion_tables[[10]] <- data.frame(Model = "Naive Bayes", result$confusion_stats)
roc_curves[[10]] <- result$roc_curve

# CatBoost
catboost_data <- catboost.load_pool(data = over_matrix, label = as.numeric(over$TEAyyNEC_factor) - 1)
catboost_model <- catboost.train(catboost_data, params = list(iterations = 100, learning_rate = 0.1, depth = 6, verbose = 0))
catboost_test_data <- catboost.load_pool(data = test_matrix)
pred_prob <- catboost.predict(catboost_model, catboost_test_data)
pred <- ifelse(pred_prob > 0.5, "Yes", "No")
result <- evaluate_model(factor(pred, levels = c("No", "Yes")), pred_prob, test$TEAyyNEC_factor, "CatBoost")
all_metrics[[11]] <- result$metrics
confusion_tables[[11]] <- data.frame(Model = "CatBoost", result$confusion_stats)
roc_curves[[11]] <- result$roc_curve

# H2O GBM
# Convert training data to H2O frame
h2o_train_data <- as.data.frame(over_matrix)
h2o_train_data$TEAyyNEC_factor <- over$TEAyyNEC_factor
h2o_train <- as.h2o(h2o_train_data)

h2o_test_data <- as.data.frame(test_matrix)
h2o_test_data$TEAyyNEC_factor <- test$TEAyyNEC_factor
h2o_test <- as.h2o(h2o_test_data)

# Set predictor and response columns
predictors <- setdiff(colnames(h2o_train), "TEAyyNEC_factor")
response <- "TEAyyNEC_factor"

# Train H2O GBM model
h2o_gbm_model <- h2o.gbm(
  x = predictors,
  y = response,
  training_frame = h2o_train,
  ntrees = 100,
  max_depth = 6,
  learn_rate = 0.1,
  seed = 1234
)

# Make predictions
h2o_predictions <- h2o.predict(h2o_gbm_model, h2o_test)
h2o_pred_df <- as.data.frame(h2o_predictions)
h2o_pred_prob <- h2o_pred_df$Yes
h2o_pred <- as.character(h2o_pred_df$predict)

# Evaluate H2O GBM
result <- evaluate_model(factor(h2o_pred, levels = c("No", "Yes")), h2o_pred_prob, test$TEAyyNEC_factor, "H2O GBM")
all_metrics[[12]] <- result$metrics
confusion_tables[[12]] <- data.frame(Model = "H2O GBM", result$confusion_stats)
roc_curves[[12]] <- result$roc_curve

# Combine metrics and confusion tables
final_metrics <- do.call(rbind, lapply(all_metrics, function(x) x))
final_confusion <- do.call(rbind, confusion_tables)

# Plot ROC curves
plot(roc_curves[[1]], col = "red", lwd = 2, main = "ROC Curves", xlab = "1 - Specificity", ylab = "Sensitivity")
for (i in 2:length(roc_curves)) {
  plot(roc_curves[[i]], col = rainbow(length(roc_curves))[i], lwd = 2, add = TRUE)
}
legend("bottomright", legend = final_metrics$Model, col = rainbow(length(roc_curves)), lwd = 2, cex = 0.7)

# Print final metrics
print("Final Metrics for all models:")
print(final_metrics)

# Print confusion matrices
print("Confusion Matrices for all models:")
print(final_confusion)

# Shutdown H2O cluster
h2o.shutdown(prompt = FALSE)