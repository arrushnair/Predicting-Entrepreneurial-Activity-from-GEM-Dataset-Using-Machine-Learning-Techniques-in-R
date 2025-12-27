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

# Initialize H2O
h2o.init(nthreads = -1)



# ==========================================
# 2. YEARWISE LOOP FOR NME (2015–2020)
# ==========================================

years <- 2015:2020

yearwise_metrics <- list()
yearwise_confusion <- list()

for (yr in years) {
  
  cat("\n==========================================\n")
  cat(" Running NME Models for Year:", yr, "\n")
  cat("==========================================\n\n")
  
  # ------------------------------------------------------
  # ONLY CHANGE 1 — YEARWISE NME FILE INPUT
  # ------------------------------------------------------
  data_path <- paste0("C:/Users/ady27/Desktop/combined2/", yr, "NME.dta")
  data <- read_dta(data_path)
  data <- data[complete.cases(data), ]
  
  
  
  # ==========================================
  # 3. ORIGINAL NME PREPROCESSING (UNCHANGED)
  # ==========================================
  data <- subset(data, country == 91)
  if (nrow(data) == 0) {
    stop("No rows found with country = 'India'. Exiting script.")
  }
  
  # ONLY CHANGE 2 — correct target variable
  data$TEAyyNEC_factor <- factor(data$TEAyyNEC, levels = c(0,1),
                                 labels = c("No","Yes"))
  
  data <- data.frame(lapply(data, function(x) {
    if (is.character(x)) return(as.factor(x))
    return(x)
  }))
  
  valid_columns <- sapply(data, function(x) is.numeric(x) || is.factor(x))
  data <- data[, valid_columns]
  
  set.seed(1234)
  pd <- sample(2, nrow(data), replace = TRUE, prob = c(0.7,0.3))
  train <- data[pd == 1, ]
  test  <- data[pd == 2, ]
  
  # ONLY CHANGE 3 — use NEC columns
  train_x <- train[, !names(train) %in% c("TEAyyNEC_factor", "TEAyyNEC")]
  train_y <- as.numeric(train$TEAyyNEC_factor) - 1
  
  set.seed(1234)
  smote_output <- SMOTE(X=train_x, target=train_y, K=5, dup_size=8)
  
  over <- data.frame(smote_output$data)
  names(over)[ncol(over)] <- "class"
  over$TEAyyNEC_factor <- factor(over$class, levels=c(0,1),
                                 labels=c("No","Yes"))
  over$class <- NULL
  
  over_matrix <- model.matrix(TEAyyNEC_factor ~ . - 1, data = over)
  test_matrix <- model.matrix(TEAyyNEC_factor ~ . - 1, data = test)
  test_matrix <- test_matrix[, colnames(over_matrix), drop=FALSE]
  test_matrix[is.na(test_matrix)] <- 0
  
  
  
  # ==========================================
  # 4. HELPER FUNCTIONS (UNCHANGED)
  # ==========================================
  
  calculate_confusion_stats <- function(true, pred) {
    tp <- sum(true=="Yes" & pred=="Yes")
    tn <- sum(true=="No" & pred=="No")
    fp <- sum(true=="No" & pred=="Yes")
    fn <- sum(true=="Yes" & pred=="No")
    
    return(data.frame(
      Class=c("Yes","No"),
      TP=c(tp,tn),
      FP=c(fp,fn),
      TN=c(tn,tp),
      FN=c(fn,fp)
    ))
  }
  
  evaluate_model <- function(pred, pred_prob, true_labels, model_name) {
    confusion <- confusionMatrix(pred, true_labels)
    roc_curve <- roc(true_labels, as.numeric(pred_prob))
    auc_value <- auc(roc_curve)
    
    metrics <- data.frame(
      Model=model_name,
      Accuracy=confusion$overall["Accuracy"],
      Recall=confusion$byClass["Sensitivity"],
      Precision=confusion$byClass["Positive Predictive Value"],
      F1_Score=F1_Score(y_true=true_labels, y_pred=pred, positive="Yes"),
      AUC=auc_value
    )
    
    confusion_stats <- calculate_confusion_stats(true_labels, pred)
    return(list(metrics=metrics, confusion_stats=confusion_stats,
                roc_curve=roc_curve))
  }
  
  
  
  # ==========================================
  # 5. MODEL TRAINING (100% UNCHANGED)
  # ==========================================
  
  all_metrics <- list()
  confusion_tables <- list()
  roc_curves <- vector("list", 12)
  
  
  # LASSO
  lasso_model <- glmnet(over_matrix, as.numeric(over$TEAyyNEC_factor)-1,
                        alpha=1, family="binomial")
  pred_prob <- predict(lasso_model, newx=test_matrix, type="response", s=0.01)
  pred <- ifelse(pred_prob>0.5, "Yes","No")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "Lasso")
  all_metrics[[1]] <- result$metrics
  confusion_tables[[1]] <- data.frame(Model="Lasso", result$confusion_stats)
  roc_curves[[1]] <- result$roc_curve
  
  
  # Ridge
  ridge_model <- glmnet(over_matrix, as.numeric(over$TEAyyNEC_factor)-1,
                        alpha=0, family="binomial")
  pred_prob <- predict(ridge_model, newx=test_matrix, type="response", s=0.01)
  pred <- ifelse(pred_prob>0.5, "Yes","No")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "Ridge")
  all_metrics[[2]] <- result$metrics
  confusion_tables[[2]] <- data.frame(Model="Ridge", result$confusion_stats)
  roc_curves[[2]] <- result$roc_curve
  
  
  # Elastic Net
  enet_model <- glmnet(over_matrix, as.numeric(over$TEAyyNEC_factor)-1,
                       alpha=0.5, family="binomial")
  pred_prob <- predict(enet_model, newx=test_matrix, type="response", s=0.01)
  pred <- ifelse(pred_prob>0.5, "Yes","No")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "Elastic Net")
  all_metrics[[3]] <- result$metrics
  confusion_tables[[3]] <- data.frame(Model="Elastic Net", result$confusion_stats)
  roc_curves[[3]] <- result$roc_curve
  
  
  # Random Forest
  rf_model <- randomForest(over_matrix, as.factor(over$TEAyyNEC_factor), ntree=100)
  pred_prob <- predict(rf_model, newdata=test_matrix, type="prob")[,"Yes"]
  pred <- predict(rf_model, newdata=test_matrix)
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "Random Forest")
  all_metrics[[4]] <- result$metrics
  confusion_tables[[4]] <- data.frame(Model="Random Forest", result$confusion_stats)
  roc_curves[[4]] <- result$roc_curve
  
  
  # ANN
  ann_model <- nnet(over_matrix, as.numeric(over$TEAyyNEC_factor)-1,
                    size=5, decay=0.1, maxit=100)
  pred_prob <- predict(ann_model, newdata=test_matrix)
  pred <- ifelse(pred_prob>0.5, "Yes","No")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "ANN")
  all_metrics[[5]] <- result$metrics
  confusion_tables[[5]] <- data.frame(Model="ANN", result$confusion_stats)
  roc_curves[[5]] <- result$roc_curve
  
  
  # kNN
  pred <- knn(train=over_matrix, test=test_matrix,
              cl=over$TEAyyNEC_factor, k=5, prob=TRUE)
  pred_prob <- attr(pred, "prob")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "kNN")
  all_metrics[[6]] <- result$metrics
  confusion_tables[[6]] <- data.frame(Model="kNN", result$confusion_stats)
  roc_curves[[6]] <- result$roc_curve
  
  
  # XGBoost
  xgb_model <- xgboost(data=over_matrix,
                       label=as.numeric(over$TEAyyNEC_factor)-1,
                       objective="binary:logistic",
                       nrounds=100, verbose=0)
  pred_prob <- predict(xgb_model, newdata=test_matrix)
  pred <- ifelse(pred_prob>0.5, "Yes","No")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "XGBoost")
  all_metrics[[7]] <- result$metrics
  confusion_tables[[7]] <- data.frame(Model="XGBoost", result$confusion_stats)
  roc_curves[[7]] <- result$roc_curve
  
  
  # Decision Tree
  dt_model <- rpart(TEAyyNEC_factor ~ ., data=over)
  pred_prob <- predict(dt_model, newdata=test, type="prob")[,"Yes"]
  pred <- predict(dt_model, newdata=test, type="class")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "Decision Tree")
  all_metrics[[8]] <- result$metrics
  confusion_tables[[8]] <- data.frame(Model="Decision Tree", result$confusion_stats)
  roc_curves[[8]] <- result$roc_curve
  
  
  # Logistic Regression
  logit_model <- glm(TEAyyNEC_factor ~ ., data=over, family="binomial")
  pred_prob <- predict(logit_model, newdata=test, type="response")
  pred <- ifelse(pred_prob>0.5, "Yes","No")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "Logistic Regression")
  all_metrics[[9]] <- result$metrics
  confusion_tables[[9]] <- data.frame(Model="Logistic Regression", result$confusion_stats)
  roc_curves[[9]] <- result$roc_curve
  
  
  # Naive Bayes
  naive_bayes_model <- naiveBayes(TEAyyNEC_factor ~ ., data=over)
  pred_prob <- predict(naive_bayes_model, newdata=test, type="raw")[,"Yes"]
  pred <- predict(naive_bayes_model, newdata=test)
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "Naive Bayes")
  all_metrics[[10]] <- result$metrics
  confusion_tables[[10]] <- data.frame(Model="Naive Bayes", result$confusion_stats)
  roc_curves[[10]] <- result$roc_curve
  
  
  # CatBoost
  catboost_data <- catboost.load_pool(data=over_matrix,
                                      label=as.numeric(over$TEAyyNEC_factor)-1)
  catboost_model <- catboost.train(catboost_data,
                                   params=list(iterations=100,
                                               learning_rate=0.1,
                                               depth=6,
                                               verbose=0))
  catboost_test <- catboost.load_pool(data=test_matrix)
  pred_prob <- catboost.predict(catboost_model, catboost_test)
  pred <- ifelse(pred_prob>0.5, "Yes","No")
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "CatBoost")
  all_metrics[[11]] <- result$metrics
  confusion_tables[[11]] <- data.frame(Model="CatBoost", result$confusion_stats)
  roc_curves[[11]] <- result$roc_curve
  
  
  # H2O GBM
  h2o_train <- as.h2o(cbind(as.data.frame(over_matrix),
                            TEAyyNEC_factor=over$TEAyyNEC_factor))
  h2o_test  <- as.h2o(cbind(as.data.frame(test_matrix),
                            TEAyyNEC_factor=test$TEAyyNEC_factor))
  
  predictors <- setdiff(colnames(h2o_train), "TEAyyNEC_factor")
  response   <- "TEAyyNEC_factor"
  
  h2o_gbm_model <- h2o.gbm(
    x=predictors, y=response,
    training_frame=h2o_train,
    ntrees=100, learn_rate=0.1,
    max_depth=6, seed=1234
  )
  
  h2o_pred <- h2o.predict(h2o_gbm_model, h2o_test)
  h2o_pred_df <- as.data.frame(h2o_pred)
  pred_prob <- h2o_pred_df$Yes
  pred <- as.character(h2o_pred_df$predict)
  
  result <- evaluate_model(factor(pred, levels=c("No","Yes")),
                           pred_prob, test$TEAyyNEC_factor, "H2O GBM")
  all_metrics[[12]] <- result$metrics
  confusion_tables[[12]] <- data.frame(Model="H2O GBM", result$confusion_stats)
  roc_curves[[12]] <- result$roc_curve
  
  
  
  # ==========================================
  # 6. YEARLY RESULTS STORAGE
  # ==========================================
  
  final_metrics <- do.call(rbind, all_metrics)
  final_confusion <- do.call(rbind, confusion_tables)
  
  yearwise_metrics[[as.character(yr)]] <- final_metrics
  yearwise_confusion[[as.character(yr)]] <- final_confusion
  
  cat("\nFinal Metrics (Year:", yr, ")\n")
  print(final_metrics)
  
  cat("\nConfusion Matrices (Year:", yr, ")\n")
  print(final_confusion)
  
}



# ==========================================
# 7. DONE — RESULTS AVAILABLE AS:
# yearwise_metrics[[ "2015" ]]  ...  [[ "2020" ]]
# yearwise_confusion[[ "2015" ]]  ...  [[ "2020" ]]
# ==========================================

library(openxlsx)

# Create a new Excel workbook
wb <- createWorkbook()

for (yr in 2015:2020) {
  
  # Metrics sheet
  addWorksheet(wb, paste0("Metrics_", yr))
  writeData(wb, sheet = paste0("Metrics_", yr), 
            yearwise_metrics[[as.character(yr)]])
  
  # Confusion matrix sheet
  addWorksheet(wb, paste0("Confusion_", yr))
  writeData(wb, sheet = paste0("Confusion_", yr), 
            yearwise_confusion[[as.character(yr)]])
}

# Save file
saveWorkbook(wb, 
             file = "C:/Users/ady27/Desktop/combined2/NMEIndiaYearWise_2015_2020_Results.xlsx",
             overwrite = TRUE)

cat("\nExcel Export Completed: NMEIndia_2015_2020_Results.xlsx\n")

