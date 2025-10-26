# ==============================================================================
# 完整版：多模型脑影像数据分析流程（包含所有模型训练）
# ==============================================================================

# 🔧 关键：包加载顺序
library(readxl)
library(purrr)
library(tidyr)
library(ggplot2)
library(readr)
library(gridExtra)
library(viridis)
library(pROC)
library(RColorBrewer)
library(randomForest)
library(kernlab)
library(xgboost)
library(klaR)
library(MASS)
library(nnet)
library(gbm)
library(caret)
library(dplyr)  # 必须最后加载

# ==============================================================================
# 0. 辅助函数
# ==============================================================================

standardize_colnames <- function(df) {
  new_names <- colnames(df) %>%
    gsub("/", "_", ., fixed = TRUE) %>%
    gsub("[^A-Za-z0-9_.]", "_", .) %>%
    make.unique(sep = "_")
  colnames(df) <- new_names
  return(df)
}

standardize_feature_names <- function(feature_vec) {
  feature_vec %>%
    gsub("/", "_", ., fixed = TRUE) %>%
    gsub("[^A-Za-z0-9_.]", "_", .)
}

validate_features <- function(df, feature_list, dataset_name = "Data") {
  missing_features <- setdiff(feature_list, colnames(df))
  if (length(missing_features) > 0) {
    cat(sprintf("\n⚠️ 警告: %s 中缺失以下特征:\n", dataset_name))
    cat(paste(head(missing_features, 10), collapse = ", "), "\n")
    return(FALSE)
  }
  cat(sprintf("✓ %s: 所有 %d 个特征验证通过\n", dataset_name, length(feature_list)))
  return(TRUE)
}

calculate_f1 <- function(confusion_matrix_obj) {
  precision <- confusion_matrix_obj$byClass["Pos Pred Value"]
  recall <- confusion_matrix_obj$byClass["Sensitivity"]
  
  if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
    f1 <- NA
  } else {
    f1 <- 2 * (precision * recall) / (precision + recall)
  }
  return(f1)
}

GLOBAL_FIT_CONTROL <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# ==============================================================================
# 1. 数据加载与预处理
# ==============================================================================

clean_and_merge_data <- function(sex_file, vgm_file, vwm_file, vcsf_file,
                                 depth_file, fractal_file, gyrification_file,
                                 thickness_file, path) {
  
  file_list <- list(
    sex = read_excel(paste0(path, sex_file)),
    vgm = read_csv(paste0(path, vgm_file), show_col_types = FALSE),
    vwm = read_csv(paste0(path, vwm_file), show_col_types = FALSE),
    vcsf = read_csv(paste0(path, vcsf_file), show_col_types = FALSE),
    depth = read_csv(paste0(path, depth_file), show_col_types = FALSE),
    fractal = read_csv(paste0(path, fractal_file), show_col_types = FALSE),
    gyrification = read_csv(paste0(path, gyrification_file), show_col_types = FALSE),
    thickness = read_csv(paste0(path, thickness_file), show_col_types = FALSE)
  )
  
  processed_list <- map2(file_list, names(file_list), function(df, name) {
    if (name == "sex") {
      colnames(df)[1] <- "number"
    } else {
      colnames(df)[1] <- "number"
      df <- df %>% rename_with(~ paste(name, .x, sep = "_"), .cols = -number)
    }
    return(df)
  })
  
  merged_data <- reduce(processed_list, full_join, by = "number")
  na_counts <- sapply(merged_data, function(x) sum(is.na(x)))
  cols_to_keep <- names(na_counts[na_counts < nrow(merged_data)])
  merged_data <- merged_data[, cols_to_keep, drop = FALSE]
  merged_data <- standardize_colnames(merged_data)
  
  return(merged_data)
}

analyze_group_differences <- function(data, name_prefix) {
  cat("\n", rep("=", 60), "\n", sep = "")
  cat("开始分析 ", name_prefix, "\n", sep = "")
  
  label_values <- unique(data$Label)
  if (length(label_values) != 2) stop("Label列必须恰好有两组。")
  
  continuous_vars <- setdiff(names(data), c("Label", "sex"))
  results_list <- list()
  
  for (i in 1:length(continuous_vars)) {
    var <- continuous_vars[i]
    if (i %% 500 == 0) cat("已处理", i, "/", length(continuous_vars), "个变量...\n")
    
    tryCatch({
      group1 <- data[[var]][data$Label == label_values[1]]
      group2 <- data[[var]][data$Label == label_values[2]]
      group1 <- group1[!is.na(group1)]; group2 <- group2[!is.na(group2)]
      
      if (length(group1) < 2 || length(group2) < 2) next
      
      t_test <- t.test(group1, group2)
      n1 <- length(group1); n2 <- length(group2)
      pooled_sd <- sqrt(((n1-1)*sd(group1)^2 + (n2-1)*sd(group2)^2) / (n1 + n2 - 2))
      cohens_d <- ifelse(pooled_sd > 0, (mean(group1) - mean(group2)) / pooled_sd, NA)
      
      results_list[[var]] <- data.frame(
        Variable = var, N_Group1 = n1, N_Group2 = n2,
        Group1_Mean = mean(group1), Group2_Mean = mean(group2),
        Mean_Diff = mean(group1) - mean(group2), 
        T_pvalue = t_test$p.value, Cohens_d = cohens_d
      )
    }, error = function(e) {})
  }
  
  results_df <- bind_rows(results_list) %>%
    mutate(T_pvalue_adj = p.adjust(T_pvalue, method = "fdr")) %>%
    arrange(T_pvalue)
  
  write_csv(results_df, paste0(base_path, name_prefix, "_group_comparison_results.csv"))
  return(list(results = results_df))
}

# ==============================================================================
# 2. 主执行流程
# ==============================================================================

base_path <- "C:/Users/19134/OneDrive/Desktop/cop1/" 
split_ratio <- 0.7

cat("\n#### A. 清洗合并数据 ####\n")
group1_data_merged <- clean_and_merge_data(
  sex_file = "COP_PD_CI.xlsx",
  vgm_file = "COP_PD_CI_ROI_BN_Atlas_274_combined_Vgm.csv",
  vwm_file = "COP_PD_CI_ROI_BN_Atlas_274_combined_Vwm.csv",
  vcsf_file = "COP_PD_CI_ROI_BN_Atlas_274_combined_Vcsf.csv",
  depth_file = "COP_PD_CI_surf_ROI_aparc_BN_Atlas_depth.csv",
  fractal_file = "COP_PD_CI_surf_ROI_aparc_BN_Atlas_fractaldimension.csv",
  gyrification_file = "COP_PD_CI_surf_ROI_aparc_BN_Atlas_gyrification.csv",
  thickness_file = "COP_PD_CI_surf_ROI_aparc_BN_Atlas_thickness.csv",
  path = base_path
) %>% mutate(Label = "Group1")

group2_data_merged <- clean_and_merge_data(
  sex_file = "COP_CI.xlsx",
  vgm_file = "COP_CI_ROI_BN_Atlas_274_combined_Vgm.csv",
  vwm_file = "COP_CI_ROI_BN_Atlas_274_combined_Vwm.csv",
  vcsf_file = "COP_CI_ROI_BN_Atlas_274_combined_Vcsf.csv",
  depth_file = "COP_CI_surf_ROI_aparc_BN_Atlas_depth.csv",
  fractal_file = "COP_CI_surf_ROI_aparc_BN_Atlas_fractaldimension.csv",
  gyrification_file = "COP_CI_surf_ROI_aparc_BN_Atlas_gyrification.csv",
  thickness_file = "COP_CI_surf_ROI_aparc_BN_Atlas_thickness.csv",
  path = base_path
) %>% mutate(Label = "Group2")

combined_data_all <- bind_rows(group1_data_merged, group2_data_merged) %>%
  dplyr::select(-number)

set.seed(123)
trainIndex <- createDataPartition(combined_data_all$Label, p = split_ratio, list = FALSE)
trainData <- combined_data_all[trainIndex, ]
testData <- combined_data_all[-trainIndex, ]

write.csv(trainData, paste0(base_path, "train_data.csv"), row.names = FALSE)
write.csv(testData, paste0(base_path, "test_data.csv"), row.names = FALSE)

# ==============================================================================
# 3. 准备特征集
# ==============================================================================

F_final_34_vars_original <- c(
  "gyrification_rA35/36r_R", "vgm_Vermis_VIb", "vwm_lPrG_6_4", "vwm_rPoG_4_2", 
  "fractal_rlsOccG_R", "vcsf_rCG_7_2", "fractal_lA37dl_L", "vcsf_rINS_6_6", 
  "vgm_lOrG_6_1", "vcsf_lIPL_6_5", "vcsf_lpSTS_2_2", "vcsf_lOrG_6_6", 
  "vcsf_rLOcC_4_3", "vwm_rPrG_6_4", "vgm_rTha_8_8", "fractal_rIFJ_R", 
  "thickness_lA2_L", "gyrification_lA2_L", "thickness_rA11l_R", "gyrification_rA11l_R", 
  "vcsf_rPhG_6_3", "vwm_rLOcC_4_4", "depth_lA20iv_L", "vgm_rSFG_7_1", 
  "fractal_rA35/36c_R", "fractal_rA23v_R", "gyrification_rA5m_R", "thickness_lA40rd_L", 
  "edu", "thickness_rA4tl_R", "vcsf_lCG_7_7", "thickness_lcLinG_L", 
  "thickness_lA41/42_L", "depth_rA6m_R"
)

F_final_vars <- c(standardize_feature_names(F_final_34_vars_original), "sex")

trainData_full <- read.csv(paste0(base_path, "train_data.csv"), check.names = FALSE)
testData_full <- read.csv(paste0(base_path, "test_data.csv"), check.names = FALSE)

trainData_full <- standardize_colnames(trainData_full)
testData_full <- standardize_colnames(testData_full)

trainData_final <- trainData_full %>% dplyr::select(Label, all_of(F_final_vars))
testData_final <- testData_full %>% dplyr::select(Label, all_of(F_final_vars))

trainData_final$sex <- factor(trainData_final$sex)
testData_final$sex <- factor(testData_final$sex)
trainData_final$Label <- factor(trainData_final$Label)
testData_final$Label <- factor(testData_final$Label)

# ==============================================================================
# 4. 训练所有9个模型
# ==============================================================================

cat("\n### 开始训练 9 个模型 ###\n")

# 1. Ridge
cat("\n[1/9] 训练 Ridge...\n")
set.seed(42)
ridge_model <- train(Label ~ ., data = trainData_final, method = "glmnet", 
                     metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                     preProcess = c("center", "scale"),
                     tuneGrid = expand.grid(alpha = 0, lambda = exp(seq(-5, 2, length.out = 30))))

# 2. Random Forest
cat("[2/9] 训练 Random Forest...\n")
set.seed(42)
rf_model <- train(Label ~ ., data = trainData_final, method = "rf", 
                  metric = "ROC", trControl = GLOBAL_FIT_CONTROL, tuneLength = 3)

# 3. SVM
cat("[3/9] 训练 SVM...\n")
set.seed(42)
svm_model <- train(Label ~ ., data = trainData_final, method = "svmRadial", 
                   metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                   preProcess = c("center", "scale"), tuneLength = 3)

# 4. KNN
cat("[4/9] 训练 KNN...\n")
set.seed(42)
knn_model <- train(Label ~ ., data = trainData_final, method = "knn", 
                   metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                   preProcess = c("center", "scale"), tuneLength = 3)

# 5. XGBoost
cat("[5/9] 训练 XGBoost...\n")
set.seed(42)
xgb_model <- train(Label ~ ., data = trainData_final, method = "xgbTree", 
                   metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                   tuneLength = 3, verbose = FALSE)

# 6. Naive Bayes
cat("[6/9] 训练 Naive Bayes...\n")
set.seed(42)
nb_model <- train(Label ~ ., data = trainData_final, method = "naive_bayes", 
                  metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                  preProcess = c("center", "scale"), tuneLength = 3)

# 7. LDA
cat("[7/9] 训练 LDA...\n")
set.seed(42)
lda_model <- train(Label ~ ., data = trainData_final, method = "lda", 
                   metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                   preProcess = c("center", "scale"))

# 8. NNET
cat("[8/9] 训练 NNET...\n")
set.seed(42)
nnet_model <- train(Label ~ ., data = trainData_final, method = "nnet", 
                    metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                    preProcess = c("center", "scale"),
                    tuneGrid = expand.grid(size = c(3, 5), decay = c(0.01, 0.1)),
                    maxit = 200, trace = FALSE)

# 9. GBM
cat("[9/9] 训练 GBM...\n")
set.seed(42)
gbm_model <- train(Label ~ ., data = trainData_final, method = "gbm", 
                   metric = "ROC", trControl = GLOBAL_FIT_CONTROL,
                   tuneLength = 3, verbose = FALSE)

cat("\n✓ 所有模型训练完成！\n")

# ==============================================================================
# 5. 评估所有模型
# ==============================================================================

cat("\n### 评估模型性能 ###\n")

models_list <- list(
  Ridge = ridge_model, RF = rf_model, SVM = svm_model, KNN = knn_model,
  XGBoost = xgb_model, NB = nb_model, LDA = lda_model, 
  NNET = nnet_model, GBM = gbm_model
)

results_df <- data.frame()

for (model_name in names(models_list)) {
  model <- models_list[[model_name]]
  pred <- predict(model, testData_final)
  prob <- predict(model, testData_final, type = "prob")
  cm <- confusionMatrix(pred, testData_final$Label)
  
  positive_class <- levels(testData_final$Label)[2]
  roc_obj <- roc(testData_final$Label, prob[[positive_class]], 
                 levels = levels(testData_final$Label), direction = "<", quiet = TRUE)
  
  results_df <- rbind(results_df, data.frame(
    Model = model_name,
    AUC = auc(roc_obj),
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Precision = cm$byClass["Pos Pred Value"],
    F1_Score = calculate_f1(cm)
  ))
}

model_comparison <- results_df %>% arrange(desc(AUC))
write.csv(model_comparison, paste0(base_path, "all_models_performance_with_f1.csv"), row.names = FALSE)

cat("\n完整性能对比表:\n")
print(model_comparison, digits = 4)

# ==============================================================================
# 6. 可视化
# ==============================================================================

cat("\n### 生成可视化图表 ###\n")

# F1值对比图
p_f1 <- ggplot(model_comparison, aes(x = reorder(Model, F1_Score), y = F1_Score, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.4f", F1_Score)), hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "所有模型 F1 值对比", x = "模型", y = "F1 Score") +
  theme_minimal() +
  ylim(0, max(model_comparison$F1_Score, na.rm = TRUE) * 1.15)

ggsave(paste0(base_path, "f1_score_comparison.png"), p_f1, width = 10, height = 7, dpi = 300)

# 热力图
heatmap_data <- model_comparison %>%
  dplyr::select(Model, AUC, Accuracy, Sensitivity, Specificity, Precision, F1_Score) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

p_heatmap <- ggplot(heatmap_data, aes(x = Metric, y = Model, fill = Value)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.3f", Value)), color = "black", size = 3) +
  scale_fill_gradient2(low = "#f7fbff", mid = "#6baed6", high = "#08519c",
                       midpoint = 0.8, limits = c(0.5, 1), name = "指标值") +
  labs(title = "所有模型性能指标热力图", x = "性能指标", y = "模型") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(paste0(base_path, "performance_heatmap_with_f1.png"), p_heatmap, width = 10, height = 7, dpi = 300)

cat("\n✅ 分析完成！所有结果已保存至:", base_path, "\n")
