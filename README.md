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
library(patchwork) # Added for plot combination
library(kernelshap) # Added for SHAP
library(shapviz)    # Added for SHAP plotting
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
# 5. 评估所有模型 (Train & Test) - [已修改]
# ==============================================================================

cat("\n### 5. 评估模型性能 (Train & Test) ###\n")

models_list <- list(
  Ridge = ridge_model, RF = rf_model, SVM = svm_model, KNN = knn_model,
  XGBoost = xgb_model, NB = nb_model, LDA = lda_model,
  NNET = nnet_model, GBM = gbm_model
)

# 准备一个函数来提取所有指标
calculate_all_metrics <- function(model, data, data_name, model_name) {
  pred <- predict(model, data)
  prob <- predict(model, data, type = "prob")
  cm <- confusionMatrix(pred, data$Label)
  
  positive_class <- levels(data$Label)[2]
  roc_obj <- roc(data$Label, prob[[positive_class]],
                 levels = levels(data$Label), direction = "<", quiet = TRUE)
  
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  
  # 确保 F1 值为 NA 时使用自定义函数
  f1_val <- cm$byClass["F1"]
  if (is.na(f1_val)) {
    f1_val <- calculate_f1(cm)
  }
  
  metrics <- data.frame(
    Model = model_name,
    Dataset = data_name,
    AUC = as.numeric(auc(roc_obj)),
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = sens,
    Specificity = spec,
    PPV = cm$byClass["Pos Pred Value"],
    NPV = cm$byClass["Neg Pred Value"],
    F1 = f1_val,
    Recall = sens, # Recall is the same as Sensitivity
    Youdens_Index = sens + spec - 1
  )
  return(metrics)
}

# 循环计算 Train 和 Test 的指标
all_results_list <- list()
# (在 Section 8 中定义, 用于匹配模型名称)
display_names_ci <- c("Ridge", "RandomForest", "SVM_Kernel", "KNN",
                      "XGBoost", "NaiveBayes", "LDA", "NeuralNet", "GradientBoosting")

for (i in 1:length(models_list)) {
  model_name <- names(models_list)[i]
  model_display_name <- display_names_ci[i] # 使用一致的显示名称
  
  cat(sprintf("... 评估 %s ...\n", model_display_name))
  model <- models_list[[model_name]]
  
  # 计算训练集指标
  all_results_list[[paste(model_name, "train")]] <-
    calculate_all_metrics(model, trainData_final, "Training", model_display_name)
  
  # 计算测试集指标
  all_results_list[[paste(model_name, "test")]] <-
    calculate_all_metrics(model, testData_final, "Validation", model_display_name)
}

all_results_df <- bind_rows(all_results_list)

# --- 创建 Section 9 需要的数据框 ---
# 1. train_performance (用于新的 Section 9)
train_performance <- all_results_df %>%
  filter(Dataset == "Training") %>%
  dplyr::select(Model, Accuracy, F1, NPV, PPV, Recall, Sensitivity, Specificity, Youdens_Index)

# 2. validation_performance (用于新的 Section 9)
validation_performance <- all_results_df %>%
  filter(Dataset == "Validation") %>%
  dplyr::select(Model, Accuracy, F1, NPV, PPV, Recall, Sensitivity, Specificity, Youdens_Index)

# --- 兼容旧代码 (Section 6, 7) ---
# 3. model_comparison (用于 Section 6, 7)
#    (我们从 all_results_df 重新创建它，但使用原始的 'RF', 'SVM' 等名称)
model_comparison_orig_names <- bind_rows(lapply(names(models_list), function(name) {
  calculate_all_metrics(models_list[[name]], testData_final, "Validation", name)
}))

model_comparison <- model_comparison_orig_names %>%
  dplyr::select(Model, AUC, Accuracy, Sensitivity, Specificity, PPV, F1) %>%
  rename(Precision = PPV, F1_Score = F1) %>% # 恢复 Section 6 热力图期望的名称
  arrange(desc(AUC))


cat("\n完整性能对比表 (Validation Set):\n")
print(model_comparison, digits = 4)
write.csv(model_comparison, paste0(base_path, "all_models_performance_with_f1.csv"), row.names = FALSE)

# ==============================================================================
# 6. 可视化 (F1 与 热力图)
# ==============================================================================

cat("\n### 6. 生成可视化图表 (F1 与 热力图) ###\n")

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

cat("\n✓ F1 Score plot saved.\n")
print(p_f1)

cat("\n✓ Performance heatmap saved.\n")
print(p_heatmap)

# ==============================================================================
# 7. ROC 曲线 (分面与组合)
# ==============================================================================

cat("\n### 7. 生成 ROC 曲线 (分面与组合) ###\n")

# 确保 models_list 和 testData_final 存在
if (!exists("models_list") || !exists("testData_final")) {
  stop("错误: 'models_list' 或 'testData_final' 不存在。")
}

positive_class <- levels(testData_final$Label)[2]
cat("计算 ROC 对象...\n")

roc_list <- list()
auc_values_map <- list()

# 为 models_list 中的每个模型计算 ROC 和 AUC
# 使用 models_list 中的名字 (e.g., "RF", "SVM") 作为基础
for (model_name in names(models_list)) {
  model_obj <- models_list[[model_name]]
  
  tryCatch({
    probabilities <- predict(model_obj, newdata = testData_final, type = "prob")[[positive_class]]
    
    roc_obj <- roc(response = testData_final$Label,
                   predictor = probabilities,
                   levels = levels(testData_final$Label),
                   direction = "<",
                   quiet = TRUE) # 增加 quiet = TRUE
    
    roc_list[[model_name]] <- roc_obj
    auc_values_map[[model_name]] <- auc(roc_obj)
    
  }, error = function(e) {
    cat(sprintf("警告: 无法为模型 '%s' 计算 ROC: %s\n", model_name, e$message))
  })
}

# --- 7a. 准备用于分面图的数据框 ---
display_names_roc <- c(
  "Ridge" = "Ridge", "RF" = "Random Forest", "SVM" = "SVM (RBF)", "KNN" = "KNN",
  "XGBoost" = "XGBoost", "NB" = "Naive Bayes", "LDA" = "LDA", "NNET" = "NNET", "GBM" = "GBM"
)

roc_data_facet <- map_dfr(names(roc_list), function(model_name) {
  roc_obj <- roc_list[[model_name]]
  coords_df <- coords(roc_obj, "all", ret = c("threshold", "specificity", "sensitivity"), transpose = FALSE)
  
  # 使用映射的显示名称
  display_name <- ifelse(model_name %in% names(display_names_roc), display_names_roc[model_name], model_name)
  
  coords_df$Model <- sprintf("%s (AUC = %.4f)", display_name, auc_values_map[[model_name]])
  return(coords_df)
})

roc_data_facet <- roc_data_facet %>%
  mutate(FalsePositiveRate = 1 - specificity) %>%
  mutate(Model = factor(Model, levels = map_chr(
    names(sort(unlist(auc_values_map), decreasing=TRUE)),
    function(name) {
      display_name <- ifelse(name %in% names(display_names_roc), display_names_roc[name], name)
      sprintf("%s (AUC = %.4f)", display_name, auc_values_map[[name]])
    }
  )))

cat("生成分面 ROC 曲线图...\n")
p_roc_facet <- ggplot(roc_data_facet, aes(x = FalsePositiveRate, y = sensitivity)) +
  geom_line(aes(color = Model), size = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
  facet_wrap(~ Model, nrow = 3) +
  theme_bw() +
  scale_color_discrete(guide = "none") +
  labs(
    title = "九模型 ROC 曲线对比 (分面图, TestData)",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    strip.text = element_text(size = 9)
  )

ggsave(paste0(base_path, "combined_roc_curves_9models_facet.png"), plot = p_roc_facet, width = 12, height = 10, dpi = 300)
cat("✓ 分面 ROC 曲线图已保存。\n")
print(p_roc_facet)


# --- 7b. 准备用于彩色对比图的数据框 ---
# 按 AUC 降序排序
ordered_script_names <- names(sort(unlist(auc_values_map), decreasing = TRUE))
roc_list_ordered <- roc_list[ordered_script_names]

roc_data_colored <- map_dfr(names(roc_list_ordered), function(model_name) {
  roc_obj <- roc_list_ordered[[model_name]]
  coords_df <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
  
  display_name <- ifelse(model_name %in% names(display_names_roc), display_names_roc[model_name], model_name)
  coords_df$ModelLabel <- sprintf("%s (AUC = %.4f)", display_name, auc_values_map[[model_name]])
  
  return(coords_df)
})

roc_data_colored <- roc_data_colored %>%
  mutate(FalsePositiveRate = 1 - specificity) %>%
  mutate(ModelLabel = factor(ModelLabel, levels = unique(ModelLabel))) # 保持 AUC 排序

cat("生成彩色 ROC 曲线图 (所有模型不同颜色)...\n")
color_palette_roc <- brewer.pal(n = max(3, length(roc_list)), name = "Paired") # 至少要3种颜色

p_roc_colored <- ggplot(roc_data_colored,
                        aes(x = FalsePositiveRate, y = sensitivity,
                            color = ModelLabel,
                            group = ModelLabel)) +
  geom_line(size = 0.8, alpha = 0.9) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = color_palette_roc, name = "模型 (AUC)") +
  theme_bw() +
  ggtitle("九模型 ROC 曲线对比 (TestData)") +
  xlab("1 - Specificity (False Positive Rate)") +
  ylab("Sensitivity (True Positive Rate)") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title = element_text(size = 12),
        legend.title = element_text(size = 11),
        legend.text = element_text(size = 9),
        legend.position = "right",
        legend.key.size = unit(0.8, "lines"))

ggsave(paste0(base_path, "combined_roc_curves_9models_colored.png"), plot = p_roc_colored, width = 11, height = 8, dpi = 300)
cat("✓ 彩色 ROC 曲线图已保存。\n")
print(p_roc_colored)


# ==============================================================================
# 8. AUC 95% CI 对比图 (Train vs Test)
# ==============================================================================
# 此部分改编自用户提供的 #39.
# 它使用 `models_list` 中的模型, 并匹配 #39 中定义的显示名称顺序

cat("\n### 8. 绘制 Train/Validation AUC with 95% CI ###\n")

# 确保 trainData_final 也存在
if (!exists("trainData_final")) {
  stop("错误: 'trainData_final' 不存在 (CI 图需要)。")
}

# 这些是用于绘图的显示名称, 顺序必须与 'models_list' 一致
display_names_ci <- c("Ridge", "RandomForest", "SVM_Kernel", "KNN",
                      "XGBoost", "NaiveBayes", "LDA", "NeuralNet", "GradientBoosting")

# 检查名称列表长度是否匹配
if (length(display_names_ci) != length(models_list)) {
  stop("错误: CI图的 'display_names_ci' 列表长度与 'models_list' 不匹配。")
}

script_model_names <- names(models_list) # e.g., "Ridge", "RF", "SVM", ...

cat("Calculating ROC, AUC, and 95% CI (using bootstrap)...\n")
auc_ci_results <- list()
n_bootstrap <- 2000 # 减少 bootstrap 次数以加快速度, 2000 仍然很稳健

for (i in 1:length(script_model_names)) {
  
  script_name <- script_model_names[i]
  model_display_name <- display_names_ci[i] # 使用 #39 中的目标显示名称
  
  model_obj <- models_list[[script_name]]
  
  # Validation (Test) Set
  prob_val <- predict(model_obj, newdata = testData_final, type = "prob")[[positive_class]]
  roc_val <- roc(response = testData_final$Label, predictor = prob_val,
                 levels = levels(testData_final$Label), direction = "<", quiet = TRUE)
  ci_val <- ci.auc(roc_val, method = "bootstrap", boot.n = n_bootstrap, progress = "none")
  
  # Training Set
  prob_train <- predict(model_obj, newdata = trainData_final, type = "prob")[[positive_class]]
  roc_train <- roc(response = trainData_final$Label, predictor = prob_train,
                   levels = levels(trainData_final$Label), direction = "<", quiet = TRUE)
  ci_train <- ci.auc(roc_train, method = "bootstrap", boot.n = n_bootstrap, progress = "none")
  
  # 存储结果 (使用 display_name 作为 key)
  auc_ci_results[[model_display_name]] <- list(
    Train_AUC = as.numeric(ci_train[2]), Train_LowerCI = as.numeric(ci_train[1]), Train_UpperCI = as.numeric(ci_train[3]),
    Validation_AUC = as.numeric(ci_val[2]), Validation_LowerCI = as.numeric(ci_val[1]), Validation_UpperCI = as.numeric(ci_val[3])
  )
  
  cat(sprintf("   - %s: Train AUC=%.3f (%.3f-%.3f), Val AUC=%.3f (%.3f-%.3f)\n",
              model_display_name,
              auc_ci_results[[model_display_name]]$Train_AUC, auc_ci_results[[model_display_name]]$Train_LowerCI, auc_ci_results[[model_display_name]]$Train_UpperCI,
              auc_ci_results[[model_display_name]]$Validation_AUC, auc_ci_results[[model_display_name]]$Validation_LowerCI, auc_ci_results[[model_display_name]]$Validation_UpperCI))
}

# --- 准备用于 CI 图的数据框 ---
plot_data_ci <- map_dfr(names(auc_ci_results), ~{
  data.frame(
    Model = .x,
    Train_AUC = auc_ci_results[[.x]]$Train_AUC,
    Train_LowerCI = auc_ci_results[[.x]]$Train_LowerCI,
    Train_UpperCI = auc_ci_results[[.x]]$Train_UpperCI,
    Validation_AUC = auc_ci_results[[.x]]$Validation_AUC,
    Validation_LowerCI = auc_ci_results[[.x]]$Validation_LowerCI,
    Validation_UpperCI = auc_ci_results[[.x]]$Validation_UpperCI
  )
}) %>%
  mutate(
    Train_Label = sprintf("%.3f (%.3f-%.3f)", Train_AUC, Train_LowerCI, Train_UpperCI),
    Validation_Label = sprintf("%.3f (%.3f-%.3f)", Validation_AUC, Validation_LowerCI, Validation_UpperCI)
  ) %>%
  arrange(Validation_AUC) %>%
  mutate(Model = factor(Model, levels = Model))

# 为每个模型分配不同颜色
color_palette_ci <- c("#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00",
                      "#E85D04", "#9C964A", "#FDA333", "#C05C55")
plot_data_ci <- plot_data_ci %>%
  mutate(Model_Color = color_palette_ci[1:n()])

# --- 创建 CI 图 ---
cat("\nGenerating AUC (95% CI) plots...\n")

# 【修正】使用 ggplot2::margin 避免函数冲突
example_theme <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold",
                              margin = ggplot2::margin(b = 10)),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black", hjust = 1,
                               margin = ggplot2::margin(r = 5)),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    plot.margin = ggplot2::margin(5, 10, 5, 10),
    legend.position = "none"
  )

# 训练集图
plot_train_exact <- ggplot(plot_data_ci, aes(y = Model)) +
  geom_segment(aes(x = Train_LowerCI, xend = Train_UpperCI,
                   y = Model, yend = Model, color = Model),
               linewidth = 1.5, show.legend = FALSE) +
  geom_point(aes(x = Train_AUC, color = Model, fill = Model),
             size = 4, shape = 23, stroke = 0.5, show.legend = FALSE) +
  geom_text(aes(x = (Train_LowerCI + Train_UpperCI) / 2, label = Train_Label),
            vjust = -0.8, size = 3, color = "black", fontface = "plain") +
  scale_color_manual(values = setNames(plot_data_ci$Model_Color, plot_data_ci$Model)) +
  scale_fill_manual(values = setNames(plot_data_ci$Model_Color, plot_data_ci$Model)) +
  scale_x_continuous(
    limits = c(0.5, 1.05),
    breaks = seq(0.5, 1.0, 0.05),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(title = "Train") +
  example_theme +
  theme(
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold",
                              margin = ggplot2::margin(t = 3, b = 8, l = 0, r = 0)), # <--- 修正
    plot.background = element_rect(fill = "white", color = NA)
  )

# 验证集 (Test) 图
plot_validation_exact <- ggplot(plot_data_ci, aes(y = Model)) +
  geom_segment(aes(x = Validation_LowerCI, xend = Validation_UpperCI,
                   y = Model, yend = Model, color = Model),
               linewidth = 1.5) +
  geom_point(aes(x = Validation_AUC, color = Model, fill = Model),
             size = 4, shape = 23, stroke = 0.5) +
  geom_text(aes(x = (Validation_LowerCI + Validation_UpperCI) / 2, label = Validation_Label),
            vjust = -0.8, size = 3, color = "black", fontface = "plain") +
  scale_color_manual(
    values = setNames(plot_data_ci$Model_Color, plot_data_ci$Model),
    name = "Model"
  ) +
  scale_fill_manual(
    values = setNames(plot_data_ci$Model_Color, plot_data_ci$Model),
    name = "Model"
  ) +
  scale_x_continuous(
    limits = c(0.5, 1.05),
    breaks = seq(0.5, 1.0, 0.05),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(title = "Test") +
  example_theme +
  theme(
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold",
                              margin = ggplot2::margin(t = 5, b = 10)), # <--- 修正
    plot.title.position = "plot",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 11, face = "bold", margin = ggplot2::margin(b = 8)), # <--- 修正
    legend.text = element_text(size = 10),
    legend.key.size = unit(0.8, "cm"),
    legend.spacing.y = unit(0.3, "cm"),
    legend.background = element_rect(fill = "white", color = "gray70", linewidth = 0.3),
    legend.margin = ggplot2::margin(5, 5, 5, 5) # <--- 修正
  ) +
  guides(color = guide_legend(
    override.aes = list(
      shape = 23,
      size = 4,
      stroke = 0.5,
      linewidth = 1.5
    )
  ))

# --- 组合并保存 CI 图 ---
cat("\nCombining CI plots side-by-side...\n")

combined_plot_exact <- (plot_train_exact + plot_validation_exact) +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(
    caption = "AUC Score (95% CI)",
    theme = theme(
      plot.caption = element_text(hjust = 0.5, size = 12, face = "bold",
                                  margin = ggplot2::margin(t = 8), color = "black") # <--- 修正
    )
  ) &
  theme(
    plot.title = element_text(
      hjust = 0.5, size = 13, face = "bold",
      margin = ggplot2::margin(t = 5, b = 8, l = 5, r = 5) # <--- 修正
    ),
    plot.background = element_rect(fill = "gray88", color = "black", linewidth = 0.5),
    plot.margin = ggplot2::margin(8, 10, 5, 10) # <--- 修正
  )

output_file_ci <- paste0(base_path, "auc_comparison_train_test_ci.png")
ggsave(output_file_ci, plot = combined_plot_exact,
       width = 14, height = 7, dpi = 300, bg = "white")
cat(sprintf("\n✓ AUC (95%% CI) plot saved to: %s\n", output_file_ci))

print(combined_plot_exact)

# ==============================================================================
# 9. Plot Performance Metrics Comparison - Matching AUC Plot Style (新添加)
# ==============================================================================

cat("\n### 9. 绘制性能指标对比图 (匹配AUC风格) ###\n")

# --- 1. 检查新生成的数据是否存在 ---
if (!exists("train_performance") || !exists("validation_performance")) {
  stop("错误: 'train_performance' 或 'validation_performance' 数据框未找到。")
}
cat("使用已生成的 performance metrics...\n")

# --- 2. 转换为长格式 ---
train_long <- train_performance %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value") %>%
  mutate(Dataset = "Training")

validation_long <- validation_performance %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value") %>%
  mutate(Dataset = "Validation")

performance_data_actual <- bind_rows(train_long, validation_long) %>%
  filter(!is.na(Value)) %>%
  mutate(
    Metric = factor(Metric, levels = c("Accuracy", "F1", "NPV", "PPV",
                                       "Recall", "Sensitivity", "Specificity",
                                       "Youdens_Index")),
    Metric_Label = case_when(
      Metric == "Youdens_Index" ~ "Youden's Index",
      Metric == "PPV" ~ "Precision (PPV)",
      Metric == "NPV" ~ "NPV",
      Metric == "Recall" ~ "Recall (Sens)",
      TRUE ~ as.character(Metric)
    ),
    Dataset = factor(Dataset, levels = c("Training", "Validation"))
  ) %>%
  # 按 Validation Accuracy 排序
  mutate(Model = factor(Model, levels = validation_performance$Model[
    order(validation_performance$Accuracy, decreasing = TRUE)]))

# --- 3. 定义颜色 ---
# (使用 Section 8 中定义的 CI 调色板)
model_colors <- color_palette_ci
names(model_colors) <- levels(performance_data_actual$Model)

# --- 4. 定义主题 (使用 ggplot2::margin 修正) ---
auc_matching_theme <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold",
                              margin = ggplot2::margin(t = 5, b = 8, l = 5, r = 5)),
    plot.background = element_rect(fill = "gray88", color = "black", linewidth = 0.5),
    panel.background = element_rect(fill = "white", color = NA),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    plot.margin = ggplot2::margin(8, 10, 5, 10),
    legend.position = "none"
  )

# --- 5. 计算Y轴范围 ---
min_y_val <- floor(min(performance_data_actual$Value, na.rm = TRUE) * 20) / 20
max_y_val <- 1.05

# --- 6. 创建训练集图 ---
cat("\nGenerating Training plot...\n")
plot_train_matching <- performance_data_actual %>%
  filter(Dataset == "Training") %>%
  ggplot(aes(x = Metric_Label, y = Value, group = Model, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3, shape = 19) +
  scale_color_manual(values = model_colors) +
  scale_y_continuous(
    limits = c(min_y_val, max_y_val),
    breaks = seq(min_y_val, 1.0, 0.05),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(title = "Train") +
  auc_matching_theme

# --- 7. 创建验证集图 (带图例) ---
cat("Generating Validation plot...\n")
plot_validation_matching <- performance_data_actual %>%
  filter(Dataset == "Validation") %>%
  ggplot(aes(x = Metric_Label, y = Value, group = Model, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3, shape = 19) +
  scale_color_manual(values = model_colors, name = "Model") +
  scale_y_continuous(
    limits = c(min_y_val, max_y_val),
    breaks = seq(min_y_val, 1.0, 0.05),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(title = "Test") +
  auc_matching_theme +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 11, face = "bold", margin = ggplot2::margin(b = 8)),
    legend.text = element_text(size = 10),
    legend.key.size = unit(0.8, "cm"),
    legend.spacing.y = unit(0.3, "cm"),
    legend.background = element_rect(fill = "white", color = "gray70", linewidth = 0.3),
    legend.margin = ggplot2::margin(5, 5, 5, 5)
  )

# --- 8. 组合图 ---
cat("\nCombining plots side-by-side...\n")

combined_plot_matching <- (plot_train_matching + plot_validation_matching) +
  plot_layout(widths = c(1, 1)) + # 1:1比例,大小一致
  plot_annotation(
    caption = "Performance Metrics Comparison",
    theme = theme(
      plot.caption = element_text(hjust = 0.5, size = 12, face = "bold",
                                  margin = ggplot2::margin(t = 8), color = "black")
    )
  ) &
  theme(
    plot.title = element_text(
      hjust = 0.5, size = 13, face = "bold",
      margin = ggplot2::margin(t = 5, b = 8, l = 5, r = 5)
    ),
    plot.background = element_rect(fill = "gray88", color = "black", linewidth = 0.5),
    plot.margin = ggplot2::margin(8, 10, 5, 10)
  )

# --- 9. 保存组合图 ---
output_file <- paste0(base_path, "performance_metrics_auc_style.png")
ggsave(output_file, plot = combined_plot_matching,
       width = 14, height = 7, dpi = 300, bg = "white")
cat(sprintf("\nPerformance metrics plot (AUC style) saved to: %s\n", output_file))

print(combined_plot_matching)

cat("\nPerformance metrics plot generated successfully.\n")

# ==============================================================================
# 10. SHAP 分析 (最佳模型)
# ==============================================================================

cat("\n### 10. SHAP 分析 (最佳模型) ###\n")

# --- 1. 识别最佳模型 ---
# (基于 model_comparison_orig_names，因为它有原始模型名称 "RF", "SVM" 等)
if (!exists("model_comparison_orig_names")) {
  stop("错误: 'model_comparison_orig_names' 未在 Section 5 中定义。")
}
best_model_name <- model_comparison_orig_names %>%
  arrange(desc(AUC)) %>%
  head(1) %>%
  pull(Model)

best_model_obj <- models_list[[best_model_name]]

cat(sprintf("识别出的最佳模型 (基于AUC): %s\n", best_model_name))
cat("准备 SHAP 分析数据...\n")

# --- 2. 准备数据 ---
# 确保 Label 列被移除
explainer_data <- trainData_final %>% dplyr::select(-Label)
eval_data <- testData_final %>% dplyr::select(-Label)

# SHAP 需要数值型数据 (将 'sex' 因子转换为 0/1)
# 注意: caret 的 preProcess = c("center", "scale") 不会转换因子
explainer_data <- explainer_data %>%
  mutate(sex = ifelse(sex == levels(trainData_final$sex)[1], 0, 1))
eval_data <- eval_data %>%
  mutate(sex = ifelse(sex == levels(trainData_final$sex)[1], 0, 1))

# --- 3. 定义预测函数 (wrapper) ---
# SHAP 需要一个只返回 positive_class 概率的函数
positive_class_shap <- levels(trainData_final$Label)[2]

# [修改] 调整 pred_wrapper 以直接使用底层模型 (例如 ksvm)
pred_wrapper <- function(model, newdata) {
  # model 现在是底层模型, 例如 ksvm
  
  # [最终修正] 不再将 sex 转回因子，直接使用数值型
  # if ("sex" %in% colnames(newdata)) {
  #    newdata$sex <- factor(
  #      newdata$sex,
  #      levels = c(0, 1),
  #      labels = levels(trainData_final$sex) # 使用原始训练数据的因子水平
  #    )
  # }
  
  
  # 调用特定于模型的 predict 方法
  # 对于 kernlab::ksvm (来自 svmRadial)
  if (inherits(model, "ksvm")) {
    # kernlab 返回一个矩阵，列名是类别标签
    probs_matrix <- tryCatch(
      kernlab::predict(model, newdata = newdata, type = "probabilities"),
      error = function(e) {
        cat("kernlab::predict 错误: ", e$message, "\n")
        cat("检查 newdata 的列名和类型是否与训练时完全一致。\n")
        # 打印 newdata 结构以供调试
        cat("--- newdata structure in ksvm error ---\n")
        print(str(newdata))
        cat("--- End newdata structure ---\n")
        NULL # 返回 NULL 表示失败
      }
    )
    if (is.null(probs_matrix) || !(positive_class_shap %in% colnames(probs_matrix))) {
      cat("错误: 无法从 kernlab::predict 获取 positive_class_shap 的概率。\n")
      cat("检查 positive_class_shap 是否正确: ", positive_class_shap, "\n")
      if (!is.null(probs_matrix)) {
        cat("检查 probs_matrix 列名: ", paste(colnames(probs_matrix), collapse=", "), "\n")
      } else {
        cat("probs_matrix 为 NULL。\n")
      }
      # 返回一个错误值或停止，这里返回 NA 向量
      return(rep(NA_real_, nrow(newdata)))
    }
    # 返回 positive class 的概率向量
    return(probs_matrix[, positive_class_shap])
    
  } else if (inherits(model, "randomForest")) {
    # randomForest 的 predict type="prob"
    # [修正] RF 也需要因子 sex
    if ("sex" %in% colnames(newdata)) {
      newdata$sex <- factor(
        newdata$sex,
        levels = c(0, 1),
        labels = levels(trainData_final$sex)
      )
    }
    probs_matrix <- predict(model, newdata = newdata, type = "prob")
    if (is.null(probs_matrix) || !(positive_class_shap %in% colnames(probs_matrix))) {
      # 处理错误...
      return(rep(NA_real_, nrow(newdata)))
    }
    return(probs_matrix[, positive_class_shap])
    
  } else if (inherits(model, "xgb.Booster")) {
    # xgboost 需要 dmatrix，并且不需要因子转换
    # 确保所有列都是数值型
    numeric_newdata <- data.matrix(newdata)
    xgb_newdata <- xgboost::xgb.DMatrix(data = numeric_newdata)
    probs_vector <- predict(model, newdata = xgb_newdata)
    # xgboost predict 直接返回 positive class 的概率 (如果是二分类)
    return(probs_vector)
    
  } else if (inherits(model, "naive_bayes")) {
    # naivebayes 包可能需要因子，也可能处理数值，取决于训练方式
    # 尝试因子转换
    if ("sex" %in% colnames(newdata)) {
      newdata$sex <- factor(
        newdata$sex,
        levels = c(0, 1),
        labels = levels(trainData_final$sex)
      )
    }
    probs_matrix <- predict(model, newdata = newdata, type = "prob")
    if (is.null(probs_matrix) || !(positive_class_shap %in% colnames(probs_matrix))) {
      # 处理错误...
      return(rep(NA_real_, nrow(newdata)))
    }
    return(probs_matrix[, positive_class_shap])
    
  } else if (inherits(model, "lda")) {
    # MASS::lda 通常处理数值型数据
    pred_obj <- predict(model, newdata = newdata)
    # lda 的 posterior 矩阵
    probs_matrix <- pred_obj$posterior
    if (is.null(probs_matrix) || !(positive_class_shap %in% colnames(probs_matrix))) {
      # 处理错误...
      return(rep(NA_real_, nrow(newdata)))
    }
    return(probs_matrix[, positive_class_shap])
    
  } else if (inherits(model, "nnet.formula") || inherits(model, "nnet")) {
    # nnet 包，通常处理数值型
    probs_matrix <- predict(model, newdata = newdata, type = "raw") # nnet 使用 raw
    # nnet 可能返回单列或多列，取决于类别数
    if (ncol(probs_matrix) == 1) {
      # 假设单列是 positive class
      return(as.vector(probs_matrix))
    } else if (positive_class_shap %in% colnames(probs_matrix)){
      return(probs_matrix[, positive_class_shap])
    } else {
      # 处理错误...
      return(rep(NA_real_, nrow(newdata)))
    }
    
  } else if (inherits(model, "gbm.object")) {
    # gbm 包，通常处理数值型
    # gbm 需要指定 n.trees
    best_iter <- model$n.trees # 或者 caret 结果中的最佳迭代次数
    probs_vector <- predict(model, newdata = newdata, n.trees = best_iter, type = "response")
    # gbm predict type="response" 返回 positive class 概率
    return(probs_vector)
    
  } else if (inherits(model, "glmnet")) {
    # glmnet (Ridge/Lasso)，需要数值矩阵
    # 需要指定 lambda 值，通常是 caret 找到的最佳值
    best_lambda <- model$lambdaOpt # 或者从 caret 模型获取
    # 确保输入是矩阵
    numeric_newdata <- data.matrix(newdata)
    probs_matrix <- predict(model, newx = numeric_newdata, s = best_lambda, type = "response")
    # glmnet family="binomial" type="response" 返回 positive class 概率
    return(as.vector(probs_matrix))
    
  } else if (inherits(model, "knn3")) {
    # caret 的 knn，通常处理数值型
    probs_matrix <- predict(model, newdata = newdata, type = "prob")
    if (is.null(probs_matrix) || !(positive_class_shap %in% colnames(probs_matrix))) {
      # 处理错误...
      return(rep(NA_real_, nrow(newdata)))
    }
    return(probs_matrix[, positive_class_shap])
  }
  # ... 其他模型类型 ...
  
  else {
    stop("未知的模型类型，无法在 pred_wrapper 中处理: ", class(model)[1])
  }
}



cat("开始计算 SHAP 值... (这可能需要几分钟)\n")

# --- 4. 计算 SHAP 值 ---
# 使用较小的 nsim (e.g., 50) 以加快速度, 100-200 更稳定
# 使用 explainer_data (训练集) 作为背景数据
# 使用 eval_data (测试集) 作为要解释的数据

# 调试信息: 检查数据和函数
cat("--- SHAP 调试信息 开始 ---\n")
cat("最佳模型对象 (原始 caret train 对象):\n")
print(class(best_model_obj))
cat("使用的底层模型对象 (best_model_obj$finalModel):\n")
underlying_model <- best_model_obj$finalModel
print(class(underlying_model))

cat("\n解释器背景数据 (explainer_data) 结构 (给 kernelshap 的输入):\n")
print(str(explainer_data)) # 应该是数值型，包括 sex=0/1

cat("\n评估数据 (eval_data) 结构 (给 kernelshap 的输入):\n")
print(str(eval_data)) # 应该是数值型，包括 sex=0/1

cat("\n测试 pred_wrapper 函数 (使用底层模型):\n")
test_pred_input <- head(eval_data) # 取几行测试
test_pred_output <- tryCatch(
  pred_wrapper(underlying_model, test_pred_input), # <--- 使用底层模型测试
  error = function(e) { paste("pred_wrapper 错误:", e$message) }
)
cat("测试输出 (pred_wrapper 的结果):\n")
print(test_pred_output)
cat("测试输出类型:", class(test_pred_output), "\n")
cat("--- SHAP 调试信息 结束 ---\n\n")


set.seed(42)
shap_values <- NULL
tryCatch({
  # 强制调用默认方法以避免 S3 调度问题
  shap_values <- kernelshap:::kernelshap.default( # <--- 使用 ::: 显式调用 default 方法
    object = underlying_model,  # <--- 使用底层模型对象
    X = explainer_data,        # 背景数据 (数值型)
    X_pred = eval_data,        # 解释数据 (数值型)
    pred_fun = pred_wrapper,   # 包装函数 (内部不转换因子)
    nsim = 50,                 # 减少 nsim 以加快速度
    parallel = FALSE
  )
}, error = function(e) {
  cat(sprintf("SHAP 计算失败: %s\n", e$message))
  cat("请检查 pred_wrapper 函数是否正确处理了模型类型及其预测输出，以及输入数据是否为数值型。\n")
})


if (!is.null(shap_values)) {
  cat("SHAP 值计算完成。\n")
  
  # --- 5. 创建 SHAP 可视化对象 ---
  # 确保 X 是数值型的 (sex 已经是)
  sv <- shapviz(shap_values, X = eval_data)
  
  # --- 6. 绘制并保存 SHAP 蜂窝图 (Beeswarm) ---
  cat("生成 SHAP 蜂窝图...\n")
  
  # 调整 ggsave 的宽度以容纳所有特征
  plot_width <- 7 + (ncol(eval_data) * 0.1)
  plot_height <- max(5, (ncol(eval_data) * 0.3))
  
  # sv_plot_beeswarm 自动返回一个 ggplot 对象
  p_shap_beeswarm <- sv_plot_beeswarm(sv, max_display = 20) + # 最多显示20个特征
    labs(title = sprintf("SHAP 蜂窝图 (模型: %s)", best_model_name)) +
    theme(plot.title = element_text(hjust = 0.5))
  
  ggsave(
    paste0(base_path, "shap_beeswarm_plot.png"),
    plot = p_shap_beeswarm,
    width = plot_width,
    height = plot_height,
    dpi = 300
  )
  
  # --- 7. 绘制并保存 SHAP 特征重要性图 (条形图) ---
  cat("生成 SHAP 特征重要性图...\n")
  p_shap_importance <- sv_plot_importance(sv, max_display = 20) +
    labs(title = sprintf("SHAP 特征重要性 (模型: %s)", best_model_name)) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  ggsave(
    paste0(base_path, "shap_importance_plot.png"),
    plot = p_shap_importance,
    width = 10,
    height = 8,
    dpi = 300
  )
  
  cat("✓ SHAP 图表已保存。\n")
  print(p_shap_beeswarm)
  print(p_shap_importance)
  
} else {
  cat("由于 SHAP 计算失败, 跳过 SHAP 绘图。\n")
}


cat("\n\n====== 脚本执行完毕 ======\n")

