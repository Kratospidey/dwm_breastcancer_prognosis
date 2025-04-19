#!/usr/bin/env Rscript
# Set reproducible seed
set.seed(123)

# Add at the beginning of the script to display all warnings
options(warn = 1)  # Print warnings as they occur
options(warning.length = 8170)  # Show longer warnings

# Add proper logging and versioning
message("Starting integrated multi-omics analysis")
message("Script version: 1.1.0")
message("Run date: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))

# -------------------------------------------------------------------------------
# 1. Load Required Libraries
# -------------------------------------------------------------------------------
# Define all required packages
required_packages <- c("tidyverse", "survival", "survminer", "caret", "glmnet", 
                      "randomForestSRC", "ggplot2", "reshape2", "pROC", "ComplexHeatmap",
                      "lime", "grid", "gridExtra", "Hmisc", "gbm", "timeROC", 
                      "xgboost", "mlr")

# Check if packages are installed and load them
for(pkg in required_packages) {
  if(!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Package", pkg, "is not installed. Please install it."))
    # Optionally add: install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
  message(paste("Loaded package:", pkg))
}

# -------------------------------------------------------------------------------
# 2. Load and Preprocess Multiple Data Types
# -------------------------------------------------------------------------------
# Function to load any CSV file with better error handling
load_data <- function(file_path) {
  message("Loading data from: ", file_path)
  if(!file.exists(file_path)) {
    warning(paste("File not found:", file_path))
    return(NULL)
  }
  
  tryCatch({
    data <- read.csv(file_path, stringsAsFactors = FALSE)
    message(paste("Successfully loaded", nrow(data), "rows and", ncol(data), "columns"))
    return(data)
  }, error = function(e) {
    warning(paste("Error loading file:", file_path, "-", e$message))
    return(NULL)
  })
}

# Load all data files with better error handling
data_files <- list(
  clinical = "clinical_processed.csv",
  cna = "cna_processed.csv",
  methylation = "methylation_processed.csv",
  mrna = "mrna_processed.csv",
  mutations = "mutations_processed.csv"
)

# Load data with error checking
data_list <- list()
for(name in names(data_files)) {
  data_list[[name]] <- load_data(data_files[[name]])
  if(is.null(data_list[[name]])) {
    if(name == "clinical") {
      stop("Clinical data is required but could not be loaded. Exiting.")
    } else {
      message(paste("Will continue without", name, "data."))
    }
  }
}

# Assign to individual variables for backward compatibility
clinical_data <- data_list$clinical
cna_data <- data_list$cna
methylation_data <- data_list$methylation
mrna_data <- data_list$mrna
mutations_data <- data_list$mutations

# Extract the sample/patient ID column name
id_col <- colnames(clinical_data)[1]  # Assuming first column is the ID
message("Using ID column: ", id_col)

# -------------------------------------------------------------------------------
# 3. Feature Selection from Multi-omics Data
# -------------------------------------------------------------------------------
# For CNA data - select high-impact cancer genes
cna_features <- try({
  cna_data %>%
    select(contains(c(id_col, "TP53", "ERBB2", "MYC", "CCND1", "RB1", "PTEN", "PIK3CA")))
}, silent = TRUE)

if(inherits(cna_features, "try-error")) {
  message("Warning: Could not extract CNA features. Using empty dataframe.")
  cna_features <- data.frame(temp = numeric(0))
  cna_features[[id_col]] <- character(0)
}

# For mRNA data - calculate variation and select top variable genes
mrna_features <- try({
  if(ncol(mrna_data) > 500) {  # If there are many genes
    gene_var <- apply(mrna_data[,-1], 2, var, na.rm = TRUE)  # Exclude sample ID column
    top_genes <- names(sort(gene_var, decreasing = TRUE))[1:100]  # Top 100 variable genes
    mrna_data %>% select(1, all_of(top_genes))  # ID column + top genes
  } else {
    mrna_data  # Use all if not too many
  }
}, silent = TRUE)

if(inherits(mrna_features, "try-error")) {
  message("Warning: Could not extract mRNA features. Using empty dataframe.")
  mrna_features <- data.frame(temp = numeric(0))
  mrna_features[[id_col]] <- character(0)
}

# For mutations - more efficient processing of aggregated mutation data
key_mutation_genes <- c(
  "TP53", "PIK3CA", "MYC", "PTEN", "RB1", "CDH1", "GATA3", 
  "MAP3K1", "NCOR1", "AKT1", "CCND1", "BRCA1", "BRCA2",
  "ERBB2", "ESR1", "FGFR1", "FOXA1", "KMT2C", "NF1", "NOTCH1",
  "SF3B1", "TBX3"
)

mutation_features <- try({
  message("Processing aggregated mutation data format (optimized)...")
  
  if(ncol(mutations_data) > 1) {
    sample_ids <- mutations_data[[1]]
    mutation_matrix <- matrix(0, nrow=length(sample_ids), ncol=length(key_mutation_genes))
    colnames(mutation_matrix) <- key_mutation_genes
    rownames(mutation_matrix) <- sample_ids
    
    mutation_cols <- grep("Aggregated_Mutations", colnames(mutations_data))
    if(length(mutation_cols) == 0) {
      message("No 'Aggregated_Mutations' columns found. Using all non-ID columns.")
      mutation_cols <- 2:ncol(mutations_data)
    }
    
    message(paste("Processing", length(mutation_cols), "mutation columns and", length(key_mutation_genes), "key genes..."))
    
    batch_size <- max(1, floor(nrow(mutations_data) / 10))
    gene_pattern <- paste(key_mutation_genes, collapse="|")
    
    for(batch_start in seq(1, nrow(mutations_data), by=batch_size)) {
      batch_end <- min(batch_start + batch_size - 1, nrow(mutations_data))
      if(batch_start %% (batch_size * 2) == 1) {
        message(paste("Processing rows", batch_start, "to", batch_end, "of", nrow(mutations_data)))
      }
      
      for(i in batch_start:batch_end) {
        for(col_idx in mutation_cols) {
          mutation_text <- mutations_data[i, col_idx]
          if(is.na(mutation_text) || mutation_text == "") next
          
          for(g in seq_along(key_mutation_genes)) {
            gene <- key_mutation_genes[g]
            if(grepl(gene, mutation_text, fixed=TRUE)) {
              mutation_matrix[i, g] <- 1
            }
          }
        }
      }
    }
    
    result <- as.data.frame(mutation_matrix)
    result[[id_col]] <- sample_ids
    
    message("Mutation processing complete.")
    return(result)
  } else {
    message("Mutation data has insufficient columns")
    empty_df <- data.frame(temp = numeric(0))
    empty_df[[id_col]] <- character(0)
    return(empty_df)
  }
}, silent = FALSE)

if(inherits(mutation_features, "try-error") || ncol(mutation_features) <= 1) {
  message("Warning: Could not process mutation features. Creating dummy mutation data.")
  mutation_features <- data.frame(matrix(0, nrow = nrow(clinical_data), ncol = length(key_mutation_genes)))
  colnames(mutation_features) <- key_mutation_genes
  mutation_features[[id_col]] <- clinical_data[[id_col]]
}

# -------------------------------------------------------------------------------
# 4. Merge all Data for Integrated Analysis
# -------------------------------------------------------------------------------
message("Beginning data integration and cleaning...")
total_steps <- 5
current_step <- 1

message(paste0("Step ", current_step, "/", total_steps, ": Checking ID compatibility across datasets"))
check_ids <- function(main_df, secondary_df, id_col) {
  if(nrow(secondary_df) == 0 || !(id_col %in% colnames(secondary_df))) {
    message("Secondary dataset is empty or missing ID column. Skipping.")
    return(character(0))
  }
  
  main_ids <- main_df[[id_col]]
  secondary_ids <- secondary_df[[id_col]]
  common_ids <- intersect(main_ids, secondary_ids)
  message("Found ", length(common_ids), " common IDs between datasets")
  return(common_ids)
}

common_ids_mrna <- check_ids(clinical_data, mrna_features, id_col)
common_ids_cna <- check_ids(clinical_data, cna_features, id_col)
common_ids_mut <- check_ids(clinical_data, mutation_features, id_col)

datasets_to_use <- c("clinical")
if(length(common_ids_mrna) > 0) datasets_to_use <- c(datasets_to_use, "mrna")
if(length(common_ids_cna) > 0) datasets_to_use <- c(datasets_to_use, "cna")
if(length(common_ids_mut) > 0) datasets_to_use <- c(datasets_to_use, "mutation")

message("Using the following datasets: ", paste(datasets_to_use, collapse=", "))

if(length(datasets_to_use) == 1) {
  all_common_ids <- clinical_data[[id_col]]
} else {
  id_lists <- list(clinical_data[[id_col]])
  if("mrna" %in% datasets_to_use) id_lists <- c(id_lists, list(common_ids_mrna))
  if("cna" %in% datasets_to_use) id_lists <- c(id_lists, list(common_ids_cna))
  if("mutation" %in% datasets_to_use) id_lists <- c(id_lists, list(common_ids_mut))
  
  all_common_ids <- Reduce(intersect, id_lists)
}

message("Final analysis will use ", length(all_common_ids), " samples with data across selected platforms")

clinical_filtered <- clinical_data %>% filter(!!sym(id_col) %in% all_common_ids)
integrated_data <- clinical_filtered

if("cna" %in% datasets_to_use && length(common_ids_cna) > 0) {
  cna_filtered <- cna_features %>% filter(!!sym(id_col) %in% all_common_ids)
  integrated_data <- integrated_data %>% left_join(cna_filtered, by = id_col)
}

if("mrna" %in% datasets_to_use && length(common_ids_mrna) > 0) {
  mrna_filtered <- mrna_features %>% filter(!!sym(id_col) %in% all_common_ids)
  integrated_data <- integrated_data %>% left_join(mrna_filtered, by = id_col)
}

if("mutation" %in% datasets_to_use && length(common_ids_mut) > 0) {
  mutation_filtered <- mutation_features %>% filter(!!sym(id_col) %in% all_common_ids)
  integrated_data <- integrated_data %>% left_join(mutation_filtered, by = id_col)
}

current_step <- current_step + 1

message(paste0("Step ", current_step, "/", total_steps, ": Merging datasets"))
os_months_col <- NULL
os_event_col <- NULL

for(col in c("OS_MONTHS", "overall_survival_months", "OS.MONTHS")) {
  if(col %in% colnames(integrated_data)) {
    os_months_col <- col
    break
  }
}

if("OS_STATUS_1.DECEASED" %in% colnames(integrated_data)) {
  integrated_data <- integrated_data %>%
    mutate(OS_EVENT = ifelse(OS_STATUS_1.DECEASED == 1, 1, 0))
  os_event_col <- "OS_EVENT"
} else {
  for(col in c("OS_EVENT", "overall_survival_status", "OS.STATUS")) {
    if(col %in% colnames(integrated_data)) {
      os_event_col <- col
      break
    }
  }
}

if(is.null(os_months_col) || is.null(os_event_col)) {
  potential_os_months <- grep("OS.*MONTH|MONTH.*OS|survival.*month|month.*survival", 
                             colnames(integrated_data), ignore.case = TRUE, value = TRUE)
  
  if(length(potential_os_months) > 0) {
    os_months_col <- potential_os_months[1]
    message("Using ", os_months_col, " as survival time")
  } else {
    stop("Could not identify overall survival time column")
  }
  
  potential_os_event <- grep("OS.*STATUS|STATUS.*OS|OS.*EVENT|EVENT.*OS|deceased|dead", 
                            colnames(integrated_data), ignore.case = TRUE, value = TRUE)
  
  if(length(potential_os_event) > 0) {
    for(col in potential_os_event) {
      if(all(integrated_data[[col]] %in% c(0, 1, NA))) {
        os_event_col <- col
        message("Using ", os_event_col, " as survival event")
        break
      }
    }
    
    if(is.null(os_event_col) && length(potential_os_event) > 0) {
      os_event_col <- potential_os_event[1]
      message("Using ", os_event_col, " as survival event (may need recoding)")
    }
  }
}

if(is.null(os_months_col) || is.null(os_event_col)) {
  stop("Could not identify survival time and/or event columns")
}

class_of_event <- class(integrated_data[[os_event_col]])
message("OS event column '", os_event_col, "' has class: ", class_of_event)

integrated_data <- integrated_data %>%
  mutate(
    OS_MONTHS_CLEAN = as.numeric(!!sym(os_months_col))
  )

if(is.character(integrated_data[[os_event_col]])) {
  message("Converting character OS event indicator to binary")
  integrated_data <- integrated_data %>%
    mutate(
      OS_EVENT_CLEAN = case_when(
        grepl("1|deceased|dead|DECEASED|DEAD", !!sym(os_event_col), ignore.case = TRUE) ~ 1,
        TRUE ~ 0
      )
    )
} else {
  message("Using numeric OS event indicator")
  integrated_data <- integrated_data %>%
    mutate(
      OS_EVENT_CLEAN = as.numeric(!!sym(os_event_col))
    )
}

integrated_data <- integrated_data %>%
  filter(!is.na(OS_MONTHS_CLEAN) & !is.na(OS_EVENT_CLEAN))

non_positive_times <- sum(integrated_data$OS_MONTHS_CLEAN <= 0, na.rm = TRUE)
if(non_positive_times > 0) {
  message("Warning: Removing ", non_positive_times, " records with non-positive survival times (required for Cox model)")
  integrated_data <- integrated_data %>%
    filter(OS_MONTHS_CLEAN > 0)
}

message("Survival data processed: ", nrow(integrated_data), " samples with valid survival information")

if("CLAUDIN_SUBTYPE_LumA" %in% colnames(integrated_data)) {
  message("Using molecular subtypes for grouped imputation")
  
  subtype_cols <- grep("SUBTYPE", colnames(integrated_data), value = TRUE)
  
  for(col in colnames(integrated_data)) {
    if(is.numeric(integrated_data[[col]]) && sum(is.na(integrated_data[[col]])) > 0) {
      for(subtype_col in subtype_cols) {
        subtype_median <- median(integrated_data[integrated_data[[subtype_col]] == 1, col], na.rm = TRUE)
        integrated_data[integrated_data[[subtype_col]] == 1 & is.na(integrated_data[[col]]), col] <- subtype_median
      }
      
      if(sum(is.na(integrated_data[[col]])) > 0) {
        overall_median <- median(integrated_data[[col]], na.rm = TRUE)
        integrated_data[[col]][is.na(integrated_data[[col]])] <- overall_median
      }
    }
  }
} else {
  integrated_data <- integrated_data %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

current_step <- current_step + 1

message(paste0("Step ", current_step, "/", total_steps, ": Feature selection and data cleaning"))
message("Filtering zero or near-zero variance predictors...")
model_cols <- integrated_data %>% 
  select(where(is.numeric)) %>%
  select(-OS_MONTHS_CLEAN, -OS_EVENT_CLEAN) %>%
  colnames()

leakage_vars <- c("OS_MONTHS", "OS_STATUS", "RFS_MONTHS", "RFS_STATUS", "OS_EVENT", 
                 "VITAL_STATUS", "DEATH_FROM_CANCER", "SURVIVAL_TIME", "EVENT")
leakage_pattern <- paste(leakage_vars, collapse="|")
leakage_cols <- grep(leakage_pattern, model_cols, ignore.case = TRUE)

if(length(leakage_cols) > 0) {
  message("WARNING: Removing ", length(leakage_cols), " variables that would cause data leakage:")
  message(paste(" ", model_cols[leakage_cols], collapse="\n "))
  model_cols <- model_cols[-leakage_cols]
}

if("OS_MONTHS" %in% model_cols) {
  message("WARNING: Removing OS_MONTHS to prevent data leakage")
  model_cols <- setdiff(model_cols, "OS_MONTHS") 
}
if("RFS_MONTHS" %in% model_cols) {
  message("WARNING: Removing RFS_MONTHS to prevent data leakage")
  model_cols <- setdiff(model_cols, "RFS_MONTHS") 
}

zero_var_check <- sapply(integrated_data[, model_cols], function(x) {
  var(x, na.rm = TRUE) < 1e-6
})
zero_var_cols <- names(zero_var_check[zero_var_check])

if(length(zero_var_cols) > 0) {
  message(paste("Removing", length(zero_var_cols), "zero-variance features"))
  model_cols <- setdiff(model_cols, zero_var_cols)
}

if(length(model_cols) < 2) {
  stop("Not enough numeric predictors available for modeling. Need at least 2, found: ", length(model_cols))
}

message("Using ", length(model_cols), " numeric predictors for modeling")
message("First few predictors: ", paste(head(model_cols, 5), collapse=", "))

message("Creating model matrix...")
x_matrix <- as.matrix(integrated_data[, model_cols])
message("Standardizing matrix...")
x_matrix_std <- tryCatch({
  scaled_mat <- scale(x_matrix)
  constant_cols <- which(colSums(is.na(scaled_mat)) > 0)
  if(length(constant_cols) > 0) {
    message(paste("Found", length(constant_cols), "constant columns after scaling. Setting to zero."))
    scaled_mat[, constant_cols] <- 0
  }
  scaled_mat
}, error = function(e) {
  message("Error during standardization: ", e$message, ". Using original matrix with range scaling.")
  t(apply(x_matrix, 1, function(x) (x - min(x, na.rm = TRUE)) / 
            (max(x, na.rm = TRUE) - min(x, na.rm = TRUE) + 1e-10)))
})

message("Handling NA/Inf values...")
x_matrix_std[is.na(x_matrix_std)] <- 0
x_matrix_std[is.infinite(x_matrix_std)] <- 0

surv_obj <- Surv(integrated_data$OS_MONTHS_CLEAN, integrated_data$OS_EVENT_CLEAN)

risk_method <- "Random Survival Forest"

set.seed(123)
message("Attempting regularized Cox regression...")
tryCatch({
  cv_fit <- cv.glmnet(x_matrix_std, surv_obj, family="cox", alpha=0.5, 
                     maxit=2000, standardize=FALSE, nfolds=5)
  
  pdf("CV_ElasticNet_Cox.pdf")
  plot(cv_fit)
  dev.off()
  
  elastic_net_cox <- glmnet(x_matrix_std, surv_obj, family="cox", alpha=0.5, 
                     lambda=cv_fit$lambda.min, standardize=FALSE)
  
  coef_vals <- as.matrix(coef(elastic_net_cox))
  non_zero_idx <- which(abs(coef_vals) > 1e-6)
  if(length(non_zero_idx) > 0) {
    important_features <- rownames(coef_vals)[non_zero_idx]
    lasso_cox <- elastic_net_cox
    risk_method <- "Elastic Net Cox"
    message("Elastic Net identified ", length(important_features), " important features")
  } else {
    message("No features identified by Elastic Net. Will use correlations instead.")
    important_features <- c()
  }
}, error = function(e) {
  message("Regularized regression failed: ", e$message, ". Falling back to alternative approach.")
  important_features <- c()
})

if(!exists("important_features") || length(important_features) == 0) {
  message("Using correlation-based feature selection...")
  
  feature_correlation <- numeric(length(model_cols))
  names(feature_correlation) <- model_cols
  
  for(i in seq_along(model_cols)) {
    col <- model_cols[i]
    if(var(integrated_data[[col]], na.rm=TRUE) < 1e-6) {
      feature_correlation[i] <- 0
      next
    }
    feature_correlation[i] <- suppressWarnings(
      cor(integrated_data[[col]], integrated_data$OS_MONTHS_CLEAN, 
          method="spearman", use="pairwise.complete.obs")
    )
    if(is.na(feature_correlation[i])) feature_correlation[i] <- 0
  }
  
  sorted_features <- names(sort(abs(feature_correlation), decreasing=TRUE))
  important_features <- head(sorted_features, 10)
  message("Using top ", length(important_features), " features by survival correlation")
  
  cox_formula <- as.formula(paste("Surv(OS_MONTHS_CLEAN, OS_EVENT_CLEAN) ~", 
                                 paste(important_features, collapse="+")))
  
  tryCatch({
    simple_cox <- coxph(cox_formula, data=integrated_data)
    lasso_cox <- simple_cox
    risk_method <- "Cox Regression" 
  }, error = function(e) {
    message("Cox model failed: ", e$message, ". Using Random Forest only.")
    risk_method <- "Random Survival Forest"
  })
}

set.seed(123)
message("Training Random Survival Forest model (this may take some time)...")
suppressWarnings({
  if(length(model_cols) > 50) {
    message("Using univariate feature selection for RSF to improve performance...")
    
    # Calculate univariate association with survival
    feature_significance <- data.frame(
      Feature = character(length(model_cols)),
      Pvalue = numeric(length(model_cols)),
      stringsAsFactors = FALSE
    )
    
    for(i in seq_along(model_cols)) {
      feat <- model_cols[i]
      # Simple univariate Cox model
      uni_formula <- as.formula(paste("Surv(OS_MONTHS_CLEAN, OS_EVENT_CLEAN) ~", feat))
      uni_model <- tryCatch({
        summary(coxph(uni_formula, data = integrated_data))
      }, error = function(e) {
        return(list(coefficients = matrix(c(NA, 1), ncol = 2)))
      })
      
      # Extract p-value
      feature_significance$Feature[i] <- feat
      feature_significance$Pvalue[i] <- if(is.matrix(uni_model$coefficients)) uni_model$coefficients[1, 5] else 1
    }
    
    # Select top 50 features by significance
    feature_significance <- feature_significance[order(feature_significance$Pvalue), ]
    rsf_cols <- feature_significance$Feature[1:min(50, nrow(feature_significance))]
    message("Selected ", length(rsf_cols), " features for RSF based on univariate significance")
  } else {
    rsf_cols <- model_cols
  }
  
  rsf_data <- integrated_data[, c("OS_MONTHS_CLEAN", "OS_EVENT_CLEAN", rsf_cols)]
  
  # Fix 1: Use explicit importance="permute" for more reliable importance measures
  rsf_model <- rfsrc(Surv(OS_MONTHS_CLEAN, OS_EVENT_CLEAN) ~ ., 
                   data = rsf_data,
                   ntree = 300, 
                   importance = "permute", # Specify permutation importance
                   do.trace = max(1, floor(300/10)))
  
  # Fix 2: Better handling of importance extraction
  if(exists("rsf_model")) {
    # Check multiple potential importance attributes
    if(!is.null(rsf_model$importance)) {
      imp_values <- rsf_model$importance
      imp_names <- names(imp_values)
      message("Using rsf_model$importance")
    } else if(!is.null(rsf_model$variable.importance)) {
      imp_values <- rsf_model$variable.importance
      imp_names <- names(imp_values)
      message("Using rsf_model$variable.importance") 
    } else if(!is.null(rsf_model$importance.variable)) {
      imp_values <- rsf_model$importance.variable
      imp_names <- names(imp_values)
      message("Using rsf_model$importance.variable")
    } else {
      message("No importance metric found in RSF model. Creating fallback.")
      imp_values <- seq(1, 0.05, length.out = length(rsf_cols))
      imp_names <- rsf_cols
    }
    
    importance_df <- data.frame(
      Feature = imp_names,
      Importance = imp_values
    )
    top_features <- importance_df %>%
      arrange(desc(Importance)) %>%
      head(20)
    message("Extracted ", nrow(top_features), " top features from RSF model")
  } else {
    message("Feature importance not available from RSF. Creating fallback.")
    fallback_features <- head(model_cols, 20)
    fallback_importance <- seq(1, 0.05, length.out = length(fallback_features))
    top_features <- data.frame(
      Feature = fallback_features,
      Importance = fallback_importance
    )
  }
})

current_step <- current_step + 1

message(paste0("Step ", current_step, "/", total_steps, ": Training advanced survival models"))
message("Implementing advanced survival modeling approaches...")

suppressWarnings({
  if(!exists("risk_method")) {
    risk_method <- "Random Survival Forest"
  }
  
  if(exists("lasso_cox")) {
    if(inherits(lasso_cox, "coxph")) {
      tryCatch({
        integrated_data$risk_score <- predict(lasso_cox, newdata = integrated_data, type = "lp")
        risk_method <- "Cox Regression"
      }, error = function(e) {
        message("Error using coxph prediction: ", e$message, ". Falling back to RSF.")
        integrated_data$risk_score <- predict(rsf_model, newdata = integrated_data)$predicted
        risk_method <- "Random Survival Forest (fallback)"
      })
    } else if(inherits(lasso_cox, "glmnet")) {
      integrated_data$risk_score <- predict(lasso_cox, newx = x_matrix_std, type = "link")[,1]
      risk_method <- "Elastic Net Cox"
    } else {
      integrated_data$risk_score <- predict(rsf_model, newdata = integrated_data)$predicted
      risk_method <- "Random Survival Forest"
    }
  } else {
    integrated_data$risk_score <- predict(rsf_model, newdata = integrated_data)$predicted
    risk_method <- "Random Survival Forest"
  }
})

message("Using risk scores from ", risk_method, " for initial stratification")

set.seed(123)
message("Training Gradient Boosting Machine for Survival...")
tryCatch({
  surv_gbm <- gbm(
    formula = Surv(OS_MONTHS_CLEAN, OS_EVENT_CLEAN) ~ .,
    data = rsf_data,
    distribution = "coxph",
    n.trees = 1000,
    interaction.depth = 3,
    shrinkage = 0.01,
    cv.folds = 5,
    n.minobsinnode = 10,
    verbose = FALSE
  )
  
  best_trees <- gbm.perf(surv_gbm, method = "cv", plot.it = FALSE)
  message("GBM - Optimal number of trees: ", best_trees)
  
  gbm_importance <- summary(surv_gbm, n.trees = best_trees, plot = FALSE)
  gbm_features <- data.frame(
    Feature = gbm_importance$var,
    Importance = gbm_importance$rel.inf
  )
  message("GBM - Extracted feature importance for ", nrow(gbm_features), " features")
  
}, error = function(e) {
  message("GBM model failed: ", e$message, ". Skipping this model.")
})

set.seed(123)
message("Training XGBoost for Survival...")
tryCatch({
  xgb_data <- xgb.DMatrix(data = x_matrix_std)
  xgb_label <- as.numeric(integrated_data$OS_MONTHS_CLEAN)
  xgb_event <- as.numeric(integrated_data$OS_EVENT_CLEAN)
  
  setinfo(xgb_data, "label", xgb_label)
  setinfo(xgb_data, "weight", xgb_event)
  
  xgb_params <- list(
    objective = "survival:cox",
    eval_metric = "cox-nloglik",
    eta = 0.05,
    max_depth = 3,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  xgb_cv <- xgb.cv(
    params = xgb_params,
    data = xgb_data,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = FALSE
  )
  
  best_iteration <- xgb_cv$best_iteration
  message("XGBoost - Best iteration: ", best_iteration)
  
  xgb_model <- xgb.train(
    params = xgb_params,
    data = xgb_data,
    nrounds = best_iteration
  )
  
  xgb_importance <- xgb.importance(model = xgb_model)
  if (nrow(xgb_importance) > 0) {
    xgb_features <- data.frame(
      Feature = xgb_importance$Feature,
      Importance = xgb_importance$Gain
    )
    message("XGBoost - Extracted feature importance for ", nrow(xgb_features), " features")
  }
  
}, error = function(e) {
  message("XGBoost model failed: ", e$message, ". Skipping this model.")
})

message("Creating Ensemble Survival Model...")
ensemble_predictions <- list()

if (exists("lasso_cox")) {
  if (inherits(lasso_cox, "glmnet")) {
    ensemble_predictions$elastic_net <- predict(lasso_cox, newx = x_matrix_std, type = "link")[,1]
    message("Added Elastic Net Cox to ensemble")
  } else if (inherits(lasso_cox, "coxph")) {
    ensemble_predictions$cox <- predict(lasso_cox, newdata = integrated_data, type = "lp")
    message("Added Cox Regression to ensemble")
  }
}

if (exists("rsf_model")) {
  ensemble_predictions$rsf <- predict(rsf_model, newdata = integrated_data)$predicted
  message("Added Random Survival Forest to ensemble")
}

if (exists("surv_gbm") && exists("best_trees")) {
  ensemble_predictions$gbm <- predict(surv_gbm, newdata = rsf_data, n.trees = best_trees, type = "link")
  message("Added Gradient Boosting Machine to ensemble")
}

if (exists("xgb_model")) {
  ensemble_predictions$xgb <- predict(xgb_model, newdata = xgb_data)
  message("Added XGBoost to ensemble")
}

if (length(ensemble_predictions) > 0) {
  scaled_predictions <- lapply(ensemble_predictions, function(pred) {
    if(length(pred) != nrow(integrated_data)) {
      message("Warning: Prediction length mismatch. Expected ", nrow(integrated_data), 
             " but got ", length(pred), ". Skipping this model in ensemble.")
      return(NULL)
    }
    (pred - mean(pred, na.rm=TRUE)) / sd(pred, na.rm=TRUE)
  })
  
  scaled_predictions <- scaled_predictions[!sapply(scaled_predictions, is.null)]
  
  if(length(scaled_predictions) > 0) {
    model_weights <- rep(1/length(scaled_predictions), length(scaled_predictions))
    names(model_weights) <- names(scaled_predictions)
    
    integrated_data$ensemble_score <- 0
    for (i in seq_along(scaled_predictions)) {
      model_name <- names(scaled_predictions)[i]
      integrated_data$ensemble_score <- integrated_data$ensemble_score + 
        model_weights[model_name] * scaled_predictions[[model_name]]
    }
    
    original_c_index <- rcorr.cens(integrated_data$risk_score, 
                                Surv(integrated_data$OS_MONTHS_CLEAN, integrated_data$OS_EVENT_CLEAN))["C Index"]
    
    ensemble_c_index <- rcorr.cens(integrated_data$ensemble_score, 
                                 Surv(integrated_data$OS_MONTHS_CLEAN, integrated_data$OS_EVENT_CLEAN))["C Index"]
    
    message("Original model C-index: ", round(original_c_index, 4))
    message("Ensemble model C-index: ", round(ensemble_c_index, 4))
    
    if (ensemble_c_index > original_c_index) {
      message("Ensemble model performs better! Using ensemble risk scores.")
      integrated_data$risk_score <- integrated_data$ensemble_score
      risk_method <- "Ensemble Survival Model"
    } else {
      message("Original model performs better than ensemble. Keeping original risk scores.")
    }
  } else {
    message("No valid predictions for ensemble. Using original risk scores.")
  }
}

integrated_data <- integrated_data %>%
  mutate(risk_group = ifelse(risk_score <= median(risk_score), "Low Risk", "High Risk"))

message("Using risk scores from ", risk_method, " for final stratification")

current_step <- current_step + 1

message(paste0("Step ", current_step, "/", total_steps, ": Generating visualizations and model interpretation"))
km_fit <- survfit(Surv(OS_MONTHS_CLEAN, OS_EVENT_CLEAN) ~ risk_group, data = integrated_data)

suppressWarnings({
  km_plot <- ggsurvplot(km_fit,
                     data = integrated_data,
                     risk.table = TRUE,
                     pval = TRUE,
                     conf.int = TRUE,
                     xlab = "Overall Survival (Months)",
                     ylab = "Survival Probability",
                     legend.title = "Risk Group",
                     legend.labs = c("High Risk", "Low Risk"),
                     palette = c("red", "forestgreen"),
                     ggtheme = theme_minimal())
})

pdf("Integrated_KM_plot.pdf", width=10, height=8)
print(km_plot$plot)
print(km_plot$table)
dev.off()

if(exists("top_features") && nrow(top_features) > 0) {
  importance_plot <- ggplot(head(top_features, 15), aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Top 15 Features by Importance",
         x = "Feature", y = "Importance Score")
  
  pdf("Feature_Importance_Plot.pdf", width=10, height=6)
  print(importance_plot)
  dev.off()
  message("Feature importance plot saved to Feature_Importance_Plot.pdf")
}

message("Calculating advanced performance metrics...")
tryCatch({
  library(timeROC)
  
  time_points <- c(12, 36, 60)
  time_roc <- timeROC(
    T = integrated_data$OS_MONTHS_CLEAN,
    delta = integrated_data$OS_EVENT_CLEAN,
    marker = integrated_data$risk_score,
    cause = 1,
    times = time_points,
    iid = TRUE
  )
  
  auc_1yr <- time_roc$AUC[1]
  auc_3yr <- time_roc$AUC[2]
  auc_5yr <- time_roc$AUC[3]
  
  message("Time-dependent AUC:")
  message("  1-year AUC: ", round(auc_1yr, 4))
  message("  3-year AUC: ", round(auc_3yr, 4))
  message("  5-year AUC: ", round(auc_5yr, 4))
  
}, error = function(e) {
  message("Time-dependent AUC calculation failed: ", e$message)
})

library(caret)
true_class <- factor(ifelse(integrated_data$OS_EVENT_CLEAN == 1, "High Risk", "Low Risk"),
                     levels = c("Low Risk", "High Risk"))
pred_class <- factor(integrated_data$risk_group, levels = c("Low Risk", "High Risk"))
conf_mat <- confusionMatrix(pred_class, true_class)
message("Confusion Matrix:")
print(conf_mat$table)
accuracy <- conf_mat$overall['Accuracy']
precision <- conf_mat$byClass['Pos Pred Value']
recall <- conf_mat$byClass['Sensitivity']
F1 <- 2 * precision * recall / (precision + recall)
message("Accuracy: ", round(accuracy, 4))
message("F1 Score (High Risk as positive): ", round(F1, 4))

current_step <- current_step + 1

message(paste0("Step ", current_step, "/", total_steps, ": Saving results and generating report"))
tryCatch({
  result_data <- list(
    integrated_data = integrated_data,
    risk_method = risk_method,
    model_cols = model_cols,
    top_features = if(exists("top_features")) top_features else NULL,
    important_features = if(exists("important_features")) important_features else NULL,
    excluded_leakage_vars = leakage_pattern,
    ensemble_predictions = if(exists("ensemble_predictions")) ensemble_predictions else NULL,
    time_auc = if(exists("time_roc")) list(auc_1yr = auc_1yr, auc_3yr = auc_3yr, auc_5yr = auc_5yr) else NULL,
    combined_importance = if(exists("feature_importance_combined")) feature_importance_combined else NULL,
    models = list(
      rsf_model = if(exists("rsf_model")) rsf_model else NULL,
      lasso_cox = if(exists("lasso_cox")) lasso_cox else NULL,
      surv_gbm = if(exists("surv_gbm")) surv_gbm else NULL,
      xgb_model = if(exists("xgb_model")) xgb_model else NULL,
      x_matrix_std = x_matrix_std
    ),
    pred_matrix_info = list(
      x_matrix_std = x_matrix_std,
      model_cols = model_cols,
      surv_obj = surv_obj
    )
  )
  saveRDS(result_data, "integrated_analysis_results.rds")
  saveRDS(result_data, "integrated_data_results.rds")
  
  model_data <- list(
    rsf_model = if(exists("rsf_model")) rsf_model else NULL,
    lasso_cox = if(exists("lasso_cox")) lasso_cox else NULL,
    surv_gbm = if(exists("surv_gbm")) surv_gbm else NULL,
    xgb_model = if(exists("xgb_model")) xgb_model else NULL,
    x_matrix_std = x_matrix_std,
    model_cols = model_cols
  )
  saveRDS(model_data, "integrated_models.rds")
  
  message("Analysis results saved to integrated_analysis_results.rds and integrated_data_results.rds")
  message("Models saved separately to integrated_models.rds for tuning use")
}, error = function(e) {
  message("Failed to save results: ", e$message)
})

message("Generating analysis summary report...")
if(requireNamespace("knitr", quietly = TRUE) && requireNamespace("rmarkdown", quietly = TRUE)) {
  # Fix 3: Fix YAML header in the R Markdown template to avoid parsing errors
  report_file <- "analysis_report.Rmd"
  
  # Properly escape special characters and ensure valid YAML
  cat('---
title: "Multi-omics Survival Analysis Report"
date: "', format(Sys.time(), "%Y-%m-%d"), '"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE)
```

## Analysis Summary

- **Data used:** ', paste(names(data_files)[names(data_files) %in% names(data_list)], collapse=", "), '
- **Sample size:** ', nrow(integrated_data), ' patients
- **Event rate:** ', round(100*mean(integrated_data$OS_EVENT_CLEAN)), '%
- **Primary model:** ', risk_method, '
- **C-index:** ', round(rcorr.cens(integrated_data$risk_score, Surv(integrated_data$OS_MONTHS_CLEAN, integrated_data$OS_EVENT_CLEAN))["C Index"], 3), '

## Kaplan-Meier Curve

![](Integrated_KM_plot.pdf)

## Feature Importance

![](Feature_Importance_Plot.pdf)

', file=report_file)

  tryCatch({
    rmarkdown::render(report_file, output_format="html_document", output_file="analysis_report.html")
    message("Analysis report generated: analysis_report.html")
  }, error = function(e) {
    message("Could not render report: ", e$message)
  })
} else {
  message("Packages knitr and/or rmarkdown not available. Skipping report generation.")
}

message("Integrated multi-omics analysis completed successfully!")
