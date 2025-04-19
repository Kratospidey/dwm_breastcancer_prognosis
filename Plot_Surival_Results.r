#!/usr/bin/env Rscript
# Advanced Visualization Script for Survival Analysis Results
# Creates publication-quality figures and interactive visualizations

# Load required libraries
library(tidyverse)
library(plotly)
library(DT)
library(htmlwidgets)
library(survival)
library(survminer)
library(ComplexHeatmap)
library(circlize)      # For color scales in heatmaps
library(RColorBrewer)  # For better color palettes
library(gridExtra)     # For arranging multiple plots
library(timeROC)       # For time-dependent ROC
library(ggsci)         # For scientific color palettes
library(viridis)       # For colorblind-friendly palettes
library(knitr)         # For tables
library(kableExtra)    # For fancy tables
library(reshape2)      # For data reshaping
library(corrplot)      # For correlation plots
library(ggpubr)        # For publication-ready plots
library(patchwork)     # For arranging multiple plots together
library(Hmisc)   # for rcorr.cens(), Somers' Dxy & C‑index

# Set options for high quality figures
options(repr.plot.width = 10, repr.plot.height = 8)
theme_set(theme_minimal(base_size = 14))

# Create output directory for figures
output_dir <- "figures"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Define utility functions for plotting
# ------------------------------------

# Function to create confusion matrix visualization
create_confusion_matrix_plot <- function(conf_mat, accuracy, precision, recall, F1) {
  # Create a data frame for ggplot
  plot_data <- as.data.frame(as.table(conf_mat))
  colnames(plot_data) <- c("Predicted", "Actual", "Count")
  
  # Calculate proportions for each row (percentage of prediction)
  plot_data <- plot_data %>%
    group_by(Predicted) %>%
    mutate(Percentage = round(100 * Count / sum(Count), 1)) %>%
    ungroup()
  
  # Create label for each cell with count and percentage
  plot_data$Label <- paste0(plot_data$Count, "\n(", plot_data$Percentage, "%)")
  
  # Generate the confusion matrix plot
  conf_plot <- ggplot(plot_data, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white", lwd = 1.5) +
    geom_text(aes(label = Label), color = "black", size = 5) +
    scale_fill_gradient(low = "white", high = "#4472C4") +
    theme_minimal(base_size = 14) +
    labs(
      title = "Confusion Matrix",
      subtitle = paste("Accuracy:", round(accuracy, 3),
                      "| Precision:", round(precision, 3),
                      "| Recall:", round(recall, 3),
                      "| F1:", round(F1, 3)),
      x = "Actual Risk Group",
      y = "Predicted Risk Group"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      panel.grid.major = element_blank()
    )
  
  return(conf_plot)
}

# Function to plot time-dependent ROC curves
plot_time_roc_curves <- function(time_roc_result, time_points, title = "Time-dependent ROC Curves") {
  # Extract data for plotting
  roc_data <- data.frame()
  
  for (i in 1:length(time_points)) {
    temp_data <- data.frame(
      FPR = time_roc_result$FP[, i],
      TPR = time_roc_result$TP[, i],
      Time = paste0(time_points[i]/12, "-Year"),
      AUC = round(time_roc_result$AUC[i], 3)
    )
    roc_data <- rbind(roc_data, temp_data)
  }
  
  # Create the ROC plot
  roc_plot <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Time)) +
    geom_line(size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = title,
      subtitle = paste("AUC:", paste(paste0(unique(roc_data$Time), ": ", unique(roc_data$AUC)), collapse = ", ")),
      x = "False Positive Rate (1-Specificity)",
      y = "True Positive Rate (Sensitivity)",
      color = "Time Point"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10)
    ) +
    coord_equal()
  
  return(roc_plot)
}

# Function to save both PDF (for publications) and interactive HTML versions
save_dual_format <- function(plot_obj, filename, width = 10, height = 8, interactive = TRUE) {
  # Save PDF version
  pdf_path <- file.path(output_dir, paste0(filename, ".pdf"))
  if(inherits(plot_obj, "gg")) {
    # For ggplot objects
    ggsave(pdf_path, plot_obj, width = width, height = height)
  } else if(inherits(plot_obj, "Heatmap")) {
    # For ComplexHeatmap objects
    pdf(pdf_path, width = width, height = height)
    draw(plot_obj)
    dev.off()
  } else {
    # For other objects
    pdf(pdf_path, width = width, height = height)
    print(plot_obj)
    dev.off()
  }
  
  # Save interactive HTML version if supported and requested
  if(interactive) {
    html_path <- file.path(output_dir, paste0(filename, ".html"))
    
    if(inherits(plot_obj, "gg")) {
      # For ggplot objects
      p_interactive <- ggplotly(plot_obj)
      htmlwidgets::saveWidget(p_interactive, html_path)
    } else if(inherits(plot_obj, "plotly")) {
      htmlwidgets::saveWidget(plot_obj, html_path)
    }
  }
  
  return(pdf_path)
}

# Check for results files
results_file <- "integrated_data_results.rds"
alt_results_file <- "integrated_analysis_results.rds"

if (!file.exists(results_file) && file.exists(alt_results_file)) {
  results_file <- alt_results_file
  cat("Using results from", results_file, "\n")
}

if (!file.exists(results_file)) {
  # Try to run the main analysis script if available
  main_script <- "Integrated_Analysis.r"
  if (file.exists(main_script)) {
    cat("Results not found. Attempting to run the main analysis script...\n")
    tryCatch({
      source(main_script)
      if (exists("integrated_data") && exists("risk_method")) {
        saveRDS(list(
          integrated_data = integrated_data,
          risk_method = risk_method,
          top_features = if(exists("top_features")) top_features else NULL,
          important_features = if(exists("important_features")) important_features else NULL,
          models = list(
            rsf_model = if(exists("rsf_model")) rsf_model else NULL,
            lasso_cox = if(exists("lasso_cox")) lasso_cox else NULL,
            surv_gbm = if(exists("surv_gbm")) surv_gbm else NULL,
            xgb_model = if(exists("xgb_model")) xgb_model else NULL
          ),
          time_auc = if(exists("time_roc")) list(
            auc_1yr = if(exists("auc_1yr")) auc_1yr else NULL, 
            auc_3yr = if(exists("auc_3yr")) auc_3yr else NULL, 
            auc_5yr = if(exists("auc_5yr")) auc_5yr else NULL
          ) else NULL
        ), file = results_file)
        cat("Analysis completed and results saved to", results_file, "\n")
      } else {
        stop("Main script did not generate required data objects.")
      }
    }, error = function(e) {
      stop("Failed to run analysis script: ", e$message)
    })
  } else {
    stop("No results found and no analysis script available. Please run Integrated_Analysis.r first.")
  }
}

# Load the results
results <- readRDS(results_file)
if ("integrated_data" %in% names(results)) {
  integrated_data <- results$integrated_data
  cat("Loaded integrated data with", nrow(integrated_data), "samples\n")
} else if ("data" %in% names(results)) {
  integrated_data <- results$data
  cat("Loaded integrated data with", nrow(integrated_data), "samples\n")
} else {
  stop("No integrated data found in results file.")
}

# Extract model metrics and performance values
# C-index
c_index <- rcorr.cens(integrated_data$risk_score, 
                     Surv(integrated_data$OS_MONTHS_CLEAN, integrated_data$OS_EVENT_CLEAN))["C Index"]
cat(sprintf("C-index: %.3f\n", c_index))

# Log-rank p-value
fit <- survfit(Surv(OS_MONTHS_CLEAN, OS_EVENT_CLEAN) ~ risk_group, data = integrated_data)
surv_diff <- survdiff(Surv(OS_MONTHS_CLEAN, OS_EVENT_CLEAN) ~ risk_group, data = integrated_data)
p_value <- 1 - pchisq(surv_diff$chisq, df = length(levels(as.factor(integrated_data$risk_group))) - 1)
cat(sprintf("Log-rank p-value: %.4g\n", p_value))

risk_method <- results$risk_method
top_features <- if("top_features" %in% names(results)) results$top_features else NULL
important_features <- if("important_features" %in% names(results)) results$important_features else NULL
time_auc <- if("time_auc" %in% names(results)) results$time_auc else NULL
models <- if("models" %in% names(results)) results$models else NULL

cat("Using risk method:", risk_method, "\n")

# Make sure required survival columns exist
req_cols <- c("OS_MONTHS_CLEAN", "OS_EVENT_CLEAN", "risk_group", "risk_score")
if (!all(req_cols %in% colnames(integrated_data))) {
  stop("Required columns missing from integrated data. Check data format.")
}

# Create publication quality Kaplan-Meier plots 
# ---------------------------------------------- 

# Publication-ready KM plot with risk table and p-value
km_plot_pub <- ggsurvplot(
  fit, 
  data = integrated_data,
  risk.table = TRUE,
  pval = TRUE,
  pval.method = TRUE,
  conf.int = TRUE,
  palette = c("red", "forestgreen"),
  xlab = "Time in Months",
  ylab = "Overall Survival Probability",
  legend.title = "Risk Group",
  legend.labs = c("High Risk", "Low Risk"),
  ggtheme = theme_minimal(base_size = 14),
  title = paste("Kaplan-Meier Survival Curve by Risk Group -", risk_method),
  font.main = c(16, "bold"),
  font.x = c(14, "bold"),
  font.y = c(14, "bold"),
  font.tickslab = c(12),
  risk.table.fontsize = 4,
  tables.height = 0.3,
  risk.table.y.text = FALSE,
  surv.median.line = "hv"
)

# Save the KM plot in both formats
save_dual_format(km_plot_pub$plot, "kaplan_meier_plot", width = 10, height = 8)

# Interactive version
p_surv <- ggplotly(km_plot_pub$plot)
htmlwidgets::saveWidget(p_surv, file.path(output_dir, "interactive_survival_curve.html"))
cat("Created Kaplan-Meier survival curves\n")

# Time-dependent ROC curves
# -------------------------
tryCatch({
  # Create time-dependent ROC curves at 1, 3, 5 years
  time_points <- c(12, 36, 60)
  time_roc <- timeROC(
    T = integrated_data$OS_MONTHS_CLEAN,
    delta = integrated_data$OS_EVENT_CLEAN,
    marker = integrated_data$risk_score,
    cause = 1,
    times = time_points,
    iid = TRUE
  )
  
  # Extract AUC values
  auc_values <- time_roc$AUC
  names(auc_values) <- paste0(time_points/12, "-Year")
  
  # Create ROC curve plot
  roc_plot <- plot_time_roc_curves(time_roc, time_points, title = "Time-dependent ROC Curves")
  save_dual_format(roc_plot, "time_dependent_roc", width = 8, height = 7)
  
  # Print AUC values
  cat("Time-dependent AUC values:\n")
  for (i in 1:length(time_points)) {
    cat(sprintf("  %d-year AUC: %.3f\n", time_points[i]/12, auc_values[i]))
  }
}, error = function(e) {
  cat("Could not generate time-dependent ROC curves:", e$message, "\n")
})

# Confusion Matrix Visualization
# ------------------------------
# Define true classes (using event status as proxy)
true_class <- factor(ifelse(integrated_data$OS_EVENT_CLEAN == 1, "High Risk", "Low Risk"),
                    levels = c("Low Risk", "High Risk"))
pred_class <- factor(integrated_data$risk_group, levels = c("Low Risk", "High Risk"))

# Generate confusion matrix
conf_mat <- table(Predicted = pred_class, Actual = true_class)
conf_mat_stats <- caret::confusionMatrix(pred_class, true_class)

# Extract metrics
accuracy <- conf_mat_stats$overall['Accuracy']
precision <- conf_mat_stats$byClass['Pos Pred Value']
recall <- conf_mat_stats$byClass['Sensitivity']
F1 <- 2 * precision * recall / (precision + recall)
specificity <- conf_mat_stats$byClass['Specificity']

# Create confusion matrix plot
conf_mat_plot <- create_confusion_matrix_plot(conf_mat, accuracy, precision, recall, F1)
save_dual_format(conf_mat_plot, "confusion_matrix_plot", width = 8, height = 7)

cat("Created confusion matrix visualization\n")

# Feature Importance Visualization
# -------------------------------
if (!is.null(top_features) && nrow(top_features) > 0) {
  # Bar plot of feature importance
  importance_plot <- ggplot(head(top_features, 20), 
                          aes(x = reorder(Feature, Importance), 
                              y = Importance, 
                              fill = Importance)) +
    geom_bar(stat = "identity") +
    scale_fill_viridis() +
    coord_flip() +
    theme_minimal(base_size = 14) +
    labs(title = "Top 20 Features by Importance",
         subtitle = paste("Based on", risk_method),
         x = "", 
         y = "Importance Score") +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title.y = element_text(face = "bold", size = 14),
      axis.text.y = element_text(size = 11),
      legend.position = "none"
    )
  
  save_dual_format(importance_plot, "feature_importance_plot", width = 10, height = 8)
  
  # Interactive plot
  p_importance <- plot_ly(
    data = head(top_features, 20),
    x = ~reorder(Feature, Importance),
    y = ~Importance,
    type = "bar",
    marker = list(color = ~Importance, colorscale = "Viridis")
  ) %>%
    layout(
      title = "Top 20 Features by Importance",
      xaxis = list(title = ""),
      yaxis = list(title = "Importance Score"),
      margin = list(l = 120)
    )
  htmlwidgets::saveWidget(p_importance, file.path(output_dir, "feature_importance.html"))
  
  cat("Created feature importance visualization\n")
}

# Risk Score Distribution and Analysis
# -----------------------------------
# Create a density plot of risk scores by event status
risk_density_plot <- ggplot(integrated_data, aes(x = risk_score, fill = factor(OS_EVENT_CLEAN))) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values = c("steelblue", "firebrick"), 
                   labels = c("Censored/Alive", "Event/Deceased"),
                   name = "Patient Status") +
  theme_minimal(base_size = 14) +
  labs(title = "Risk Score Distribution by Survival Status",
       x = "Risk Score",
       y = "Density") +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold", size = 14)
  )

save_dual_format(risk_density_plot, "risk_score_distribution", width = 9, height = 7)

# Risk score vs survival time scatter plot (enhanced)
p_scatter <- plot_ly(
  data = integrated_data,
  x = ~OS_MONTHS_CLEAN,
  y = ~risk_score,
  color = ~factor(OS_EVENT_CLEAN),
  colors = c("steelblue", "firebrick"),
  type = "scatter",
  mode = "markers",
  marker = list(size = 8, opacity = 0.7),
  text = ~paste("ID:", integrated_data[[1]], 
               "<br>Risk Score:", round(risk_score, 3),
               "<br>Survival Time:", round(OS_MONTHS_CLEAN, 1), "months",
               "<br>Status:", ifelse(OS_EVENT_CLEAN == 1, "Deceased", "Alive/Censored"))
) %>%
  layout(
    title = "Risk Score vs. Survival Time",
    xaxis = list(title = "Survival Time (Months)"),
    yaxis = list(title = "Risk Score"),
    legend = list(title = list(text = "Event Status"),
                 x = 0.85, y = 0.95,
                 tracegroupgap = 0,
                 itemsizing = "constant")
  )
htmlwidgets::saveWidget(p_scatter, file.path(output_dir, "risk_vs_survival.html"))

cat("Created risk score analysis plots\n")

# Feature Heatmap
# --------------
# Create a comprehensive heatmap of top features across patients
if (!is.null(top_features) && nrow(top_features) > 0) {
  # Get top N features
  n_features <- min(30, nrow(top_features))
  top_feature_names <- top_features$Feature[1:n_features]
  
  # Ensure all selected features exist in the dataset
  top_feature_names <- top_feature_names[top_feature_names %in% colnames(integrated_data)]
  
  if (length(top_feature_names) > 0) {
    # Extract data for heatmap
    heatmap_data <- as.matrix(integrated_data[, top_feature_names])
    rownames(heatmap_data) <- integrated_data[[1]]  # Set patient IDs as row names
    
    # Z-score normalize the data for better visualization
    heatmap_data_scaled <- t(scale(t(heatmap_data)))
    
    # Create annotation for patients (as row annotation, not column annotation)
    row_ha <- rowAnnotation(
      Risk_Group = integrated_data$risk_group,
      Event = factor(integrated_data$OS_EVENT_CLEAN, labels = c("Censored", "Deceased")),
      col = list(
        Risk_Group = c("High Risk" = "red", "Low Risk" = "forestgreen"),
        Event = c("Deceased" = "black", "Censored" = "grey")
      ),
      annotation_name_side = "top"
    )
    
    # Create the heatmap with row annotations
    heatmap_obj <- Heatmap(
      heatmap_data_scaled,
      name = "Z-Score",
      column_title = "Top Features",
      row_title = "Patients",
      show_row_names = FALSE,  # Too many patients to show names
      show_column_names = TRUE,
      cluster_rows = TRUE,
      cluster_columns = TRUE,
      right_annotation = row_ha,  # Use as row annotation on the right side
      col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
      row_names_gp = gpar(fontsize = 8),
      column_names_gp = gpar(fontsize = 10),
      heatmap_legend_param = list(title = "Z-Score")
    )
    
    # Save the heatmap
    pdf(file.path(output_dir, "feature_heatmap.pdf"), width = 14, height = 10)  # Increased width to accommodate row annotations
    draw(heatmap_obj)
    dev.off()
    
    cat("Created feature heatmap visualization\n")
  }
}

# Model Performance Dashboard
# --------------------------
# Create a dashboard of model performance metrics
performance_summary <- data.frame(
  Metric = c("C-index", "Log-rank p-value", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity"),
  Value = c(c_index, p_value, accuracy, precision, recall, F1, specificity)
)

# Create time AUC table if available
if (!is.null(time_auc)) {
  # Add time-dependent AUCs
  if (!is.null(time_auc$auc_1yr)) {
    performance_summary <- rbind(performance_summary, 
                                data.frame(Metric = "1-year AUC", Value = time_auc$auc_1yr))
  }
  if (!is.null(time_auc$auc_3yr)) {
    performance_summary <- rbind(performance_summary, 
                                data.frame(Metric = "3-year AUC", Value = time_auc$auc_3yr))
  }
  if (!is.null(time_auc$auc_5yr)) {
    performance_summary <- rbind(performance_summary, 
                                data.frame(Metric = "5-year AUC", Value = time_auc$auc_5yr))
  }
}

# Format p-value specially for scientific notation when very small
performance_summary$Formatted_Value <- sapply(1:nrow(performance_summary), function(i) {
  if (performance_summary$Metric[i] == "Log-rank p-value" && performance_summary$Value[i] < 0.001) {
    return(format(performance_summary$Value[i], scientific = TRUE, digits = 3))
  } else {
    return(format(round(performance_summary$Value[i], 4), nsmall = 4))
  }
})

# Create HTML table
performance_table <- kable(performance_summary[, c("Metric", "Formatted_Value")], 
                          col.names = c("Metric", "Value"),
                          format = "html", 
                          caption = paste("Model Performance Metrics -", risk_method)) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
               full_width = FALSE, position = "center")

# Save the table
writeLines(performance_table, file.path(output_dir, "performance_metrics.html"))

# Create a comprehensive patient data table with metrics and risk scores
# -------------------------------------------------------------------
patient_data <- integrated_data %>%
  select(1, OS_MONTHS_CLEAN, OS_EVENT_CLEAN, risk_score, risk_group) %>%
  mutate(
    Survival_Status = ifelse(OS_EVENT_CLEAN == 1, "Deceased", "Alive/Censored"),
    Risk_Score = round(risk_score, 4),
    Survival_Time = round(OS_MONTHS_CLEAN, 2),
    Risk_Group = risk_group
  ) %>%
  arrange(desc(Risk_Score))

names(patient_data)[1] <- "Patient_ID" 

# Create an interactive data table
dt <- datatable(
  patient_data %>% select(Patient_ID, Survival_Time, Survival_Status, Risk_Score, Risk_Group),
  options = list(
    pageLength = 10,
    searchHighlight = TRUE,
    dom = 'Bfrtip',
    buttons = c('copy', 'csv', 'excel', 'pdf'),
    lengthMenu = list(c(10, 25, 50, 100, -1), c('10', '25', '50', '100', 'All'))
  ),
  rownames = FALSE,
  caption = "Patient Risk Scores and Survival Data",
  filter = "top",
  class = "cell-border stripe",
  extensions = c('Buttons', 'FixedHeader'),
  selection = 'none'
) %>%
  formatStyle('Risk_Group',
             backgroundColor = styleEqual(
               c("High Risk", "Low Risk"),
               c("#ffcccc", "#ccffcc")
             )) %>%
  formatStyle('Survival_Status',
             backgroundColor = styleEqual(
               c("Deceased", "Alive/Censored"),
               c("#f8d7da", "#d1ecf1")
             ))

saveWidget(dt, file.path(output_dir, "patient_data_table.html"), selfcontained = TRUE)
cat("Created enhanced patient data table\n")

# Create survival analysis summary in PDF format
# ---------------------------------------------
# Generate a comprehensive report
tryCatch({
  if (requireNamespace("rmarkdown", quietly = TRUE)) {
    # Create a summary report in R Markdown
    report_file <- "survival_analysis_report.Rmd"
    
    # Check if pdflatex is available
    has_latex <- system("which pdflatex >/dev/null 2>&1") == 0
    output_format <- if(has_latex) "pdf_document" else "html_document"
    output_extension <- if(has_latex) "pdf" else "html"
    
    # Inform user about the output format
    if(!has_latex) {
      cat("Note: LaTeX (pdflatex) is not installed. Generating HTML report instead of PDF.\n")
      cat("To install LaTeX for PDF generation, use: sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra\n")
    }
    
    # Write the R Markdown content
    writeLines(c(
      '---',
      'title: "Survival Analysis Results"',
      'author: "Multi-omics Analysis Pipeline"',
      paste0('date: "', format(Sys.Date(), "%B %d, %Y"), '"'),
      'output:',
      paste0('  ', output_format, ':'),
      '    toc: true',
      '    toc_depth: 3',
      '    number_sections: true',
      '    fig_caption: true',
      '---',
      '',
      '```{r setup, include=FALSE}',
      'knitr::opts_chunk$set(echo = FALSE, message = FALSE, warning = FALSE,',
      '                      fig.width = 10, fig.height = 6)',
      'library(knitr)',
      'library(kableExtra)',
      '```',
      '',
      '# Executive Summary',
      '',
      paste('This report summarizes the survival analysis results using the', risk_method, 'model.'),
      paste('The analysis included', nrow(integrated_data), 'patients with valid survival information.'),
      '',
      '## Key Findings',
      '',
      paste('- **C-index:** ', round(c_index, 3)),
      paste('- **Log-rank p-value:** ', format(p_value, digits = 3, scientific = TRUE)),
      paste('- **Number of patients:** ', nrow(integrated_data)),
      paste('- **Event rate:** ', round(100*mean(integrated_data$OS_EVENT_CLEAN), 1), '%', sep=''),
      '',
      '# Survival Analysis Results',
      '',
      '```{r km-plot, fig.cap="Kaplan-Meier survival curves by risk group"}',
      'knitr::include_graphics("figures/kaplan_meier_plot.pdf")',
      '```',
      '',
      '# Feature Importance',
      '',
      '```{r feature-importance, fig.cap="Top features by importance"}',
      'knitr::include_graphics("figures/feature_importance_plot.pdf")',
      '```',
      '',
      '# Model Performance Metrics',
      '',
      '```{r perf-table}',
      'knitr::kable(performance_summary[, c("Metric", "Formatted_Value")],',
      '            col.names = c("Metric", "Value"),',
      '            caption = "Model Performance Metrics") %>%',
      'kable_styling(bootstrap_options = c("striped", "hover"),',
      '             latex_options = c("hold_position", "striped"))',
      '```',
      '',
      '# Risk Score Distribution',
      '',
      '```{r risk-dist, fig.cap="Risk score distribution by survival status"}',
      'knitr::include_graphics("figures/risk_score_distribution.pdf")',
      '```',
      '',
      '# Confusion Matrix',
      '',
      '```{r conf-matrix, fig.cap="Confusion matrix visualization"}',
      'knitr::include_graphics("figures/confusion_matrix_plot.pdf")',
      '```',
      '',
      '# Time-dependent ROC Curves',
      '',
      '```{r roc-curves, fig.cap="Time-dependent ROC curves"}',
      'knitr::include_graphics("figures/time_dependent_roc.pdf")',
      '```',
      '',
      '# Feature Heatmap',
      '',
      '```{r feature-heatmap, fig.cap="Feature heatmap visualization"}',
      'knitr::include_graphics("figures/feature_heatmap.pdf")',
      '```'
    ), report_file)
    
    # Render the report
    output_file <- paste0("survival_analysis_report.", output_extension)
    rmarkdown::render(report_file, output_format = output_format, output_file = file.path(output_dir, output_file))
    cat(paste0("Generated survival analysis report: ", output_dir, "/", output_file, "\n"))
  } else {
    cat("R Markdown package not available. Cannot generate report.\n")
  }
}, error = function(e) {
  cat("Failed to generate survival analysis report:", e$message, "\n")
  cat("Continuing with other outputs...\n")
})

cat("\nAll visualizations created successfully!\n")
cat("Output files saved to the 'figures' directory\n")

# Print a summary of results to console
cat("\n======= MODEL SUMMARY =======\n")
cat("Risk prediction method:", risk_method, "\n")
cat("Number of patients:", nrow(integrated_data), "\n")
cat("Event rate:", round(100*mean(integrated_data$OS_EVENT_CLEAN), 1), "%\n")
cat("C-index:", round(c_index, 3), "\n")
cat("Log-rank p-value:", format(p_value, digits = 3, scientific = TRUE), "\n")
cat("===========================\n")
