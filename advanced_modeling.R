# Advanced Predictive Modeling Platform - R Analysis
# Author: Gabriel Demetrios Lafis
# Advanced statistical modeling and visualization

# Load required libraries
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(corrplot)
library(plotly)
library(tidyr)
library(VIM)
library(mice)

# Set theme for plots
theme_set(theme_minimal())

# Advanced Predictive Modeling Class
PredictiveModeling <- setRefClass("PredictiveModeling",
  fields = list(
    data = "data.frame",
    target_column = "character",
    models = "list",
    results = "list"
  ),
  
  methods = list(
    # Initialize with data
    initialize = function(data_path = NULL, target_col = "target") {
      if (!is.null(data_path)) {
        data <<- read.csv(data_path)
      } else {
        # Generate sample data for demonstration
        set.seed(42)
        n <- 1000
        data <<- data.frame(
          feature1 = rnorm(n, 50, 15),
          feature2 = runif(n, 0, 100),
          feature3 = rpois(n, 5),
          feature4 = rbinom(n, 1, 0.3),
          feature5 = rexp(n, 0.1)
        )
        data$target <<- with(data, 
          0.3 * feature1 + 0.2 * feature2 + 0.1 * feature3 + 
          10 * feature4 + 0.05 * feature5 + rnorm(n, 0, 5)
        )
      }
      target_column <<- target_col
      models <<- list()
      results <<- list()
    },
    
    # Exploratory Data Analysis
    perform_eda = function() {
      cat("=== Exploratory Data Analysis ===\n")
      
      # Data summary
      print(summary(data))
      
      # Missing values analysis
      missing_plot <- VIM::aggr(data, col = c('navyblue', 'red'), 
                               numbers = TRUE, sortVars = TRUE)
      
      # Correlation matrix
      numeric_data <- data[sapply(data, is.numeric)]
      cor_matrix <- cor(numeric_data, use = "complete.obs")
      
      # Correlation plot
      corrplot(cor_matrix, method = "circle", type = "upper", 
               order = "hclust", tl.cex = 0.8, tl.col = "black")
      
      # Distribution plots
      data_long <- numeric_data %>%
        gather(key = "variable", value = "value")
      
      dist_plot <- ggplot(data_long, aes(x = value)) +
        geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
        facet_wrap(~variable, scales = "free") +
        labs(title = "Feature Distributions",
             x = "Value", y = "Frequency") +
        theme_minimal()
      
      print(dist_plot)
      
      return(list(
        summary = summary(data),
        correlation = cor_matrix,
        missing_values = sum(is.na(data))
      ))
    },
    
    # Feature Engineering
    engineer_features = function() {
      cat("=== Feature Engineering ===\n")
      
      # Create interaction features
      data$feature1_feature2 <<- data$feature1 * data$feature2
      data$feature1_squared <<- data$feature1^2
      data$feature2_log <<- log(data$feature2 + 1)
      
      # Binning continuous variables
      data$feature1_binned <<- cut(data$feature1, 
                                  breaks = quantile(data$feature1, probs = 0:4/4),
                                  labels = c("Low", "Medium-Low", "Medium-High", "High"),
                                  include.lowest = TRUE)
      
      # Polynomial features
      data$feature2_poly2 <<- data$feature2^2
      data$feature2_poly3 <<- data$feature2^3
      
      cat("New features created: feature1_feature2, feature1_squared, feature2_log, feature1_binned, feature2_poly2, feature2_poly3\n")
    },
    
    # Train multiple models
    train_models = function() {
      cat("=== Training Multiple Models ===\n")
      
      # Prepare data
      feature_cols <- setdiff(names(data), target_column)
      X <- data[, feature_cols]
      y <- data[[target_column]]
      
      # Split data
      set.seed(42)
      train_index <- createDataPartition(y, p = 0.8, list = FALSE)
      X_train <- X[train_index, ]
      X_test <- X[-train_index, ]
      y_train <- y[train_index]
      y_test <- y[-train_index]
      
      # Train Control
      ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
      
      # Linear Regression
      cat("Training Linear Regression...\n")
      models$lm <<- train(X_train, y_train, method = "lm", trControl = ctrl)
      
      # Random Forest
      cat("Training Random Forest...\n")
      models$rf <<- train(X_train, y_train, method = "rf", 
                         trControl = ctrl, ntree = 100)
      
      # XGBoost
      cat("Training XGBoost...\n")
      # Convert categorical variables to numeric for XGBoost
      X_train_xgb <- X_train
      X_test_xgb <- X_test
      
      # Handle categorical variables
      for (col in names(X_train_xgb)) {
        if (is.factor(X_train_xgb[[col]])) {
          X_train_xgb[[col]] <- as.numeric(X_train_xgb[[col]])
          X_test_xgb[[col]] <- as.numeric(X_test_xgb[[col]])
        }
      }
      
      models$xgb <<- train(X_train_xgb, y_train, method = "xgbTree",
                          trControl = ctrl, verbosity = 0)
      
      # Support Vector Machine
      cat("Training SVM...\n")
      models$svm <<- train(X_train, y_train, method = "svmRadial", trControl = ctrl)
      
      # Store test data for evaluation
      results$X_test <<- X_test
      results$y_test <<- y_test
      results$X_test_xgb <<- X_test_xgb
      
      cat("All models trained successfully!\n")
    },
    
    # Evaluate models
    evaluate_models = function() {
      cat("=== Model Evaluation ===\n")
      
      model_performance <- data.frame(
        Model = character(),
        RMSE = numeric(),
        MAE = numeric(),
        R_squared = numeric(),
        stringsAsFactors = FALSE
      )
      
      for (model_name in names(models)) {
        cat(paste("Evaluating", model_name, "...\n"))
        
        # Make predictions
        if (model_name == "xgb") {
          predictions <- predict(models[[model_name]], results$X_test_xgb)
        } else {
          predictions <- predict(models[[model_name]], results$X_test)
        }
        
        # Calculate metrics
        rmse <- sqrt(mean((predictions - results$y_test)^2))
        mae <- mean(abs(predictions - results$y_test))
        r_squared <- cor(predictions, results$y_test)^2
        
        # Store results
        model_performance <- rbind(model_performance, 
                                 data.frame(Model = model_name, 
                                          RMSE = rmse, 
                                          MAE = mae, 
                                          R_squared = r_squared))
        
        # Store predictions for plotting
        results[[paste0(model_name, "_predictions")]] <<- predictions
      }
      
      # Sort by R-squared
      model_performance <- model_performance[order(-model_performance$R_squared), ]
      print(model_performance)
      
      # Create performance visualization
      perf_plot <- model_performance %>%
        gather(key = "Metric", value = "Value", -Model) %>%
        ggplot(aes(x = Model, y = Value, fill = Model)) +
        geom_bar(stat = "identity") +
        facet_wrap(~Metric, scales = "free_y") +
        labs(title = "Model Performance Comparison",
             x = "Model", y = "Value") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      print(perf_plot)
      
      results$performance <<- model_performance
      return(model_performance)
    },
    
    # Feature importance analysis
    analyze_feature_importance = function() {
      cat("=== Feature Importance Analysis ===\n")
      
      # Random Forest feature importance
      if ("rf" %in% names(models)) {
        rf_importance <- varImp(models$rf)
        
        importance_plot <- ggplot(rf_importance$importance, 
                                aes(x = reorder(rownames(rf_importance$importance), Overall), 
                                    y = Overall)) +
          geom_bar(stat = "identity", fill = "steelblue") +
          coord_flip() +
          labs(title = "Random Forest - Feature Importance",
               x = "Features", y = "Importance") +
          theme_minimal()
        
        print(importance_plot)
      }
      
      # XGBoost feature importance
      if ("xgb" %in% names(models)) {
        xgb_importance <- varImp(models$xgb)
        
        xgb_plot <- ggplot(xgb_importance$importance, 
                          aes(x = reorder(rownames(xgb_importance$importance), Overall), 
                              y = Overall)) +
          geom_bar(stat = "identity", fill = "darkgreen") +
          coord_flip() +
          labs(title = "XGBoost - Feature Importance",
               x = "Features", y = "Importance") +
          theme_minimal()
        
        print(xgb_plot)
      }
    },
    
    # Generate predictions for new data
    predict_new_data = function(new_data) {
      if (length(models) == 0) {
        stop("No models trained yet. Please run train_models() first.")
      }
      
      predictions <- list()
      
      for (model_name in names(models)) {
        if (model_name == "xgb") {
          # Handle categorical variables for XGBoost
          new_data_xgb <- new_data
          for (col in names(new_data_xgb)) {
            if (is.factor(new_data_xgb[[col]])) {
              new_data_xgb[[col]] <- as.numeric(new_data_xgb[[col]])
            }
          }
          predictions[[model_name]] <- predict(models[[model_name]], new_data_xgb)
        } else {
          predictions[[model_name]] <- predict(models[[model_name]], new_data)
        }
      }
      
      return(predictions)
    },
    
    # Generate comprehensive report
    generate_report = function() {
      cat("=== Comprehensive Modeling Report ===\n")
      cat("Author: Gabriel Demetrios Lafis\n")
      cat("Date:", Sys.Date(), "\n\n")
      
      cat("Dataset Summary:\n")
      cat("- Observations:", nrow(data), "\n")
      cat("- Features:", ncol(data) - 1, "\n")
      cat("- Target Variable:", target_column, "\n\n")
      
      if (length(results) > 0 && "performance" %in% names(results)) {
        cat("Best Performing Model:\n")
        best_model <- results$performance[1, ]
        cat("- Model:", best_model$Model, "\n")
        cat("- R-squared:", round(best_model$R_squared, 4), "\n")
        cat("- RMSE:", round(best_model$RMSE, 4), "\n")
        cat("- MAE:", round(best_model$MAE, 4), "\n")
      }
      
      cat("\nAnalysis completed successfully!\n")
    }
  )
)

# Example usage and demonstration
cat("=== Advanced Predictive Modeling Platform ===\n")
cat("Author: Gabriel Demetrios Lafis\n")
cat("Demonstrating advanced R capabilities for machine learning\n\n")

# Create instance
modeler <- PredictiveModeling$new()

# Run complete analysis pipeline
cat("Running complete analysis pipeline...\n")
modeler$perform_eda()
modeler$engineer_features()
modeler$train_models()
modeler$evaluate_models()
modeler$analyze_feature_importance()
modeler$generate_report()

cat("\n=== Advanced R Analysis Complete ===\n")
cat("This demonstrates:\n")
cat("- Object-oriented programming in R\n")
cat("- Advanced statistical modeling\n")
cat("- Machine learning with multiple algorithms\n")
cat("- Data visualization with ggplot2\n")
cat("- Feature engineering and selection\n")
cat("- Model evaluation and comparison\n")

