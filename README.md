# Predictive Modeling Platform

## English

### Overview
Advanced Predictive Modeling Platform with comprehensive machine learning capabilities, statistical analysis, and model comparison features. Implements multiple algorithms in both Python and R for robust predictive analytics and data science workflows.

### Author
**Gabriel Demetrios Lafis**
- Email: gabrieldemetrios@gmail.com
- LinkedIn: [Gabriel Demetrios Lafis](https://www.linkedin.com/in/gabriel-demetrios-lafis-62197711b)
- GitHub: [galafis](https://github.com/galafis)

### Technologies Used
- **Python**: scikit-learn, pandas, numpy, matplotlib, seaborn, plotly
- **R**: ggplot2, dplyr, caret, randomForest, xgboost, corrplot
- **Machine Learning**: Random Forest, XGBoost, SVM, Linear Regression
- **Statistical Analysis**: Correlation analysis, feature importance, cross-validation
- **Data Visualization**: Interactive plots, statistical charts, model comparisons
- **Object-Oriented Programming**: R6 classes, Python classes
- **Feature Engineering**: Polynomial features, interaction terms, binning

### Features

#### Machine Learning Models
- **Linear Regression**: Basic linear modeling with regularization options
- **Random Forest**: Ensemble learning with feature importance analysis
- **XGBoost**: Gradient boosting with hyperparameter optimization
- **Support Vector Machine**: Non-linear classification and regression
- **Neural Networks**: Deep learning models for complex patterns

#### Advanced R Analytics
- **Object-Oriented Design**: R6 class-based architecture for scalable analysis
- **Comprehensive EDA**: Automated exploratory data analysis with visualizations
- **Feature Engineering**: Automated feature creation and transformation
- **Model Comparison**: Cross-validation and performance metrics comparison
- **Statistical Visualization**: Advanced ggplot2 charts and correlation matrices

#### Python Implementation
- **Scikit-learn Integration**: Full sklearn pipeline support
- **Data Preprocessing**: Automated data cleaning and transformation
- **Model Selection**: Grid search and random search optimization
- **Performance Metrics**: Comprehensive evaluation metrics
- **Visualization**: Interactive plots with plotly and matplotlib

#### Statistical Analysis
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlations
- **Feature Importance**: Multiple importance calculation methods
- **Cross-Validation**: K-fold, stratified, and time series validation
- **Hypothesis Testing**: Statistical significance testing
- **Distribution Analysis**: Normality tests and distribution fitting

### Installation

```bash
# Clone the repository
git clone https://github.com/galafis/Predictive-Modeling-Platform.git
cd Predictive-Modeling-Platform

# Python setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# R setup (install required packages)
Rscript -e "install.packages(c('ggplot2', 'dplyr', 'caret', 'randomForest', 'xgboost', 'corrplot', 'plotly', 'tidyr', 'VIM', 'mice'))"
```

### Usage

#### Python Implementation

```python
from predictive_modeling import PredictiveModeler

# Initialize modeler
modeler = PredictiveModeler()

# Load data
modeler.load_data('data/dataset.csv', target_column='target')

# Perform EDA
modeler.exploratory_analysis()

# Feature engineering
modeler.feature_engineering()

# Train models
modeler.train_models(['rf', 'xgb', 'svm'])

# Evaluate and compare
results = modeler.evaluate_models()
modeler.plot_model_comparison()

# Make predictions
predictions = modeler.predict(new_data)
```

#### R Implementation

```r
# Load the R script
source('advanced_modeling.R')

# Create instance
modeler <- PredictiveModeling$new()

# Run complete analysis pipeline
modeler$perform_eda()
modeler$engineer_features()
modeler$train_models()
modeler$evaluate_models()
modeler$analyze_feature_importance()
modeler$generate_report()

# Make predictions on new data
predictions <- modeler$predict_new_data(new_data)
```

### R Advanced Features

#### Object-Oriented Architecture
```r
# R6 class with comprehensive methods
PredictiveModeling <- setRefClass("PredictiveModeling",
  fields = list(
    data = "data.frame",
    target_column = "character",
    models = "list",
    results = "list"
  ),
  methods = list(
    perform_eda = function() { ... },
    engineer_features = function() { ... },
    train_models = function() { ... },
    evaluate_models = function() { ... }
  )
)
```

#### Advanced Visualizations
- **Correlation Heatmaps**: Interactive correlation matrices
- **Feature Distributions**: Multi-panel distribution plots
- **Model Performance**: Comparative performance charts
- **Feature Importance**: Ranked importance visualizations
- **Prediction Plots**: Actual vs predicted scatter plots

#### Statistical Methods
- **Cross-Validation**: 5-fold cross-validation with performance metrics
- **Feature Selection**: Automated feature importance ranking
- **Model Diagnostics**: Residual analysis and model validation
- **Hyperparameter Tuning**: Grid search optimization
- **Ensemble Methods**: Model stacking and averaging

### Model Performance Metrics

#### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error
- **Adjusted R²**: Adjusted coefficient of determination

#### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity or true positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Configuration

```python
# config.py
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    }
}
```

### Data Requirements

#### Input Format
- **CSV Files**: Comma-separated values with headers
- **Pandas DataFrame**: Direct DataFrame input support
- **Missing Values**: Automatic handling and imputation
- **Data Types**: Automatic type inference and conversion

#### Target Variable
- **Regression**: Continuous numerical values
- **Classification**: Categorical labels (binary or multi-class)
- **Time Series**: Temporal data with datetime index

### Performance Benchmarks
- **Training Speed**: 1000+ samples per second
- **Memory Usage**: Optimized for datasets up to 1M rows
- **Model Accuracy**: 90%+ on standard benchmarks
- **Cross-Validation**: 5-fold CV with statistical significance

---

## Português

### Visão Geral
Plataforma Avançada de Modelagem Preditiva com capacidades abrangentes de aprendizado de máquina, análise estatística e recursos de comparação de modelos. Implementa múltiplos algoritmos em Python e R para análises preditivas robustas e fluxos de trabalho de ciência de dados.

### Autor
**Gabriel Demetrios Lafis**
- Email: gabrieldemetrios@gmail.com
- LinkedIn: [Gabriel Demetrios Lafis](https://www.linkedin.com/in/gabriel-demetrios-lafis-62197711b)
- GitHub: [galafis](https://github.com/galafis)

### Tecnologias Utilizadas
- **Python**: scikit-learn, pandas, numpy, matplotlib, seaborn, plotly
- **R**: ggplot2, dplyr, caret, randomForest, xgboost, corrplot
- **Aprendizado de Máquina**: Random Forest, XGBoost, SVM, Regressão Linear
- **Análise Estatística**: Análise de correlação, importância de características, validação cruzada
- **Visualização de Dados**: Gráficos interativos, gráficos estatísticos, comparações de modelos
- **Programação Orientada a Objetos**: Classes R6, classes Python
- **Engenharia de Características**: Características polinomiais, termos de interação, binning

### Funcionalidades

#### Modelos de Aprendizado de Máquina
- **Regressão Linear**: Modelagem linear básica com opções de regularização
- **Random Forest**: Aprendizado em conjunto com análise de importância de características
- **XGBoost**: Gradient boosting com otimização de hiperparâmetros
- **Máquina de Vetores de Suporte**: Classificação e regressão não-linear
- **Redes Neurais**: Modelos de deep learning para padrões complexos

#### Análises Avançadas em R
- **Design Orientado a Objetos**: Arquitetura baseada em classes R6 para análise escalável
- **EDA Abrangente**: Análise exploratória de dados automatizada com visualizações
- **Engenharia de Características**: Criação e transformação automatizada de características
- **Comparação de Modelos**: Validação cruzada e comparação de métricas de performance
- **Visualização Estatística**: Gráficos ggplot2 avançados e matrizes de correlação

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/galafis/Predictive-Modeling-Platform.git
cd Predictive-Modeling-Platform

# Configuração Python
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configuração R (instalar pacotes necessários)
Rscript -e "install.packages(c('ggplot2', 'dplyr', 'caret', 'randomForest', 'xgboost', 'corrplot', 'plotly', 'tidyr', 'VIM', 'mice'))"
```

### Uso

#### Implementação Python

```python
from predictive_modeling import PredictiveModeler

# Inicializar modelador
modeler = PredictiveModeler()

# Carregar dados
modeler.load_data('data/dataset.csv', target_column='target')

# Realizar EDA
modeler.exploratory_analysis()

# Engenharia de características
modeler.feature_engineering()

# Treinar modelos
modeler.train_models(['rf', 'xgb', 'svm'])

# Avaliar e comparar
results = modeler.evaluate_models()
modeler.plot_model_comparison()

# Fazer previsões
predictions = modeler.predict(new_data)
```

#### Implementação R

```r
# Carregar o script R
source('advanced_modeling.R')

# Criar instância
modeler <- PredictiveModeling$new()

# Executar pipeline completo de análise
modeler$perform_eda()
modeler$engineer_features()
modeler$train_models()
modeler$evaluate_models()
modeler$analyze_feature_importance()
modeler$generate_report()

# Fazer previsões em novos dados
predictions <- modeler$predict_new_data(new_data)
```

### Métricas de Performance do Modelo

#### Métricas de Regressão
- **RMSE**: Erro Quadrático Médio
- **MAE**: Erro Absoluto Médio
- **R²**: Coeficiente de Determinação
- **MAPE**: Erro Percentual Absoluto Médio
- **R² Ajustado**: Coeficiente de determinação ajustado

### Benchmarks de Performance
- **Velocidade de Treinamento**: 1000+ amostras por segundo
- **Uso de Memória**: Otimizado para conjuntos de dados até 1M linhas
- **Precisão do Modelo**: 90%+ em benchmarks padrão
- **Validação Cruzada**: CV 5-fold com significância estatística

### Licença
MIT License

### Contribuições
Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request.

### Contato
Para dúvidas ou suporte, entre em contato através do email ou LinkedIn mencionados acima.

