# Machine Learning Regression Pipeline

A production-ready Python script for training and evaluating regression models using Linear Regression and XGBoost.

## Features

- **Data Loading & Validation**: Robust data loading with error handling
- **Data Preprocessing**: Automatic feature scaling and data quality checks
- **Multiple Models**: 
  - Linear Regression (scikit-learn)
  - XGBoost Regression
- **Comprehensive Evaluation**: 
  - R² Score
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
- **Visualizations**:
  - Predicted vs Actual scatter plots
  - Residual plots
  - Feature importance (XGBoost)
- **Logging**: Comprehensive logging to file and console
- **Production-Ready**: Modular design, error handling, type hints

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage

```bash
python ml_regression.py
```

The script will:
1. Load the USA Housing dataset
2. Preprocess and scale features
3. Train Linear Regression model
4. Train XGBoost Regression model
5. Compare model performance
6. Generate visualizations
7. Save plots and log file

## Output

The script generates:
- **Console output**: Model metrics and comparison table
- **Log file**: `regression_pipeline.log` with detailed execution log
- **Plots**:
  - `linear_regression_predictions.png`
  - `linear_regression_residuals.png`
  - `xgboost_predictions.png`
  - `xgboost_residuals.png`
  - `xgboost_feature_importance.png`

## Code Structure

```
ml_regression.py
├── DataLoader          # Handles data loading
├── DataPreprocessor    # Feature engineering & scaling
├── ModelTrainer       # Model training & evaluation
├── Visualizer         # Plotting functions
└── main()             # Main execution pipeline
```

## Customization

### Using Your Own Dataset

1. Update `DATA_PATH` in the `main()` function
2. Update `TARGET_COLUMN` to match your target variable name
3. Adjust feature columns if needed

### Adjusting Model Parameters

**XGBoost parameters** can be customized in the `train_xgboost()` call:

```python
xgb_results = trainer.train_xgboost(
    X_train_scaled, X_test_scaled, y_train, y_test,
    n_estimators=200,      # Number of trees
    max_depth=8,           # Maximum tree depth
    learning_rate=0.01,    # Learning rate
    subsample=0.9,         # Row sampling ratio
    colsample_bytree=0.9   # Column sampling ratio
)
```

## Example Output

```
Model Comparison
================
              Model    R² Score      RMSE         MAE    MAPE (%)
XGBoost Regression     0.9234    123456.78   98765.43      5.23
Linear Regression      0.9178    134567.89  102345.67      5.67
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0

## Notes

- The script uses the USA Housing dataset by default
- Features are automatically scaled using StandardScaler
- Missing values are handled with median imputation
- All models are evaluated on the same test set for fair comparison

