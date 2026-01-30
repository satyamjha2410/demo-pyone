"""
Machine Learning Regression Pipeline
====================================
A production-ready script for training and evaluating regression models
using Linear Regression and XGBoost.

Author: Senior Developer
Date: 2025
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regression_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Container for model evaluation results"""
    model_name: str
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    model: Any


class DataLoader:
    """Handles data loading and basic validation"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file"""
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict(),
            'describe': self.data.describe()
        }


class DataPreprocessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self, target_column: str = 'Price'):
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable
        
        Args:
            df: Raw dataframe
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("Preparing features and target variable")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Drop non-numeric columns (like Address)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.target_column not in numeric_cols:
            raise ValueError(f"Target column '{self.target_column}' not found in numeric columns")
        
        # Separate features and target
        feature_cols = [col for col in numeric_cols if col != self.target_column]
        self.feature_columns = feature_cols
        
        X = data[feature_cols]
        y = data[self.target_column]
        
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Target: {self.target_column}")
        logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler"""
        logger.info("Scaling features")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def check_data_quality(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Perform data quality checks"""
        logger.info("Performing data quality checks")
        
        # Check for missing values
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        
        if missing_X > 0:
            logger.warning(f"Found {missing_X} missing values in features")
        if missing_y > 0:
            logger.warning(f"Found {missing_y} missing values in target")
        
        # Check for infinite values
        inf_X = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        inf_y = np.isinf(y).sum()
        
        if inf_X > 0:
            logger.warning(f"Found {inf_X} infinite values in features")
        if inf_y > 0:
            logger.warning(f"Found {inf_y} infinite values in target")
        
        logger.info("Data quality checks completed")


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_linear_regression(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray,
        y_train: pd.Series, 
        y_test: pd.Series
    ) -> ModelResults:
        """Train Linear Regression model"""
        logger.info("Training Linear Regression model")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_test_pred)
        
        results = ModelResults(
            model_name="Linear Regression",
            **metrics,
            model=model
        )
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = results
        
        logger.info(f"Linear Regression - R²: {results.r2:.4f}, RMSE: {results.rmse:.2f}")
        
        return results
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        **xgb_params
    ) -> ModelResults:
        """Train XGBoost Regression model"""
        logger.info("Training XGBoost Regression model")
        
        # Default XGBoost parameters (can be overridden)
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(xgb_params)
        
        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_test_pred)
        
        results = ModelResults(
            model_name="XGBoost Regression",
            **metrics,
            model=model
        )
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        logger.info(f"XGBoost Regression - R²: {results.r2:.4f}, RMSE: {results.rmse:.2f}")
        
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Avoid division by zero in MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': results.model_name,
                'R² Score': results.r2,
                'RMSE': results.rmse,
                'MAE': results.mae,
                'MAPE (%)': results.mape
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        return comparison_df


class Visualizer:
    """Handles visualization of results"""
    
    @staticmethod
    def plot_predictions(
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> None:
        """Plot predicted vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predicted vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_residuals(
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> None:
        """Plot residuals"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name} - Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_feature_importance(
        model: Any,
        feature_names: list,
        model_name: str,
        save_path: str = None
    ) -> None:
        """Plot feature importance (for XGBoost)"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'{model_name} - Feature Importance')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices], 
                      rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
        else:
            logger.warning(f"{model_name} does not support feature importance visualization")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("Starting Machine Learning Regression Pipeline")
    logger.info("=" * 60)
    
    try:
        # Configuration
        DATA_PATH = "Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv"
        TARGET_COLUMN = "Price"
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        
        # Initialize components
        loader = DataLoader(DATA_PATH)
        preprocessor = DataPreprocessor(target_column=TARGET_COLUMN)
        trainer = ModelTrainer()
        visualizer = Visualizer()
        
        # Load data
        data = loader.load_data()
        data_info = loader.get_data_info()
        logger.info(f"Dataset info: {data_info['shape']}")
        
        # Prepare features
        X, y = preprocessor.prepare_features(data)
        
        # Data quality checks
        preprocessor.check_data_quality(X, y)
        
        # Handle missing values if any
        if X.isnull().sum().sum() > 0:
            logger.info("Filling missing values with median")
            X = X.fillna(X.median())
        
        if y.isnull().sum() > 0:
            logger.info("Filling missing target values with median")
            y = y.fillna(y.median())
        
        # Split data
        logger.info(f"Splitting data: {TEST_SIZE*100}% test, {(1-TEST_SIZE)*100}% train")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Train Linear Regression
        logger.info("\n" + "-" * 60)
        lr_results = trainer.train_linear_regression(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Train XGBoost
        logger.info("\n" + "-" * 60)
        xgb_results = trainer.train_xgboost(
            X_train_scaled, X_test_scaled, y_train, y_test,
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05
        )
        
        # Compare models
        logger.info("\n" + "=" * 60)
        logger.info("Model Comparison")
        logger.info("=" * 60)
        comparison = trainer.compare_models()
        print("\n" + comparison.to_string(index=False) + "\n")
        
        # Visualizations
        logger.info("Generating visualizations...")
        
        # Linear Regression plots
        visualizer.plot_predictions(
            y_test, trainer.models['linear_regression'].predict(X_test_scaled),
            "Linear Regression",
            save_path="linear_regression_predictions.png"
        )
        
        visualizer.plot_residuals(
            y_test, trainer.models['linear_regression'].predict(X_test_scaled),
            "Linear Regression",
            save_path="linear_regression_residuals.png"
        )
        
        # XGBoost plots
        visualizer.plot_predictions(
            y_test, trainer.models['xgboost'].predict(X_test_scaled),
            "XGBoost Regression",
            save_path="xgboost_predictions.png"
        )
        
        visualizer.plot_residuals(
            y_test, trainer.models['xgboost'].predict(X_test_scaled),
            "XGBoost Regression",
            save_path="xgboost_residuals.png"
        )
        
        # Feature importance for XGBoost
        visualizer.plot_feature_importance(
            trainer.models['xgboost'],
            preprocessor.feature_columns,
            "XGBoost Regression",
            save_path="xgboost_feature_importance.png"
        )
        
        # Print detailed results
        logger.info("\n" + "=" * 60)
        logger.info("Detailed Results")
        logger.info("=" * 60)
        
        for model_name, results in trainer.results.items():
            logger.info(f"\n{results.model_name}:")
            logger.info(f"  R² Score: {results.r2:.4f}")
            logger.info(f"  RMSE: {results.rmse:.2f}")
            logger.info(f"  MAE: {results.mae:.2f}")
            logger.info(f"  MAPE: {results.mape:.2f}%")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

