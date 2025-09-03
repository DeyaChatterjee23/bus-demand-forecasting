"""
Enhanced models module with better structure and error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all forecasting models."""

    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    def save(self, filepath: str) -> None:
        """Save the model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'BaseModel':
        """Load the model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        logger.info(f"Model loaded from {filepath}")
        return self


class XGBoostModel(BaseModel):
    """XGBoost regression model."""

    def __init__(self, **kwargs):
        super().__init__()
        self.params = {
            'n_estimators': 5000,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.5,
            'random_state': 42,
            'eval_metric': 'rmse',
            'n_jobs': -1
        }
        self.params.update(kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """Fit XGBoost model."""
        logger.info("Training XGBoost model...")

        self.feature_names = list(X.columns)
        X_processed = self._prepare_features(X)

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_processed, y)

        self.is_fitted = True
        logger.info("XGBoost model training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_processed = self._prepare_features(X)
        predictions = self.model.predict(X_processed)
        return np.maximum(predictions, 0)  # Ensure non-negative predictions

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for XGBoost (no scaling needed)."""
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        # Select and order features consistently
        available_features = [col for col in self.feature_names if col in X.columns]
        return X[available_features].fillna(0)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df


class LightGBMModel(BaseModel):
    """LightGBM regression model."""

    def __init__(self, **kwargs):
        super().__init__()
        self.params = {
            'n_estimators': 5000,
            'max_depth': 5,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        self.params.update(kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LightGBMModel':
        """Fit LightGBM model."""
        logger.info("Training LightGBM model...")

        self.feature_names = list(X.columns)
        X_processed = self._prepare_features(X)

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_processed, y)

        self.is_fitted = True
        logger.info("LightGBM model training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_processed = self._prepare_features(X)
        predictions = self.model.predict(X_processed)
        return np.maximum(predictions, 0)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for LightGBM."""
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        available_features = [col for col in self.feature_names if col in X.columns]
        return X[available_features].fillna(0)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df


class EnsembleModel:
    """Ensemble of multiple models with weighted predictions."""

    def __init__(self, models: Dict[str, BaseModel], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
        self.is_fitted = False

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """Fit all models in the ensemble."""
        logger.info("Training ensemble models...")

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X, y)

        self.is_fitted = True
        logger.info("Ensemble training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        predictions = np.zeros(len(X))

        for name, model in self.models.items():
            model_pred = model.predict(X)
            predictions += self.weights[name] * model_pred

        return predictions

    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        return {name: model.predict(X) for name, model in self.models.items()}

    def save(self, dirpath: str) -> None:
        """Save all models in the ensemble."""
        dir_path = Path(dirpath)
        dir_path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model.save(dir_path / f"{name}_model.joblib")

        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'model_names': list(self.models.keys()),
            'is_fitted': self.is_fitted
        }
        joblib.dump(metadata, dir_path / "ensemble_metadata.joblib")

        logger.info(f"Ensemble saved to {dirpath}")


class ModelEvaluator:
    """Model evaluation utilities."""

    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds

    def evaluate_model(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model using time series cross-validation."""
        logger.info("Evaluating model performance...")

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        # Cross-validation scores
        cv_scores = cross_val_score(
            model.model, X, y, cv=tscv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )

        rmse_scores = np.sqrt(-cv_scores)

        # Final model performance
        model.fit(X, y)
        y_pred = model.predict(X)

        metrics = {
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'train_mae': mean_absolute_error(y, y_pred),
            'train_r2': r2_score(y, y_pred)
        }

        logger.info(f"Model evaluation completed: RMSE={metrics['cv_rmse_mean']:.4f}±{metrics['cv_rmse_std']:.4f}")

        return metrics

    def compare_models(self, models: Dict[str, BaseModel], 
                      X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare multiple models."""
        results = {}

        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            results[name] = self.evaluate_model(model, X, y)

        return pd.DataFrame(results).T


class ModelManager:
    """Manage model training, evaluation, and deployment."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model manager with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models_config = self.config.get('models', {})
        self.ensemble_config = self.config.get('ensemble', {})
        self.evaluator = ModelEvaluator(cv_folds=self.config.get('validation', {}).get('cv_folds', 5))

    def create_models(self) -> Dict[str, BaseModel]:
        """Create models based on configuration."""
        models = {}

        if 'xgboost' in self.models_config:
            models['xgb'] = XGBoostModel(**self.models_config['xgboost'])

        if 'lightgbm' in self.models_config:
            models['lgb'] = LightGBMModel(**self.models_config['lightgbm'])

        return models

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        """Train ensemble model."""
        models = self.create_models()
        weights = self.ensemble_config.get('weights', None)

        ensemble = EnsembleModel(models, weights)
        ensemble.fit(X, y)

        return ensemble

    def evaluate_and_select_best(self, X: pd.DataFrame, y: pd.Series) -> Tuple[BaseModel, Dict[str, Any]]:
        """Evaluate models and select the best one."""
        models = self.create_models()

        # Evaluate all models
        comparison_df = self.evaluator.compare_models(models, X, y)
        logger.info("Model comparison:")
        logger.info(comparison_df)

        # Select best model based on CV RMSE
        best_model_name = comparison_df['cv_rmse_mean'].idxmin()
        best_model = models[best_model_name]

        logger.info(f"Best model selected: {best_model_name}")

        return best_model, comparison_df.to_dict()
