"""
Configuration utilities for the RedBus forecasting system.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return get_default_config()

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        'data': {
            'train_path': 'data/raw/train.csv',
            'test_path': 'data/raw/test.csv',
            'transactions_path': 'data/raw/transactions.csv',
            'processed_path': 'data/processed/'
        },
        'models': {
            'xgboost': {
                'n_estimators': 5000,
                'max_depth': 10,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 5000,
                'max_depth': 5,
                'learning_rate': 0.01,
                'random_state': 42
            }
        },
        'ensemble': {
            'weights': {
                'xgb': 0.6,
                'lgb': 0.4
            }
        },
        'validation': {
            'test_size': 0.2,
            'cv_folds': 5,
            'random_state': 42
        }
    }


def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise
