"""
Data loading and validation module for RedBus Demand Forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from pydantic import BaseModel, validator
import yaml

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Configuration for data loading."""
    train_path: str
    test_path: str
    transactions_path: str
    processed_path: str

    @validator('*')
    def paths_must_exist(cls, v):
        if not Path(v).exists() and 'processed' not in v:
            logger.warning(f"Path does not exist: {v}")
        return v


class DataLoader:
    """Enhanced data loader with validation and error handling."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration."""
        self.config = self._load_config(config_path)
        self.data_config = DataConfig(**self.config['data'])

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, test, and transactions data with validation.

        Returns:
            Tuple of (train_df, test_df, transactions_df)
        """
        logger.info("Loading datasets...")

        try:
            # Load datasets
            train_df = pd.read_csv(self.data_config.train_path)
            test_df = pd.read_csv(self.data_config.test_path)
            transactions_df = pd.read_csv(self.data_config.transactions_path)

            # Convert date columns
            date_cols = ['doj', 'doi']
            for df in [train_df, test_df, transactions_df]:
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            # Validate data
            self._validate_data(train_df, test_df, transactions_df)

            logger.info(f"Data loaded successfully:")
            logger.info(f"  Train: {train_df.shape}")
            logger.info(f"  Test: {test_df.shape}")
            logger.info(f"  Transactions: {transactions_df.shape}")

            return train_df, test_df, transactions_df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _validate_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      transactions_df: pd.DataFrame) -> None:
        """Validate loaded data for consistency and quality."""

        # Check required columns
        required_train_cols = ['srcid', 'destid', 'doj', 'final_seatcount']
        required_test_cols = ['srcid', 'destid', 'doj']
        required_trans_cols = ['srcid', 'destid', 'doj', 'doi', 'cumsum_seatcount']

        for df, cols, name in [
            (train_df, required_train_cols, 'train'),
            (test_df, required_test_cols, 'test'),
            (transactions_df, required_trans_cols, 'transactions')
        ]:
            missing_cols = set(cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in {name}: {missing_cols}")

        # Check for missing values in critical columns
        critical_cols = ['srcid', 'destid', 'doj']
        for df, name in [(train_df, 'train'), (test_df, 'test')]:
            for col in critical_cols:
                if df[col].isna().any():
                    logger.warning(f"Missing values in {name}.{col}: {df[col].isna().sum()}")

        # Validate date ranges
        train_date_range = (train_df['doj'].min(), train_df['doj'].max())
        test_date_range = (test_df['doj'].min(), test_df['doj'].max())

        logger.info(f"Train date range: {train_date_range}")
        logger.info(f"Test date range: {test_date_range}")

        # Check for data leakage
        if test_date_range[0] < train_date_range[1]:
            logger.warning("Potential data leakage: test dates overlap with train dates")

    def save_processed_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save processed data to the processed directory."""
        processed_path = Path(self.data_config.processed_path)
        processed_path.mkdir(parents=True, exist_ok=True)

        filepath = processed_path / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")

    def load_processed_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load processed data if it exists."""
        filepath = Path(self.data_config.processed_path) / filename

        if filepath.exists():
            logger.info(f"Loading processed data from {filepath}")
            return pd.read_csv(filepath)
        else:
            logger.info(f"Processed data not found: {filepath}")
            return None
