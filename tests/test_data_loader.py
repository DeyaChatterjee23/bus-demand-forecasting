"""
Tests for data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

# Adjust import path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from redbus_forecasting.data.data_loader import DataLoader


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    train_data = {
        'srcid': [1, 2, 3],
        'destid': [4, 5, 6],
        'doj': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'final_seatcount': [25, 30, 35]
    }

    test_data = {
        'srcid': [1, 2],
        'destid': [4, 5],
        'doj': ['2024-01-04', '2024-01-05']
    }

    transactions_data = {
        'srcid': [1, 1, 2],
        'destid': [4, 4, 5],
        'doj': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'doi': ['2023-12-20', '2023-12-25', '2023-12-28'],
        'cumsum_seatcount': [10, 20, 15],
        'cumsum_searchcount': [50, 100, 75],
        'dbd': [15, 10, 5]
    }

    return train_data, test_data, transactions_data


@pytest.fixture
def temp_csv_files(sample_data):
    """Create temporary CSV files for testing."""
    train_data, test_data, transactions_data = sample_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV files
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        transactions_df = pd.DataFrame(transactions_data)

        train_path = os.path.join(temp_dir, 'train.csv')
        test_path = os.path.join(temp_dir, 'test.csv')
        transactions_path = os.path.join(temp_dir, 'transactions.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        transactions_df.to_csv(transactions_path, index=False)

        yield train_path, test_path, transactions_path


def test_data_loader_initialization():
    """Test DataLoader initialization."""
    # Mock config loading
    with patch.object(DataLoader, '_load_config') as mock_config:
        mock_config.return_value = {
            'data': {
                'train_path': 'train.csv',
                'test_path': 'test.csv',
                'transactions_path': 'transactions.csv',
                'processed_path': 'processed/'
            }
        }

        loader = DataLoader()
        assert loader.config is not None
        assert loader.data_config is not None


def test_load_data_success(temp_csv_files):
    """Test successful data loading."""
    train_path, test_path, transactions_path = temp_csv_files

    # Mock config
    mock_config = {
        'data': {
            'train_path': train_path,
            'test_path': test_path,
            'transactions_path': transactions_path,
            'processed_path': 'processed/'
        }
    }

    with patch.object(DataLoader, '_load_config', return_value=mock_config):
        loader = DataLoader()
        train_df, test_df, transactions_df = loader.load_data()

        assert len(train_df) == 3
        assert len(test_df) == 2
        assert len(transactions_df) == 3

        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(train_df['doj'])
        assert pd.api.types.is_datetime64_any_dtype(transactions_df['doj'])


def test_data_validation():
    """Test data validation functionality."""
    # Create data with missing columns
    train_df = pd.DataFrame({'srcid': [1], 'destid': [2]})  # Missing required columns
    test_df = pd.DataFrame({'srcid': [1], 'destid': [2], 'doj': ['2024-01-01']})
    transactions_df = pd.DataFrame({
        'srcid': [1], 'destid': [2], 'doj': ['2024-01-01'], 
        'doi': ['2023-12-01'], 'cumsum_seatcount': [10]
    })

    mock_config = {
        'data': {
            'train_path': 'train.csv',
            'test_path': 'test.csv', 
            'transactions_path': 'transactions.csv',
            'processed_path': 'processed/'
        }
    }

    with patch.object(DataLoader, '_load_config', return_value=mock_config):
        loader = DataLoader()

        with pytest.raises(ValueError):
            loader._validate_data(train_df, test_df, transactions_df)
