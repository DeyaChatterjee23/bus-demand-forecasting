"""
Enhanced feature engineering module with modular approach.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import holidays
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import logging
import yaml
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseFeatureEngineer(ABC):
    """Base class for feature engineering components."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BaseFeatureEngineer':
        """Fit the feature engineer on training data."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class TemporalFeatureEngineer(BaseFeatureEngineer):
    """Create temporal features from date columns."""

    def __init__(self, date_col: str = 'doj'):
        self.date_col = date_col
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'TemporalFeatureEngineer':
        """Fit method for temporal features (no fitting needed)."""
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        if not self.fitted:
            raise ValueError("TemporalFeatureEngineer not fitted")

        df = df.copy()
        date_col = self.date_col

        logger.info(f"Creating temporal features from {date_col}")

        # Basic temporal features
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_day'] = df[date_col].dt.day
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
        df[f'{date_col}_dayofyear'] = df[date_col].dt.dayofyear
        df[f'{date_col}_week'] = df[date_col].dt.isocalendar().week
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter

        # Weekend and month flags
        df[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        df[f'{date_col}_is_month_start'] = (df[date_col].dt.day <= 5).astype(int)
        df[f'{date_col}_is_month_end'] = (df[date_col].dt.day >= 25).astype(int)

        # Days to weekend
        df[f'{date_col}_days_to_weekend'] = df[date_col].dt.dayofweek.apply(
            lambda x: (4 - x) if x <= 4 else (11 - x)
        )

        # Cyclical features for better ML performance
        df[f'{date_col}_sin_month'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
        df[f'{date_col}_cos_month'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
        df[f'{date_col}_sin_day'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365)
        df[f'{date_col}_cos_day'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365)

        return df


class HolidayFeatureEngineer(BaseFeatureEngineer):
    """Create holiday-related features."""

    def __init__(self, date_col: str = 'doj', country: str = 'India'):
        self.date_col = date_col
        self.country = country
        self.holidays_dict = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'HolidayFeatureEngineer':
        """Fit method to create holiday dictionary."""
        date_range = df[self.date_col].dt.year
        years = range(date_range.min() - 1, date_range.max() + 2)

        if self.country.lower() == 'india':
            self.holidays_dict = holidays.India(years=years)
        else:
            self.holidays_dict = holidays.country_holidays(self.country, years=years)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday features."""
        if not self.fitted:
            raise ValueError("HolidayFeatureEngineer not fitted")

        df = df.copy()
        date_col = self.date_col

        logger.info(f"Creating holiday features from {date_col}")

        # Holiday flag
        df[f'{date_col}_is_holiday'] = df[date_col].dt.date.apply(
            lambda x: x in self.holidays_dict
        ).astype(int)

        # Days to next/previous holiday
        df[f'{date_col}_days_to_holiday'] = df[date_col].apply(
            lambda x: min([abs((holiday - x.date()).days) 
                          for holiday in self.holidays_dict.keys() 
                          if abs((holiday - x.date()).days) <= 30] + [30])
        )

        # Seasonal patterns in India
        df[f'{date_col}_wedding_season'] = df[date_col].dt.month.isin([11, 12, 1, 2]).astype(int)
        df[f'{date_col}_exam_season'] = df[date_col].dt.month.isin([3, 4, 5, 6]).astype(int)
        df[f'{date_col}_festival_season'] = df[date_col].dt.month.isin([9, 10, 11]).astype(int)

        return df


class RouteFeatureEngineer(BaseFeatureEngineer):
    """Create route-specific features."""

    def __init__(self, transactions_df: pd.DataFrame):
        self.transactions_df = transactions_df
        self.route_stats = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'RouteFeatureEngineer':
        """Fit method to create route statistics."""
        # Create route information from transactions
        cols_to_keep = ['srcid', 'destid']

        # Add tier and region columns if they exist
        for prefix in ['srcid', 'destid']:
            for suffix in ['tier', 'region']:
                col = f'{prefix}_{suffix}'
                if col in self.transactions_df.columns:
                    cols_to_keep.append(col)

        # Create static route info
        self.route_stats = self.transactions_df[cols_to_keep].drop_duplicates(
            subset=['srcid', 'destid']
        ).copy()

        self.route_stats['route_id'] = (
            self.route_stats['srcid'].astype(str) + '_' + 
            self.route_stats['destid'].astype(str)
        )

        # Create combination features
        if 'srcid_tier' in self.route_stats.columns and 'destid_tier' in self.route_stats.columns:
            self.route_stats['tier_combination'] = (
                self.route_stats['srcid_tier'] + '_to_' + self.route_stats['destid_tier']
            )
            self.route_stats['same_tier'] = (
                self.route_stats['srcid_tier'] == self.route_stats['destid_tier']
            ).astype(int)

        if 'srcid_region' in self.route_stats.columns and 'destid_region' in self.route_stats.columns:
            self.route_stats['region_combination'] = (
                self.route_stats['srcid_region'] + '_to_' + self.route_stats['destid_region']
            )
            self.route_stats['same_region'] = (
                self.route_stats['srcid_region'] == self.route_stats['destid_region']
            ).astype(int)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add route features to dataframe."""
        if not self.fitted:
            raise ValueError("RouteFeatureEngineer not fitted")

        df = df.copy()

        logger.info("Creating route features")

        # Create route_id
        df['route_id'] = df['srcid'].astype(str) + '_' + df['destid'].astype(str)

        # Merge with route statistics
        merge_cols = [col for col in self.route_stats.columns if col != 'srcid' and col != 'destid']
        df = df.merge(self.route_stats[merge_cols], on='route_id', how='left')

        return df


class BookingPatternFeatureEngineer(BaseFeatureEngineer):
    """Create booking pattern features from transactions."""

    def __init__(self, transactions_df: pd.DataFrame):
        self.transactions_df = transactions_df
        self.booking_features = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'BookingPatternFeatureEngineer':
        """Fit method to create booking pattern features."""
        logger.info("Creating booking pattern features")

        # Group by route and date
        group_cols = ['srcid', 'destid', 'doj']
        trans = self.transactions_df.sort_values(group_cols + ['doi'])

        # Calculate differences within each group
        trans['seat_diff'] = trans.groupby(group_cols)['cumsum_seatcount'].diff().fillna(0)
        trans['search_diff'] = trans.groupby(group_cols)['cumsum_searchcount'].diff().fillna(0)
        trans['booking_acceleration'] = trans.groupby(group_cols)['seat_diff'].diff().fillna(0)

        # Filter for data >= 15 days before departure
        first_15_days = trans[trans['dbd'] >= 15].copy()

        if first_15_days.empty:
            logger.warning("No data available 15+ days before departure")
            self.booking_features = pd.DataFrame(columns=group_cols)
            self.fitted = True
            return self

        # Define aggregation functions
        def trend_slope(s):
            if len(s) < 2:
                return 0
            return np.polyfit(np.arange(len(s)), s.values, 1)[0]

        def concavity(s):
            if len(s) < 3:
                return 0
            return np.diff(s.values, 2).mean()

        # Main aggregation
        self.booking_features = first_15_days.groupby(group_cols).agg({
            'cumsum_seatcount': ['last', trend_slope, concavity],
            'cumsum_searchcount': 'last',
            'seat_diff': ['mean', 'std', 'max'],
            'search_diff': ['mean', 'std'],
            'booking_acceleration': 'mean'
        }).reset_index()

        # Flatten column names
        self.booking_features.columns = [
            '_'.join(col).strip('_') if col[1] else col[0] 
            for col in self.booking_features.columns
        ]

        # Rename columns for clarity
        rename_map = {
            'cumsum_seatcount_last': 'cumsum_seatcount_day15',
            'cumsum_searchcount_last': 'cumsum_searchcount_day15',
            'cumsum_seatcount_trend_slope': 'booking_trend_slope',
            'cumsum_seatcount_concavity': 'booking_concavity',
            'seat_diff_mean': 'booking_velocity_mean',
            'seat_diff_std': 'booking_velocity_std',
            'seat_diff_max': 'booking_velocity_max',
            'search_diff_mean': 'search_velocity_mean',
            'search_diff_std': 'search_velocity_std',
            'booking_acceleration_mean': 'booking_acceleration'
        }

        self.booking_features.rename(columns=rename_map, inplace=True)

        # Calculate conversion rate
        self.booking_features['conversion_rate_day15'] = (
            self.booking_features['cumsum_seatcount_day15'] / 
            self.booking_features['cumsum_searchcount_day15']
        ).fillna(0).replace(np.inf, 0)

        # Handle early bookings (days 20-25)
        early_bookings = trans[trans['dbd'].between(20, 25)]
        if not early_bookings.empty:
            early_stats = early_bookings.groupby(group_cols).agg({
                'cumsum_seatcount': 'last',
                'cumsum_searchcount': 'last'
            }).reset_index()

            early_stats.columns = group_cols + ['early_booking_seats', 'early_booking_searches']

            self.booking_features = self.booking_features.merge(
                early_stats, on=group_cols, how='left'
            )
        else:
            self.booking_features['early_booking_seats'] = 0
            self.booking_features['early_booking_searches'] = 0

        # Fill NaN values
        self.booking_features.fillna(0, inplace=True)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add booking pattern features to dataframe."""
        if not self.fitted:
            raise ValueError("BookingPatternFeatureEngineer not fitted")

        df = df.copy()

        # Merge with booking features
        merge_cols = ['srcid', 'destid', 'doj']
        df = df.merge(self.booking_features, on=merge_cols, how='left')

        # Fill missing values for routes not in training data
        booking_cols = [col for col in self.booking_features.columns if col not in merge_cols]
        df[booking_cols] = df[booking_cols].fillna(0)

        return df


class CategoricalEncoder(BaseFeatureEngineer):
    """Encode categorical features with proper handling of unseen categories."""

    def __init__(self, categorical_cols: List[str]):
        self.categorical_cols = categorical_cols
        self.encoders = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'CategoricalEncoder':
        """Fit encoders on training data."""
        for col in self.categorical_cols:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].astype(str).fillna('unknown'))

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns."""
        if not self.fitted:
            raise ValueError("CategoricalEncoder not fitted")

        df = df.copy()

        for col in self.categorical_cols:
            if col in df.columns and col in self.encoders:
                # Handle unseen categories
                known_labels = set(self.encoders[col].classes_)
                df[f'{col}_temp'] = df[col].astype(str).fillna('unknown').apply(
                    lambda x: x if x in known_labels else 'unknown'
                )

                # Add unknown category if not in classes
                if 'unknown' not in known_labels:
                    self.encoders[col].classes_ = np.append(
                        self.encoders[col].classes_, 'unknown'
                    )

                df[f'{col}_encoded'] = self.encoders[col].transform(df[f'{col}_temp'])
                df.drop(f'{col}_temp', axis=1, inplace=True)

        return df
