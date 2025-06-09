#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section A.3: Feature Engineering for Modeling
==============================================
Government Payroll Forecasting Model - Data Science Test
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')


def prepare_data_for_modeling(payroll_df, exogenous_df):
    """
    Prepare and engineer features for modeling
    """
    print("=== Section A.3: Feature Engineering for Modeling ===\n")

    # 1. Merge datasets
    print("1. Merging payroll and exogenous data...")
    merged_data = merge_datasets(payroll_df, exogenous_df)

    # 2. Create time-based features
    print("2. Creating time-based features...")
    merged_data = create_time_features(merged_data)

    # 3. Create lag features
    print("3. Creating lag and rolling features...")
    merged_data = create_lag_features(merged_data)

    # 4. Encode categorical variables
    print("4. Encoding categorical variables...")
    merged_data, encoders = encode_categorical_features(merged_data)

    # 5. Create interaction features
    print("5. Creating interaction features...")
    merged_data = create_interaction_features(merged_data)

    # 6. Handle missing values
    print("6. Handling missing values...")
    merged_data = handle_missing_values(merged_data)

    # 7. Feature selection
    print("7. Performing feature selection...")
    final_features, feature_importance = select_best_features(merged_data)

    print(f"\n✅ Feature engineering completed!")
    print(f"   • Final dataset shape: {merged_data.shape}")
    print(f"   • Selected features: {len(final_features)}")

    return merged_data, encoders, final_features, feature_importance


def merge_datasets(payroll_df, exogenous_df):
    """
    Merge payroll and exogenous datasets
    """
    # Prepare date columns
    payroll_df['date'] = pd.to_datetime(payroll_df['TARICH_SACHAR'])
    exogenous_df['date'] = pd.to_datetime(exogenous_df['DATE'])

    # Extract year-month for merging
    payroll_df['year_month'] = payroll_df['date'].dt.to_period('M')
    exogenous_df['year_month'] = exogenous_df['date'].dt.to_period('M')

    # Merge on year-month
    merged = pd.merge(payroll_df, exogenous_df, on='year_month', how='inner', suffixes=('_payroll', '_exog'))

    print(f"   • Payroll data: {len(payroll_df)} records")
    print(f"   • Exogenous data: {len(exogenous_df)} records")
    print(f"   • Merged data: {len(merged)} records")

    return merged


def create_time_features(df):
    """
    Create time-based features
    """
    # Use payroll date as primary date
    df['date'] = df['date_payroll']

    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # Time since start
    start_date = df['date'].min()
    df['days_since_start'] = (df['date'] - start_date).dt.days
    df['months_since_start'] = df['days_since_start'] / 30.44  # Average days per month

    # Year-over-year indicator
    df['year_since_2019'] = df['year'] - 2019

    print(f"   • Created {8} time-based features")

    return df


def create_lag_features(df):
    """
    Create lag and rolling window features
    """
    # Target variable
    target_col = 'MASKORET_BRUTO_HEFRESHIM'

    # Sort by ministry and date
    df = df.sort_values(['NAME_MISRAD_AVIV', 'date'])

    # Create lag features for each ministry
    lag_periods = [1, 3, 6, 12]  # 1, 3, 6, and 12 months
    rolling_windows = [3, 6, 12]  # 3, 6, and 12 months

    for ministry in df['NAME_MISRAD_AVIV'].unique():
        if pd.isna(ministry):
            continue

        mask = df['NAME_MISRAD_AVIV'] == ministry
        ministry_data = df.loc[mask, target_col]

        # Lag features
        for lag in lag_periods:
            df.loc[mask, f'salary_lag_{lag}'] = ministry_data.shift(lag)

        # Rolling features
        for window in rolling_windows:
            df.loc[mask, f'salary_rolling_mean_{window}'] = ministry_data.rolling(window=window, min_periods=1).mean()
            df.loc[mask, f'salary_rolling_std_{window}'] = ministry_data.rolling(window=window, min_periods=1).std()

        # Growth rate features
        df.loc[mask, 'salary_growth_1m'] = ministry_data.pct_change(periods=1)
        df.loc[mask, 'salary_growth_3m'] = ministry_data.pct_change(periods=3)
        df.loc[mask, 'salary_growth_12m'] = ministry_data.pct_change(periods=12)

    print(
        f"   • Created lag and rolling features for {len(lag_periods)} lag periods and {len(rolling_windows)} windows")

    return df


def encode_categorical_features(df):
    """
    Encode categorical variables
    """
    encoders = {}

    # Label encoding for ministry (ordinal relationship might exist)
    le_ministry = LabelEncoder()
    df['ministry_encoded'] = le_ministry.fit_transform(df['NAME_MISRAD_AVIV'].fillna('Unknown'))
    encoders['ministry'] = le_ministry

    # One-hot encoding for other categorical features if they exist
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['NAME_MISRAD_AVIV', 'date', 'year_month']:
            categorical_cols.append(col)

    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        print(f"   • One-hot encoded {len(categorical_cols)} categorical features")
        return df_encoded, encoders
    else:
        print(f"   • Label encoded ministry feature")
        return df, encoders


def create_interaction_features(df):
    """
    Create interaction features between important variables
    """
    # Economic interaction features
    if 'CPI_INDEX' in df.columns and 'UNEMPLOYMENT_RATE' in df.columns:
        df['cpi_unemployment_interaction'] = df['CPI_INDEX'] * df['UNEMPLOYMENT_RATE']

    if 'CPI_INDEX' in df.columns:
        df['cpi_month_interaction'] = df['CPI_INDEX'] * df['month']
        df['cpi_trend'] = df['CPI_INDEX'] * df['days_since_start']

    if 'GDP_GROWTH_RATE' in df.columns and 'UNEMPLOYMENT_RATE' in df.columns:
        df['gdp_unemployment_ratio'] = df['GDP_GROWTH_RATE'] / (
                    df['UNEMPLOYMENT_RATE'] + 0.1)  # Add small constant to avoid division by zero

    # Ministry-specific interactions
    if 'ministry_encoded' in df.columns and 'month' in df.columns:
        df['ministry_month_interaction'] = df['ministry_encoded'] * df['month']

    # Time-based interactions
    df['year_month_interaction'] = df['year'] * df['month']
    df['quarter_year_interaction'] = df['quarter'] * df['year_since_2019']

    print(f"   • Created interaction features")

    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    # Count missing values before
    missing_before = df.isnull().sum().sum()

    # Strategy 1: Forward fill for time series data
    time_series_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'growth' in col]
    for col in time_series_cols:
        df[col] = df.groupby('NAME_MISRAD_AVIV')[col].fillna(method='ffill')

    # Strategy 2: Fill with median for economic indicators
    economic_cols = ['CPI_INDEX', 'UNEMPLOYMENT_RATE', 'GDP_GROWTH_RATE', 'SHEKEL_USD_RATE']
    for col in economic_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Strategy 3: Fill with 0 for indicator variables
    indicator_cols = [col for col in df.columns if 'INDICATOR' in col or col.endswith('_FLAG')]
    for col in indicator_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Strategy 4: Fill remaining with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    missing_after = df.isnull().sum().sum()
    print(f"   • Handled {missing_before - missing_after} missing values")

    return df


def select_best_features(df):
    """
    Select the best features for modeling
    """
    # Define target and feature columns
    target_col = 'MASKORET_BRUTO_HEFRESHIM'

    # Exclude non-feature columns
    exclude_cols = [
        target_col, 'NAME_MISRAD_AVIV', 'date', 'date_payroll', 'date_exog',
        'year_month', 'TARICH_SACHAR', 'DATE'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.number, int, float]]

    # Remove features with too many missing values
    feature_cols = [col for col in feature_cols if df[col].isnull().sum() / len(df) < 0.5]

    # Remove constant features
    feature_cols = [col for col in feature_cols if df[col].nunique() > 1]

    if len(feature_cols) == 0:
        print("   ⚠️ No suitable features found for selection")
        return [], {}

    # Prepare data for feature selection
    X = df[feature_cols].fillna(0)  # Fill any remaining NaN with 0
    y = df[target_col].fillna(df[target_col].median())

    # Remove rows where target is NaN
    valid_rows = ~y.isnull()
    X = X[valid_rows]
    y = y[valid_rows]

    if len(X) == 0:
        print("   ⚠️ No valid data for feature selection")
        return feature_cols, {}

    # Feature selection using statistical tests
    try:
        k_best = min(20, len(feature_cols))  # Select top 20 features or all if less
        selector = SelectKBest(score_func=f_regression, k=k_best)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names and scores
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        feature_scores = dict(zip(selected_features, selector.scores_[selector.get_support()]))

        # Sort by importance
        feature_importance = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))

        print(f"   • Selected {len(selected_features)} features out of {len(feature_cols)}")
        print(f"   • Top 5 features: {list(feature_importance.keys())[:5]}")

        return selected_features, feature_importance

    except Exception as e:
        print(f"   ⚠️ Feature selection failed: {e}")
        return feature_cols[:20], {}  # Return first 20 features as fallback


def create_final_dataset(df, selected_features):
    """
    Create the final dataset for modeling
    """
    target_col = 'MASKORET_BRUTO_HEFRESHIM'

    # Include target and selected features
    final_cols = [target_col] + selected_features + ['NAME_MISRAD_AVIV', 'date']
    final_df = df[final_cols].copy()

    # Remove rows with missing target values
    final_df = final_df.dropna(subset=[target_col])

    # Sort by ministry and date
    final_df = final_df.sort_values(['NAME_MISRAD_AVIV', 'date'])

    return final_df


# Main execution function for Section A.3
def run_feature_engineering():
    """
    Main function to run the complete feature engineering process
    """
    print("Loading data for feature engineering...")

    # Import data creation functions
    from section_a1_data_cleaning import create_mock_payroll_data, create_mock_exogenous_data

    # Create sample data
    payroll_df = create_mock_payroll_data()
    exogenous_df = create_mock_exogenous_data()

    # Run feature engineering
    engineered_data, encoders, selected_features, feature_importance = prepare_data_for_modeling(payroll_df,
                                                                                                 exogenous_df)

    # Create final modeling dataset
    final_dataset = create_final_dataset(engineered_data, selected_features)

    print(f"\n=== Feature Engineering Summary ===")
    print(f"✓ Final dataset shape: {final_dataset.shape}")
    print(f"✓ Selected features: {len(selected_features)}")
    print(f"✓ Feature importance calculated: {'Yes' if feature_importance else 'No'}")

    if feature_importance:
        print(f"\n📊 Top 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(feature_importance.items())[:10], 1):
            print(f"   {i:2d}. {feature}: {score:.2f}")

    print("\n✅ Section A.3 completed successfully!")

    return final_dataset, encoders, selected_features, feature_importance


if __name__ == "__main__":
    dataset, encoders, features, importance = run_feature_engineering()