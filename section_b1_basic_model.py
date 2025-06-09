#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section B.1: Basic Forecasting Model Implementation
====================================================
Government Payroll Forecasting Model - Data Science Test
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class BasicPayrollForecastingModel:
    """
    Basic forecasting model using historical data only
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}

    def prepare_data(self, df, target_col='MASKORET_BRUTO_HEFRESHIM'):
        """
        Prepare data for modeling - historical data only
        """
        print("=== Section B.1: Basic Model Implementation ===\n")
        print("1. Preparing data for basic model...")

        # Select only historical/time-based features (no exogenous factors)
        historical_features = [
            'year', 'month', 'quarter', 'days_since_start', 'months_since_start',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'ministry_encoded', 'year_since_2019'
        ]

        # Add lag features if available
        lag_features = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'growth' in col]
        historical_features.extend(lag_features)

        # Filter features that actually exist in the dataframe
        available_features = [f for f in historical_features if f in df.columns]

        if not available_features:
            print("   ⚠️ No historical features found, creating basic time features...")
            available_features = self._create_basic_time_features(df)

        self.feature_names = available_features
        print(f"   • Using {len(available_features)} historical features")

        # Prepare X and y
        X = df[available_features].copy()
        y = df[target_col].copy()

        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        print(f"   • Dataset shape: {X.shape}")
        print(f"   • Target variable range: ₪{y.min():,.0f} - ₪{y.max():,.0f}")

        return X, y

    def _create_basic_time_features(self, df):
        """
        Create basic time features if they don't exist
        """
        if 'date' not in df.columns:
            return []

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

        # Simple ministry encoding
        if 'NAME_MISRAD_AVIV' in df.columns:
            df['ministry_encoded'] = pd.Categorical(df['NAME_MISRAD_AVIV']).codes

        return ['year', 'month', 'quarter', 'days_since_start', 'ministry_encoded']

    def train_models(self, X, y, test_size=0.2):
        """
        Train multiple models using historical data only
        """
        print("\n2. Training basic forecasting models...")

        # Time series split for proper evaluation
        tscv = TimeSeriesSplit(n_splits=3)

        # Split data - use time series split
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        print(f"   • Training set: {X_train.shape[0]} samples")
        print(f"   • Test set: {X_test.shape[0]} samples")

        # Model 1: Linear Regression
        print("\n   Training Linear Regression...")
        lr_model = LinearRegression()

        # Scale features for linear regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)

        self.models['Linear Regression'] = lr_model
        self.scalers['Linear Regression'] = scaler

        # Model 2: Random Forest
        print("   Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        self.models['Random Forest'] = rf_model

        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = {
            'Linear Regression': lr_pred,
            'Random Forest': rf_pred
        }

        print("   ✓ Models trained successfully")

        return X_train, X_test, y_train, y_test

    def evaluate_models(self):
        """
        Evaluate model performance using standard metrics
        """
        print("\n3. Evaluating model performance...")

        results = {}

        for model_name, predictions in self.predictions.items():
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            mape = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100

            results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }

            print(f"\n   {model_name} Results:")
            print(f"   • RMSE: ₪{rmse:,.0f}")
            print(f"   • MAE: ₪{mae:,.0f}")
            print(f"   • R²: {r2:.3f}")
            print(f"   • MAPE: {mape:.1f}%")

        self.results = results
        return results

    def compare_models(self):
        """
        Create comparison visualizations
        """
        print("\n4. Creating model comparison visualizations...")

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Basic Model Performance Comparison', fontsize=16, fontweight='bold')

        # 1. Metrics comparison
        metrics_df = pd.DataFrame(self.results).T

        # RMSE comparison
        axes[0, 0].bar(metrics_df.index, metrics_df['RMSE'], color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE (₪)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # R² comparison
        axes[0, 1].bar(metrics_df.index, metrics_df['R²'], color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('R² Comparison')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 2. Actual vs Predicted (best model)
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        best_predictions = self.predictions[best_model]

        axes[1, 0].scatter(self.y_test, best_predictions, alpha=0.6, color='green')
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'Actual vs Predicted - {best_model}')
        axes[1, 0].set_xlabel('Actual Values (₪)')
        axes[1, 0].set_ylabel('Predicted Values (₪)')

        # 3. Residuals plot
        residuals = self.y_test - best_predictions
        axes[1, 1].scatter(best_predictions, residuals, alpha=0.6, color='orange')
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title(f'Residuals Plot - {best_model}')
        axes[1, 1].set_xlabel('Predicted Values (₪)')
        axes[1, 1].set_ylabel('Residuals (₪)')

        plt.tight_layout()
        plt.show()

        # Feature importance for Random Forest
        if 'Random Forest' in self.models:
            self