#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section A.1: Data Loading and Cleaning
=======================================
Government Payroll Forecasting Model - Data Science Test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_and_clean_data():
    """
    Load and clean the dataset from provided files
    """
    print("=== Section A.1: Data Loading and Cleaning ===\n")

    # 1. Load payroll data
    print("1. Loading payroll data...")
    try:
        payroll_df = pd.read_csv('payroll_data_csv.txt')
        print(f"✓ Loaded {len(payroll_df)} payroll records")
    except FileNotFoundError:
        print("⚠️ File not found - creating mock data")
        payroll_df = create_mock_payroll_data()

    # 2. Load exogenous data
    print("2. Loading exogenous data...")
    try:
        exogenous_df = pd.read_csv('exogenous_data_csv.txt')
        print(f"✓ Loaded {len(exogenous_df)} exogenous records")
    except FileNotFoundError:
        print("⚠️ File not found - creating mock data")
        exogenous_df = create_mock_exogenous_data()

    # 3. Clean payroll data
    print("\n3. Cleaning payroll data...")

    # Remove missing values
    initial_rows = len(payroll_df)
    payroll_df = payroll_df.dropna(subset=['TARICH_SACHAR', 'NAME_MISRAD_AVIV', 'MASKORET_BRUTO_HEFRESHIM'])
    print(f"✓ Removed {initial_rows - len(payroll_df)} rows with missing values")

    # Fix data types
    payroll_df['TARICH_SACHAR'] = pd.to_datetime(payroll_df['TARICH_SACHAR'])
    payroll_df['MASKORET_BRUTO_HEFRESHIM'] = pd.to_numeric(payroll_df['MASKORET_BRUTO_HEFRESHIM'], errors='coerce')

    # Remove extreme outliers using IQR method
    Q1 = payroll_df['MASKORET_BRUTO_HEFRESHIM'].quantile(0.25)
    Q3 = payroll_df['MASKORET_BRUTO_HEFRESHIM'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_before = len(payroll_df)
    payroll_df = payroll_df[
        (payroll_df['MASKORET_BRUTO_HEFRESHIM'] >= lower_bound) &
        (payroll_df['MASKORET_BRUTO_HEFRESHIM'] <= upper_bound)
        ]
    outliers_removed = outliers_before - len(payroll_df)
    print(f"✓ Removed {outliers_removed} extreme outliers")

    # 4. Clean exogenous data
    print("\n4. Cleaning exogenous data...")

    exogenous_df['DATE'] = pd.to_datetime(exogenous_df['DATE'])

    # Fill missing values in exogenous data
    numeric_columns = ['CPI_INDEX', 'UNEMPLOYMENT_RATE', 'GDP_GROWTH_RATE', 'SHEKEL_USD_RATE']
    for col in numeric_columns:
        if col in exogenous_df.columns:
            missing_before = exogenous_df[col].isna().sum()
            exogenous_df[col] = exogenous_df[col].fillna(method='ffill')  # Forward fill
            print(f"✓ Filled {missing_before} missing values in {col}")

    return payroll_df, exogenous_df


def display_data_summary(payroll_df, exogenous_df):
    """
    Display descriptive statistics of the datasets
    """
    print("\n=== Descriptive Statistics ===\n")

    # Payroll data summary
    print("1. PAYROLL DATA SUMMARY:")
    print(f"   • Total records: {len(payroll_df):,}")
    print(f"   • Date range: {payroll_df['TARICH_SACHAR'].min()} to {payroll_df['TARICH_SACHAR'].max()}")
    print(f"   • Ministries: {payroll_df['NAME_MISRAD_AVIV'].nunique()}")
    print(f"   • Total salary sum: ₪{payroll_df['MASKORET_BRUTO_HEFRESHIM'].sum():,.0f}")
    print(f"   • Average monthly salary: ₪{payroll_df['MASKORET_BRUTO_HEFRESHIM'].mean():,.0f}")
    print(f"   • Median monthly salary: ₪{payroll_df['MASKORET_BRUTO_HEFRESHIM'].median():,.0f}")

    print("\n   Ministries in dataset:")
    for ministry in sorted(payroll_df['NAME_MISRAD_AVIV'].unique()):
        count = len(payroll_df[payroll_df['NAME_MISRAD_AVIV'] == ministry])
        print(f"   - {ministry}: {count} records")

    # Exogenous data summary
    print(f"\n2. EXOGENOUS DATA SUMMARY:")
    print(f"   • Total records: {len(exogenous_df):,}")
    print(f"   • Date range: {exogenous_df['DATE'].min()} to {exogenous_df['DATE'].max()}")

    # Statistical summary for key metrics
    if 'CPI_INDEX' in exogenous_df.columns:
        print(f"   • CPI Index range: {exogenous_df['CPI_INDEX'].min():.1f} - {exogenous_df['CPI_INDEX'].max():.1f}")
    if 'UNEMPLOYMENT_RATE' in exogenous_df.columns:
        print(
            f"   • Unemployment rate range: {exogenous_df['UNEMPLOYMENT_RATE'].min():.1f}% - {exogenous_df['UNEMPLOYMENT_RATE'].max():.1f}%")
    if 'GDP_GROWTH_RATE' in exogenous_df.columns:
        print(
            f"   • GDP growth range: {exogenous_df['GDP_GROWTH_RATE'].min():.1f}% - {exogenous_df['GDP_GROWTH_RATE'].max():.1f}%")


def create_mock_payroll_data():
    """Create mock payroll data for testing"""
    ministries = [
        'Ministry of Health', 'Ministry of Education', 'Ministry of Defense',
        'Ministry of Finance', 'Prime Minister Office'
    ]

    dates = pd.date_range('2019-01-01', '2024-06-01', freq='MS')
    data = []

    np.random.seed(42)  # For reproducibility

    for ministry in ministries:
        base_salary = np.random.uniform(50000, 100000)
        for date in dates:
            # Add trend + seasonality + noise
            trend = 1 + (date.year - 2019) * 0.03
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(1, 0.05)

            # Special events impact
            covid_effect = 1
            if '2020-03' <= date.strftime('%Y-%m') <= '2021-06':
                covid_effect = 1.2 if ministry == 'Ministry of Health' else 1.05

            war_effect = 1
            if date.strftime('%Y-%m') >= '2023-10':
                war_effect = 1.15 if ministry == 'Ministry of Defense' else 1.03

            total_salary = int(base_salary * trend * seasonal * noise * covid_effect * war_effect)

            data.append({
                'TARICH_SACHAR': date,
                'NAME_MISRAD_AVIV': ministry,
                'MASKORET_BRUTO_HEFRESHIM': total_salary
            })

    return pd.DataFrame(data)


def create_mock_exogenous_data():
    """Create mock exogenous data for testing"""
    dates = pd.date_range('2019-01-01', '2024-06-01', freq='MS')
    data = []

    np.random.seed(42)  # For reproducibility

    cpi_base = 100
    unemployment_base = 4
    gdp_base = 3
    usd_base = 3.5

    for i, date in enumerate(dates):
        # Add trends and random variations
        cpi = cpi_base + i * 0.3 + np.random.normal(0, 0.5)
        unemployment = max(1, unemployment_base + np.random.normal(0, 0.3))
        gdp = gdp_base + np.random.normal(0, 0.5)
        usd = max(3, usd_base + np.random.normal(0, 0.1))

        # Special events
        covid_indicator = 1 if '2020-03' <= date.strftime('%Y-%m') <= '2021-12' else 0
        war_indicator = 1 if date.strftime('%Y-%m') >= '2023-10' else 0

        data.append({
            'DATE': date,
            'CPI_INDEX': cpi,
            'UNEMPLOYMENT_RATE': unemployment,
            'GDP_GROWTH_RATE': gdp,
            'SHEKEL_USD_RATE': usd,
            'COVID_INDICATOR': covid_indicator,
            'WAR_INDICATOR': war_indicator
        })

    return pd.DataFrame(data)


# Main execution
if __name__ == "__main__":
    # Load and clean data
    payroll_data, exogenous_data = load_and_clean_data()

    # Display summary statistics
    display_data_summary(payroll_data, exogenous_data)

    print("\n✅ Section A.1 completed successfully!")
    print("Data is now cleaned and ready for analysis.")