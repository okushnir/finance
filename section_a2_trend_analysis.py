#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section A.2: Trend and Seasonality Analysis
============================================
Government Payroll Forecasting Model - Data Science Test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')


def analyze_trends_and_seasonality(payroll_df, exogenous_df):
    """
    Perform trend and seasonality analysis
    """
    print("=== Section A.2: Trend and Seasonality Analysis ===\n")

    # Prepare data for analysis
    print("1. Preparing data for trend analysis...")

    # Convert date column and create time features
    payroll_df['date'] = pd.to_datetime(payroll_df['TARICH_SACHAR'])
    payroll_df['year'] = payroll_df['date'].dt.year
    payroll_df['month'] = payroll_df['date'].dt.month
    payroll_df['quarter'] = payroll_df['date'].dt.quarter

    # Aggregate by date and ministry
    monthly_data = payroll_df.groupby(['date', 'NAME_MISRAD_AVIV'])['MASKORET_BRUTO_HEFRESHIM'].sum().reset_index()

    print(f"✓ Prepared {len(monthly_data)} monthly observations")

    # 2. Overall trend analysis
    print("\n2. Analyzing overall trends...")
    overall_trend = analyze_overall_trend(payroll_df)

    # 3. Seasonality analysis
    print("\n3. Analyzing seasonality patterns...")
    seasonal_patterns = analyze_seasonality(payroll_df)

    # 4. Ministry-specific analysis
    print("\n4. Analyzing differences between ministries...")
    ministry_analysis = analyze_ministry_differences(payroll_df)

    # 5. Create visualizations
    print("\n5. Creating visualizations...")
    create_trend_visualizations(payroll_df, exogenous_df)

    return {
        'overall_trend': overall_trend,
        'seasonal_patterns': seasonal_patterns,
        'ministry_analysis': ministry_analysis
    }


def analyze_overall_trend(payroll_df):
    """
    Analyze overall trend in payroll data
    """
    # Monthly aggregation
    monthly_total = payroll_df.groupby('date')['MASKORET_BRUTO_HEFRESHIM'].sum().reset_index()
    monthly_total = monthly_total.sort_values('date')

    # Calculate year-over-year growth
    monthly_total['year'] = monthly_total['date'].dt.year
    yearly_total = monthly_total.groupby('year')['MASKORET_BRUTO_HEFRESHIM'].sum().reset_index()
    yearly_total['growth_rate'] = yearly_total['MASKORET_BRUTO_HEFRESHIM'].pct_change() * 100

    # Calculate trend statistics
    trend_stats = {
        'total_growth_period': ((monthly_total['MASKORET_BRUTO_HEFRESHIM'].iloc[-1] /
                                 monthly_total['MASKORET_BRUTO_HEFRESHIM'].iloc[0]) - 1) * 100,
        'average_yearly_growth': yearly_total['growth_rate'].mean(),
        'trend_slope': np.polyfit(range(len(monthly_total)), monthly_total['MASKORET_BRUTO_HEFRESHIM'], 1)[0]
    }

    print(f"   • Total growth over period: {trend_stats['total_growth_period']:.1f}%")
    print(f"   • Average yearly growth: {trend_stats['average_yearly_growth']:.1f}%")
    print(f"   • Monthly trend slope: ₪{trend_stats['trend_slope']:,.0f}")

    return trend_stats


def analyze_seasonality(payroll_df):
    """
    Analyze seasonal patterns in payroll data
    """
    # Monthly seasonality
    monthly_avg = payroll_df.groupby('month')['MASKORET_BRUTO_HEFRESHIM'].mean()
    overall_avg = payroll_df['MASKORET_BRUTO_HEFRESHIM'].mean()

    # Calculate seasonal indices
    seasonal_index = (monthly_avg / overall_avg) * 100

    # Quarterly seasonality
    quarterly_avg = payroll_df.groupby('quarter')['MASKORET_BRUTO_HEFRESHIM'].mean()
    quarterly_index = (quarterly_avg / overall_avg) * 100

    # Find peak and low seasons
    peak_month = seasonal_index.idxmax()
    low_month = seasonal_index.idxmin()
    peak_quarter = quarterly_index.idxmax()
    low_quarter = quarterly_index.idxmin()

    seasonal_stats = {
        'monthly_index': seasonal_index,
        'quarterly_index': quarterly_index,
        'peak_month': peak_month,
        'low_month': low_month,
        'peak_quarter': peak_quarter,
        'low_quarter': low_quarter,
        'seasonality_strength': seasonal_index.std()
    }

    print(f"   • Peak month: {peak_month} ({seasonal_index[peak_month]:.1f}% of average)")
    print(f"   • Low month: {low_month} ({seasonal_index[low_month]:.1f}% of average)")
    print(f"   • Seasonality strength: {seasonal_stats['seasonality_strength']:.1f}%")

    return seasonal_stats


def analyze_ministry_differences(payroll_df):
    """
    Analyze differences between ministries
    """
    # Ministry statistics
    ministry_stats = payroll_df.groupby('NAME_MISRAD_AVIV').agg({
        'MASKORET_BRUTO_HEFRESHIM': ['mean', 'sum', 'std', 'count']
    }).round(0)

    ministry_stats.columns = ['avg_salary', 'total_salary', 'std_salary', 'observations']

    # Calculate growth rates by ministry
    ministry_growth = {}
    for ministry in payroll_df['NAME_MISRAD_AVIV'].unique():
        if pd.isna(ministry):
            continue

        ministry_data = payroll_df[payroll_df['NAME_MISRAD_AVIV'] == ministry].copy()
        ministry_data = ministry_data.sort_values('date')

        if len(ministry_data) > 1:
            first_value = ministry_data['MASKORET_BRUTO_HEFRESHIM'].iloc[0]
            last_value = ministry_data['MASKORET_BRUTO_HEFRESHIM'].iloc[-1]
            growth_rate = ((last_value / first_value) - 1) * 100
            ministry_growth[ministry] = growth_rate

    # Find ministries with highest and lowest growth
    if ministry_growth:
        highest_growth_ministry = max(ministry_growth, key=ministry_growth.get)
        lowest_growth_ministry = min(ministry_growth, key=ministry_growth.get)

        print(
            f"   • Highest growth ministry: {highest_growth_ministry} ({ministry_growth[highest_growth_ministry]:.1f}%)")
        print(f"   • Lowest growth ministry: {lowest_growth_ministry} ({ministry_growth[lowest_growth_ministry]:.1f}%)")

    ministry_analysis = {
        'ministry_stats': ministry_stats,
        'ministry_growth': ministry_growth
    }

    return ministry_analysis


def create_trend_visualizations(payroll_df, exogenous_df):
    """
    Create comprehensive visualizations for trend analysis
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Payroll Trend and Seasonality Analysis', fontsize=16, fontweight='bold')

    # 1. Overall trend over time
    monthly_total = payroll_df.groupby('date')['MASKORET_BRUTO_HEFRESHIM'].sum().reset_index()
    monthly_total = monthly_total.sort_values('date')

    axes[0, 0].plot(monthly_total['date'], monthly_total['MASKORET_BRUTO_HEFRESHIM'],
                    linewidth=2, color='navy', marker='o', markersize=4)
    axes[0, 0].set_title('Overall Payroll Trend Over Time')
    axes[0, 0].set_ylabel('Total Payroll (₪)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Add trend line
    x_numeric = np.arange(len(monthly_total))
    z = np.polyfit(x_numeric, monthly_total['MASKORET_BRUTO_HEFRESHIM'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(monthly_total['date'], p(x_numeric), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    axes[0, 0].legend()

    # 2. Seasonality pattern
    monthly_avg = payroll_df.groupby('month')['MASKORET_BRUTO_HEFRESHIM'].mean()

    axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Monthly Seasonality Pattern')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average Payroll (₪)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Ministry comparison
    ministry_totals = payroll_df.groupby('NAME_MISRAD_AVIV')['MASKORET_BRUTO_HEFRESHIM'].sum().sort_values(
        ascending=True)
    ministry_totals = ministry_totals.dropna()

    if len(ministry_totals) > 0:
        # Take top 6 ministries to avoid overcrowding
        top_ministries = ministry_totals.tail(6)

        axes[1, 0].barh(range(len(top_ministries)), top_ministries.values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_yticks(range(len(top_ministries)))
        # Truncate long ministry names for better display
        ministry_labels = [name[:25] + '...' if len(str(name)) > 25 else str(name) for name in top_ministries.index]
        axes[1, 0].set_yticklabels(ministry_labels, fontsize=8)
        axes[1, 0].set_title('Total Payroll by Ministry (Top 6)')
        axes[1, 0].set_xlabel('Total Payroll (₪)')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Correlation with exogenous factors
    if 'DATE' in exogenous_df.columns and 'CPI_INDEX' in exogenous_df.columns:
        # Merge data for correlation analysis
        monthly_payroll = payroll_df.groupby('date')['MASKORET_BRUTO_HEFRESHIM'].sum().reset_index()
        exogenous_df['date'] = pd.to_datetime(exogenous_df['DATE'])

        merged_data = pd.merge(monthly_payroll, exogenous_df, on='date', how='inner')

        if len(merged_data) > 0:
            ax2 = axes[1, 1]
            ax2_twin = ax2.twinx()

            line1 = ax2.plot(merged_data['date'], merged_data['MASKORET_BRUTO_HEFRESHIM'],
                             'b-', linewidth=2, label='Payroll')
            line2 = ax2_twin.plot(merged_data['date'], merged_data['CPI_INDEX'],
                                  'r--', linewidth=2, label='CPI Index')

            ax2.set_title('Payroll vs CPI Index')
            ax2.set_ylabel('Total Payroll (₪)', color='b')
            ax2_twin.set_ylabel('CPI Index', color='r')
            ax2.tick_params(axis='x', rotation=45)

            # Calculate correlation
            correlation = merged_data['MASKORET_BRUTO_HEFRESHIM'].corr(merged_data['CPI_INDEX'])
            ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                     transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        else:
            axes[1, 1].text(0.5, 0.5, 'No overlapping data\nfor correlation analysis',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Payroll vs CPI Index')
    else:
        axes[1, 1].text(0.5, 0.5, 'Exogenous data not available\nfor correlation analysis',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Payroll vs Economic Indicators')

    plt.tight_layout()
    plt.show()

    # Additional analysis: Seasonal decomposition for main ministry
    perform_seasonal_decomposition(payroll_df)


def perform_seasonal_decomposition(payroll_df):
    """
    Perform seasonal decomposition analysis for the largest ministry
    """
    print("\n6. Performing seasonal decomposition analysis...")

    try:
        # Get the largest ministry by total payroll for decomposition
        ministry_totals = payroll_df.groupby('NAME_MISRAD_AVIV')['MASKORET_BRUTO_HEFRESHIM'].sum()
        largest_ministry = ministry_totals.idxmax()

        print(f"   • Selected ministry for decomposition: {largest_ministry}")

        # Filter data for the largest ministry
        ministry_data = payroll_df[payroll_df['NAME_MISRAD_AVIV'] == largest_ministry]

        if len(ministry_data) > 12:  # Need enough data for decomposition
            ministry_monthly = ministry_data.groupby('date')['MASKORET_BRUTO_HEFRESHIM'].sum().reset_index()
            ministry_monthly = ministry_monthly.sort_values('date').set_index('date')

            # Ensure we have enough periods for decomposition
            if len(ministry_monthly) >= 24:  # At least 2 years of data
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    ministry_monthly['MASKORET_BRUTO_HEFRESHIM'],
                    model='multiplicative',
                    period=12,
                    extrapolate_trend='freq'
                )

                # Plot decomposition
                fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                fig.suptitle(f'Seasonal Decomposition - {largest_ministry}', fontsize=14, fontweight='bold')

                decomposition.observed.plot(ax=axes[0], title='Original Data', color='navy')
                decomposition.trend.plot(ax=axes[1], title='Trend Component', color='green')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component', color='orange')
                decomposition.resid.plot(ax=axes[3], title='Residual Component', color='red')

                for ax in axes:
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

                # Print decomposition insights
                print(f"\n📊 Seasonal Decomposition Insights for {largest_ministry}:")

                # Trend analysis
                trend_values = decomposition.trend.dropna()
                if len(trend_values) > 1:
                    trend_change = ((trend_values.iloc[-1] / trend_values.iloc[0]) - 1) * 100
                    print(f"   • Overall trend change: {trend_change:.1f}%")

                # Seasonal strength
                seasonal_strength = (decomposition.seasonal.var() / decomposition.observed.var()) * 100
                print(f"   • Seasonal strength: {seasonal_strength:.1f}%")

                # Peak and low seasons
                seasonal_avg = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()
                peak_month = seasonal_avg.idxmax()
                low_month = seasonal_avg.idxmin()
                print(f"   • Peak seasonal month: {peak_month}")
                print(f"   • Low seasonal month: {low_month}")

            else:
                print(f"   ⚠️ Insufficient data for seasonal decomposition")
                print(f"   • Need at least 24 months, have {len(ministry_monthly)}")
        else:
            print(f"   ⚠️ No sufficient ministry data for seasonal decomposition")

    except Exception as e:
        print(f"   ⚠️ Could not perform seasonal decomposition: {e}")


def create_fallback_payroll_data():
    """Create fallback payroll data if import fails"""
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

            total_salary = int(base_salary * trend * seasonal * noise)

            data.append({
                'TARICH_SACHAR': date,
                'NAME_MISRAD_AVIV': ministry,
                'MASKORET_BRUTO_HEFRESHIM': total_salary
            })

    return pd.DataFrame(data)


def create_fallback_exogenous_data():
    """Create fallback exogenous data if import fails"""
    dates = pd.date_range('2019-01-01', '2024-06-01', freq='MS')
    data = []

    np.random.seed(42)  # For reproducibility

    cpi_base = 100
    for i, date in enumerate(dates):
        data.append({
            'DATE': date,
            'CPI_INDEX': cpi_base + i * 0.3 + np.random.normal(0, 0.5),
            'UNEMPLOYMENT_RATE': max(1, 4 + np.random.normal(0, 0.3)),
            'GDP_GROWTH_RATE': 3 + np.random.normal(0, 0.5)
        })

    return pd.DataFrame(data)


def run_trend_analysis():
    """
    Main function to run the complete trend and seasonality analysis
    """
    print("Loading data for trend analysis...")

    # Try to import data creation functions from Section A.1
    try:
        from section_a1_data_cleaning import create_mock_payroll_data, create_mock_exogenous_data

        payroll_df = create_mock_payroll_data()
        exogenous_df = create_mock_exogenous_data()
    except ImportError:
        print("⚠️ Could not import from section_a1_data_cleaning, creating data directly...")
        payroll_df = create_fallback_payroll_data()
        exogenous_df = create_fallback_exogenous_data()

    # Run the analysis
    results = analyze_trends_and_seasonality(payroll_df, exogenous_df)

    print("\n=== Summary of Findings ===")
    print(f"✓ Overall trend: {results['overall_trend']['average_yearly_growth']:.1f}% yearly growth")
    print(f"✓ Seasonality: Peak in month {results['seasonal_patterns']['peak_month']}")
    print(f"✓ Ministry analysis: {len(results['ministry_analysis']['ministry_growth'])} ministries analyzed")

    print("\n✅ Section A.2 completed successfully!")
    return results


if __name__ == "__main__":
    results = run_trend_analysis()