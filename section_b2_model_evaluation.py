#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section B.2: Model Evaluation and Assessment
=============================================
Government Payroll Forecasting Model - Data Science Test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import learning_curve, validation_curve
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and assessment
    """

    def __init__(self, models, predictions, y_true, feature_names=None):
        self.models = models
        self.predictions = predictions
        self.y_true = y_true
        self.feature_names = feature_names or []
        self.evaluation_results = {}

    def evaluate_model_accuracy(self):
        """
        Evaluate model accuracy using multiple metrics
        """
        print("=== Section B.2: Model Evaluation and Assessment ===\n")
        print("1. Evaluating model accuracy with comprehensive metrics...")

        evaluation_results = {}

        for model_name, y_pred in self.predictions.items():
            # Basic metrics
            rmse = np.sqrt(mean_squared_error(self.y_true, y_pred))
            mae = mean_absolute_error(self.y_true, y_pred)
            r2 = r2_score(self.y_true, y_pred)

            # MAPE calculation (handle division by zero)
            mape = np.mean(np.abs((self.y_true - y_pred) / np.where(self.y_true != 0, self.y_true, 1))) * 100

            # Additional metrics
            # Mean Bias Error
            mbe = np.mean(y_pred - self.y_true)

            # Normalized RMSE
            nrmse = rmse / (self.y_true.max() - self.y_true.min()) * 100

            # Symmetric MAPE (more robust)
            smape = np.mean(2 * np.abs(y_pred - self.y_true) / (np.abs(y_pred) + np.abs(self.y_true))) * 100

            # Prediction intervals (simple approach)
            residuals = y_pred - self.y_true
            prediction_std = np.std(residuals)

            evaluation_results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape,
                'SMAPE': smape,
                'MBE': mbe,
                'NRMSE': nrmse,
                'Prediction_Std': prediction_std,
                'Residuals': residuals
            }

            print(f"\n   📊 {model_name} Detailed Metrics:")
            print(f"   • RMSE: ₪{rmse:,.0f}")
            print(f"   • MAE: ₪{mae:,.0f}")
            print(f"   • R²: {r2:.3f}")
            print(f"   • MAPE: {mape:.1f}%")
            print(f"   • SMAPE: {smape:.1f}%")
            print(f"   • Mean Bias Error: ₪{mbe:,.0f}")
            print(f"   • Normalized RMSE: {nrmse:.1f}%")

        self.evaluation_results = evaluation_results
        return evaluation_results

    def analyze_model_strengths_weaknesses(self):
        """
        Analyze model strengths and weaknesses
        """
        print("\n2. Analyzing model strengths and weaknesses...")

        # Find best and worst performing models
        best_r2_model = max(self.evaluation_results.keys(),
                            key=lambda x: self.evaluation_results[x]['R²'])
        best_rmse_model = min(self.evaluation_results.keys(),
                              key=lambda x: self.evaluation_results[x]['RMSE'])

        print(f"\n   🏆 Best R² Score: {best_r2_model} ({self.evaluation_results[best_r2_model]['R²']:.3f})")
        print(f"   🎯 Best RMSE: {best_rmse_model} (₪{self.evaluation_results[best_rmse_model]['RMSE']:,.0f})")

        # Analyze each model's characteristics
        for model_name, metrics in self.evaluation_results.items():
            print(f"\n   🔍 {model_name} Analysis:")

            # Strengths
            strengths = []
            weaknesses = []

            if metrics['R²'] > 0.7:
                strengths.append("Good explanatory power (R² > 0.7)")
            elif metrics['R²'] < 0.5:
                weaknesses.append("Low explanatory power (R² < 0.5)")

            if metrics['MAPE'] < 10:
                strengths.append("Low prediction error (MAPE < 10%)")
            elif metrics['MAPE'] > 20:
                weaknesses.append("High prediction error (MAPE > 20%)")

            if abs(metrics['MBE']) < metrics['MAE'] * 0.1:
                strengths.append("Unbiased predictions")
            else:
                bias_direction = "over-predicting" if metrics['MBE'] > 0 else "under-predicting"
                weaknesses.append(f"Biased towards {bias_direction}")

            if metrics['NRMSE'] < 15:
                strengths.append("Good relative accuracy")
            elif metrics['NRMSE'] > 25:
                weaknesses.append("Poor relative accuracy")

            # Print analysis
            if strengths:
                print(f"     ✅ Strengths: {', '.join(strengths)}")
            if weaknesses:
                print(f"     ⚠️ Weaknesses: {', '.join(weaknesses)}")

    def create_comprehensive_visualizations(self):
        """
        Create comprehensive evaluation visualizations
        """
        print("\n3. Creating comprehensive evaluation visualizations...")

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')

        # Figure 1: Metrics comparison
        self._plot_metrics_comparison()

        # Figure 2: Residual analysis
        self._plot_residual_analysis()

        # Figure 3: Prediction accuracy analysis
        self._plot_prediction_accuracy()

        # Figure 4: Error distribution
        self._plot_error_distribution()

    def _plot_metrics_comparison(self):
        """
        Plot comprehensive metrics comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')

        model_names = list(self.evaluation_results.keys())

        # RMSE comparison
        rmse_values = [self.evaluation_results[model]['RMSE'] for model in model_names]
        axes[0, 0].bar(model_names, rmse_values, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Root Mean Square Error (RMSE)')
        axes[0, 0].set_ylabel('RMSE (₪)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # R² comparison
        r2_values = [self.evaluation_results[model]['R²'] for model in model_names]
        axes[0, 1].bar(model_names, r2_values, color=['lightgreen', 'orange'])
        axes[0, 1].set_title('R² Score')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)

        # MAPE comparison
        mape_values = [self.evaluation_results[model]['MAPE'] for model in model_names]
        axes[1, 0].bar(model_names, mape_values, color=['yellow', 'purple'])
        axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Bias analysis
        mbe_values = [self.evaluation_results[model]['MBE'] for model in model_names]
        colors = ['red' if x > 0 else 'blue' for x in mbe_values]
        axes[1, 1].bar(model_names, mbe_values, color=colors)
        axes[1, 1].set_title('Mean Bias Error (MBE)')
        axes[1, 1].set_ylabel('MBE (₪)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def _plot_residual_analysis(self):
        """
        Plot residual analysis for each model
        """
        n_models = len(self.predictions)
        fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')

        for i, (model_name, y_pred) in enumerate(self.predictions.items()):
            residuals = self.evaluation_results[model_name]['Residuals']

            # Residuals vs Predicted
            axes[0, i].scatter(y_pred, residuals, alpha=0.6)
            axes[0, i].axhline(y=0, color='red', linestyle='--')
            axes[0, i].set_title(f'{model_name}\nResiduals vs Predicted')
            axes[0, i].set_xlabel('Predicted Values (₪)')
            axes[0, i].set_ylabel('Residuals (₪)')

            # Q-Q plot (normal distribution check)
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, i])
            axes[1, i].set_title(f'{model_name}\nQ-Q Plot')

        plt.tight_layout()
        plt.show()

    def _plot_prediction_accuracy(self):
        """
        Plot prediction accuracy analysis
        """
        best_model = max(self.evaluation_results.keys(),
                         key=lambda x: self.evaluation_results[x]['R²'])
        best_predictions = self.predictions[best_model]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Prediction Accuracy Analysis - {best_model}', fontsize=16, fontweight='bold')

        # Actual vs Predicted scatter plot
        axes[0].scatter(self.y_true, best_predictions, alpha=0.6, color='green')

        # Perfect prediction line
        min_val = min(self.y_true.min(), best_predictions.min())
        max_val = max(self.y_true.max(), best_predictions.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Add confidence intervals
        residuals = best_predictions - self.y_true
        std_residual = np.std(residuals)
        axes[0].fill_between([min_val, max_val],
                             [min_val - 2 * std_residual, max_val - 2 * std_residual],
                             [min_val + 2 * std_residual, max_val + 2 * std_residual],
                             alpha=0.2, color='gray', label='95% Confidence Interval')

        axes[0].set_xlabel('Actual Values (₪)')
        axes[0].set_ylabel('Predicted Values (₪)')
        axes[0].set_title('Actual vs Predicted Values')
        axes[0].legend()

        # Time series plot (if we have time information)
        if hasattr(self, 'dates') and self.dates is not None:
            axes[1].plot(self.dates, self.y_true, label='Actual', linewidth=2)
            axes[1].plot(self.dates, best_predictions, label='Predicted', linewidth=2, linestyle='--')
            axes[1].set_title('Time Series: Actual vs Predicted')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Payroll (₪)')
            axes[1].legend()
            axes[1].tick_params(axis='x', rotation=45)
        else:
            # Alternative: prediction vs index
            indices = range(len(self.y_true))
            axes[1].plot(indices, self.y_true, label='Actual', linewidth=2)
            axes[1].plot(indices, best_predictions, label='Predicted', linewidth=2, linestyle='--')
            axes[1].set_title('Actual vs Predicted Over Sample Index')
            axes[1].set_xlabel('Sample Index')
            axes[1].set_ylabel('Payroll (₪)')
            axes[1].legend()

        plt.tight_layout()
        plt.show()

    def _plot_error_distribution(self):
        """
        Plot error distribution analysis
        """
        fig, axes = plt.subplots(1, len(self.predictions), figsize=(6 * len(self.predictions), 6))
        if len(self.predictions) == 1:
            axes = [axes]

        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')

        for i, (model_name, _) in enumerate(self.predictions.items()):
            residuals = self.evaluation_results[model_name]['Residuals']

            # Histogram of residuals
            axes[i].hist(residuals, bins=20, alpha=0.7, color='lightblue', density=True)

            # Overlay normal distribution
            mu, sigma = np.mean(residuals), np.std(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            normal_dist = ((1 / (sigma * np.sqrt(2 * np.pi))) *
                           np.exp(-0.5 * ((x - mu) / sigma) ** 2))
            axes[i].plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')

            axes[i].set_title(f'{model_name}\nResiduals Distribution')
            axes[i].set_xlabel('Residuals (₪)')
            axes[i].set_ylabel('Density')
            axes[i].legend()

            # Add statistics text
            axes[i].text(0.05, 0.95, f'Mean: ₪{mu:,.0f}\nStd: ₪{sigma:,.0f}',
                         transform=axes[i].transAxes,
                         bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5),
                         verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def generate_evaluation_report(self):
        """
        Generate a comprehensive evaluation report
        """
        print("\n4. Generating comprehensive evaluation report...")

        report = []
        report.append("=" * 60)
        report.append("         MODEL EVALUATION REPORT")
        report.append("=" * 60)

        # Overall summary
        best_model = max(self.evaluation_results.keys(),
                         key=lambda x: self.evaluation_results[x]['R²'])

        report.append(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
        report.append(f"   • R² Score: {self.evaluation_results[best_model]['R²']:.3f}")
        report.append(f"   • RMSE: ₪{self.evaluation_results[best_model]['RMSE']:,.0f}")
        report.append(f"   • MAPE: {self.evaluation_results[best_model]['MAPE']:.1f}%")

        # Detailed comparison
        report.append(f"\n📊 DETAILED COMPARISON:")
        report.append("-" * 60)

        # Create comparison table
        comparison_df = pd.DataFrame(self.evaluation_results).T
        comparison_df = comparison_df.round(3)

        report.append(comparison_df.to_string())

        # Model-specific insights
        report.append(f"\n🔍 MODEL-SPECIFIC INSIGHTS:")
        report.append("-" * 60)

        for model_name, metrics in self.evaluation_results.items():
            report.append(f"\n{model_name}:")

            # Performance category
            if metrics['R²'] > 0.8:
                performance = "Excellent"
            elif metrics['R²'] > 0.6:
                performance = "Good"
            elif metrics['R²'] > 0.4:
                performance = "Fair"
            else:
                performance = "Poor"

            report.append(f"   • Overall Performance: {performance}")
            report.append(f"   • Prediction Accuracy: {100 - metrics['MAPE']:.1f}%")

            # Bias analysis
            if abs(metrics['MBE']) < metrics['MAE'] * 0.1:
                bias_status = "Unbiased"
            elif metrics['MBE'] > 0:
                bias_status = "Tends to over-predict"
            else:
                bias_status = "Tends to under-predict"

            report.append(f"   • Bias Status: {bias_status}")

        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        report.append("-" * 60)

        if self.evaluation_results[best_model]['R²'] > 0.7:
            report.append("✅ Model shows good predictive performance")
            report.append("✅ Suitable for production use with proper monitoring")
        else:
            report.append("⚠️ Model performance needs improvement")
            report.append("⚠️ Consider additional features or alternative algorithms")

        if any(abs(metrics['MBE']) > metrics['MAE'] * 0.1 for metrics in self.evaluation_results.values()):
            report.append("⚠️ Address model bias through feature engineering or calibration")

        report.append("📈 Monitor model performance regularly")
        report.append("🔄 Retrain model periodically with new data")

        # Print report
        for line in report:
            print(line)

        return "\n".join(report)


# Main execution function
def run_model_evaluation(trained_model):
    """
    Run comprehensive model evaluation
    """
    print("Running comprehensive model evaluation...")

    if not hasattr(trained_model, 'predictions') or not trained_model.predictions:
        print("❌ No trained model predictions available")
        return None

    # Initialize evaluator
    evaluator = ModelEvaluator(
        models=trained_model.models,
        predictions=trained_model.predictions,
        y_true=trained_model.y_test,
        feature_names=trained_model.feature_names
    )

    # Run evaluation steps
    evaluation_results = evaluator.evaluate_model_accuracy()
    evaluator.analyze_model_strengths_weaknesses()
    evaluator.create_comprehensive_visualizations()
    evaluation_report = evaluator.generate_evaluation_report()

    print("\n✅ Section B.2 completed successfully!")

    return evaluator, evaluation_results, evaluation_report


if __name__ == "__main__":
    # This would typically be run after Section B.1
    print("Note: This module should be run after training models in Section B.1")
    print("Example usage:")
    print("from section_b1_basic_model import run_basic_model")
    print("from section_b2_model_evaluation import run_model_evaluation")
    print("")
    print("trained_model, _, _ = run_basic_model()")
    print("evaluator, results, report = run_model_evaluation(trained_model)")