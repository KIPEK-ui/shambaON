"""
Visualization Module for AI Models
===================================

Comprehensive visualization tools for model evaluation:
1. Confusion matrices
2. ROC curves
3. Feature importance
4. Mean reversion analysis
5. Model comparison
6. Performance dashboards
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import accuracy_score

# ML metrics
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, classification_report
)

# Configure logging
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = sns.color_palette("husl", 8)
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'


# =====================================================================
# CONFUSION MATRIX VISUALIZATION
# =====================================================================

class ConfusionMatrixVisualizer:
    """Generate confusion matrix visualizations."""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = 'Model',
                             labels: List[str] = None,
                             normalize: bool = True,
                             output_path: str = None) -> plt.Figure:
        """
        Create confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            labels: Label names (e.g., ['No Flood', 'Flood'])
            normalize: Whether to normalize counts
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            display_cm = cm_norm
            fmt = '.2%'
        else:
            display_cm = cm
            fmt = 'd'
        
        if labels is None:
            labels = [f'Class {i}' for i in range(cm.shape[0])]
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
        
        # Plot heatmap
        sns.heatmap(display_cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                   ax=ax, linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {model_name}\n(Normalized)' if normalize else
                    f'Confusion Matrix - {model_name}',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {output_path}")
        
        return fig
    
    @staticmethod
    def plot_multi_confusion_matrices(predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                     labels: List[str] = None,
                                     output_path: str = None) -> plt.Figure:
        """
        Create grid of confusion matrices for multiple models.
        
        Args:
            predictions_dict: Dict of {model_name: (y_true, y_pred)}
            labels: Label names
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        
        n_models = len(predictions_dict)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                                dpi=FIGURE_DPI)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        if labels is None:
            labels = ['No Flood', 'Flood']
        
        for idx, (model_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx], cbar=False, linewidths=0.5)
            
            axes[idx].set_title(f'{model_name}', fontweight='bold', fontsize=11)
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('True', fontsize=10)
        
        # Hide unused subplots
        for idx in range(len(predictions_dict), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Confusion Matrices - Multi-Model Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Multi-confusion matrix saved: {output_path}")
        
        return fig


# =====================================================================
# ROC CURVE VISUALIZATION
# =====================================================================

class ROCCurveVisualizer:
    """Generate ROC curve visualizations."""
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                      model_name: str = 'Model',
                      output_path: str = None) -> Tuple[plt.Figure, float]:
        """
        Plot ROC curve with AUC score.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            model_name: Name of the model
            output_path: Path to save figure
        
        Returns:
            Tuple of (figure, roc_auc_score)
        """
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='#2E86AB', lw=3,
               label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
               label='Random Classifier (AUC = 0.5000)')
        
        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"ROC curve saved: {output_path}")
        
        return fig, roc_auc
    
    @staticmethod
    def plot_multi_roc_curves(predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                             output_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for multiple models on same plot.
        
        Args:
            predictions_dict: Dict of {model_name: (y_true, y_score)}
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        
        fig, ax = plt.subplots(figsize=(12, 9), dpi=FIGURE_DPI)
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
               label='Random Classifier', alpha=0.7)
        
        colors = sns.color_palette("husl", len(predictions_dict))
        
        for (model_name, (y_true, y_score)), color in zip(predictions_dict.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2.5,
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Multi-Model Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Multi-ROC curves saved: {output_path}")
        
        return fig


# =====================================================================
# FEATURE IMPORTANCE VISUALIZATION
# =====================================================================

class FeatureImportanceVisualizer:
    """Generate feature importance visualizations."""
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str],
                               importances: np.ndarray,
                               model_name: str = 'Model',
                               top_n: int = 15,
                               output_path: str = None) -> plt.Figure:
        """
        Plot feature importance as horizontal bar chart.
        
        Args:
            feature_names: Names of features
            importances: Importance scores
            model_name: Name of the model
            top_n: Number of top features to show
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        fig, ax = plt.subplots(figsize=(12, max(6, top_n/2.5)), dpi=FIGURE_DPI)
        
        # Create bar plot
        colors = sns.color_palette("viridis", len(sorted_importances))
        bars = ax.barh(range(len(sorted_importances)), sorted_importances, color=colors)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        ax.set_yticks(range(len(sorted_importances)))
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Feature importance saved: {output_path}")
        
        return fig


# =====================================================================
# MEAN REVERSION ANALYSIS
# =====================================================================

class MeanReversionAnalyzer:
    """Analyze and visualize mean reversion in flood risk predictions."""
    
    @staticmethod
    def calculate_mean_reversion_metrics(y_true: np.ndarray,
                                        y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate mean reversion metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
        
        Returns:
            Dictionary of mean reversion metrics
        """
        
        # Calculate deviations from mean
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        
        # Deviation from mean
        dev_true = y_true - mean_true
        dev_pred = y_pred - mean_pred
        
        # Mean reversion ratio (how much predictions revert to mean)
        with np.errstate(divide='ignore', invalid='ignore'):
            reversion_ratio = np.mean(dev_pred ** 2) / np.mean(dev_true ** 2)
            reversion_ratio = np.nan_to_num(reversion_ratio, nan=1.0)
        
        # Calculate autocorrelation for mean reversion tendency
        autocorr = np.corrcoef(y_pred[:-1], y_pred[1:])[0, 1]
        
        # Standard metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'mean_true': float(mean_true),
            'mean_pred': float(mean_pred),
            'reversion_ratio': float(reversion_ratio),
            'autocorrelation': float(np.nan_to_num(autocorr, nan=0.0)),
            'rmse': float(rmse),
            'mae': float(mae)
        }
    
    @staticmethod
    def plot_mean_reversion_analysis(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    model_name: str = 'Model',
                                    output_path: str = None) -> plt.Figure:
        """
        Create comprehensive mean reversion analysis plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            model_name: Name of the model
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        
        metrics = MeanReversionAnalyzer.calculate_mean_reversion_metrics(y_true, y_pred)
        
        fig = plt.figure(figsize=(15, 10), dpi=FIGURE_DPI)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Time series plot
        ax1 = fig.add_subplot(gs[0, :])
        x_axis = np.arange(len(y_true))
        ax1.plot(x_axis, y_true, label='True Values', linewidth=2, alpha=0.7, color='#2E86AB')
        ax1.plot(x_axis, y_pred, label='Predictions', linewidth=2, alpha=0.7, color='#A23B72')
        ax1.axhline(y=metrics['mean_true'], color='#2E86AB', linestyle='--', linewidth=1.5, alpha=0.5, label='True Mean')
        ax1.axhline(y=metrics['mean_pred'], color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.5, label='Pred Mean')
        ax1.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Flood Risk Score', fontsize=11, fontweight='bold')
        ax1.set_title(f'Actual vs Predicted Values - {model_name}', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Deviation from mean
        ax2 = fig.add_subplot(gs[1, 0])
        dev_true = y_true - metrics['mean_true']
        dev_pred = y_pred - metrics['mean_pred']
        ax2.scatter(dev_true, dev_pred, alpha=0.6, s=50, color='#F18F01')
        
        # Add regression line
        z = np.polyfit(dev_true, dev_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(dev_true.min(), dev_true.max(), 100)
        ax2.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Fit line')
        
        # Add diagonal
        lims = [
            np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()]),
        ]
        ax2.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
        
        ax2.set_xlabel('True Deviation from Mean', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted Deviation from Mean', fontsize=11, fontweight='bold')
        ax2.set_title('Mean Reversion Analysis', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Metrics display
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        metrics_text = f"""
        MEAN REVERSION METRICS
        {'='*50}
        
        True Mean:           {metrics['mean_true']:.4f}
        Predicted Mean:      {metrics['mean_pred']:.4f}
        
        Reversion Ratio:     {metrics['reversion_ratio']:.4f}
        (1.0 = full variance, <1 = reversion to mean)
        
        Autocorrelation:     {metrics['autocorrelation']:.4f}
        
        PREDICTION ERRORS
        {'='*50}
        RMSE:                {metrics['rmse']:.4f}
        MAE:                 {metrics['mae']:.4f}
        
        INTERPRETATION
        {'='*50}
        • Reversion Ratio < 1.0 → Predictions revert to mean
        • Autocorr > 0 → Positive serial correlation
        • Lower errors indicate better predictions
        """
        
        ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f'Mean Reversion Analysis - {model_name}',
                    fontsize=15, fontweight='bold', y=0.995)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Mean reversion analysis saved: {output_path}")
        
        return fig


# =====================================================================
# MODEL PERFORMANCE DASHBOARD
# =====================================================================

class ModelPerformanceDashboard:
    """Create comprehensive model performance dashboards."""
    
    @staticmethod
    def create_dashboard(predictions_dict: Dict[str, Dict[str, Any]],
                        output_path: str = None) -> plt.Figure:
        """
        Create comprehensive performance dashboard for multiple models.
        
        Args:
            predictions_dict: Dict of model metrics and predictions
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        
        n_models = len(predictions_dict)
        fig = plt.figure(figsize=(16, 10), dpi=FIGURE_DPI)
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        colors = sns.color_palette("husl", n_models)
        
        # Extract metrics
        model_names = list(predictions_dict.keys())
        roc_aucs = [predictions_dict[m].get('roc_auc', 0) for m in model_names]
        f1_scores = [predictions_dict[m].get('f1', 0) for m in model_names]
        precisions = [predictions_dict[m].get('precision', 0) for m in model_names]
        recalls = [predictions_dict[m].get('recall', 0) for m in model_names]
        
        # 1. ROC-AUC comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(model_names, roc_aucs, color=colors, alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, roc_aucs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        ax1.set_ylabel('ROC-AUC', fontweight='bold')
        ax1.set_title('ROC-AUC Score', fontweight='bold', fontsize=11)
        ax1.set_ylim([0, 1.1])
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. F1-Score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(model_names, f1_scores, color=colors, alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        ax2.set_ylabel('F1-Score', fontweight='bold')
        ax2.set_title('F1-Score', fontweight='bold', fontsize=11)
        ax2.set_ylim([0, 1.1])
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Precision vs Recall
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(recalls, precisions, s=200, c=range(n_models), cmap='tab10', 
                   alpha=0.7, edgecolors='black', linewidth=2)
        for i, name in enumerate(model_names):
            ax3.annotate(name, (recalls[i], precisions[i]), 
                        fontsize=8, ha='center')
        ax3.set_xlabel('Recall', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('Precision vs Recall', fontweight='bold', fontsize=11)
        ax3.set_xlim([0, 1.05])
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics comparison radar
        ax4 = fig.add_subplot(gs[1, :], projection='polar')
        categories = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        for idx, model_name in enumerate(model_names):
            values = [roc_aucs[idx], f1_scores[idx], precisions[idx], recalls[idx]]
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax4.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.set_title('Multi-Metric Comparison', fontweight='bold', fontsize=12, pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax4.grid(True)
        
        # 5. Metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        table_data = []
        for model_name in model_names:
            table_data.append([
                model_name,
                f"{predictions_dict[model_name].get('roc_auc', 0):.4f}",
                f"{predictions_dict[model_name].get('f1', 0):.4f}",
                f"{predictions_dict[model_name].get('precision', 0):.4f}",
                f"{predictions_dict[model_name].get('recall', 0):.4f}",
            ])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Model', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall'],
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Performance dashboard saved: {output_path}")
        
        return fig


class MetricsVisualizer:
    """Visualize individual metrics across all models."""
    
    @staticmethod
    def create_metrics_charts(predictions_dict: Dict[str, Dict],
                             output_path: str = None) -> plt.Figure:
        """Create individual metric charts for ROC-AUC, Recall, Accuracy, F1-Score."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        model_names = list(predictions_dict.keys())
        metrics = {
            'roc_auc': 'ROC-AUC Score',
            'recall': 'Recall Score',
            'accuracy': 'Accuracy Score',
            'f1': 'F1-Score'
        }
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for idx, (metric_key, metric_label) in enumerate(metrics.items()):
            ax = axes[idx // 2, idx % 2]
            
            # Extract metric values
            values = []
            for model_name in model_names:
                value = predictions_dict[model_name].get(metric_key, 0)
                values.append(value)
            
            # Create bar chart
            bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Formatting
            ax.set_ylabel(metric_label, fontweight='bold', fontsize=11)
            ax.set_ylim([0, 1.1])
            ax.set_title(metric_label, fontweight='bold', fontsize=12, pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Metrics charts saved: {output_path}")
        
        return fig


# =====================================================================
# MAIN VISUALIZATION ORCHESTRATOR
# =====================================================================

def generate_all_visualizations(models_dir: str = 'models',
                               output_dir: str = 'models/visualizations',
                               X_val: np.ndarray = None,
                               y_val: np.ndarray = None,
                               feature_names: List[str] = None) -> Dict[str, str]:
    """
    Generate all model visualizations.
    
    Args:
        models_dir: Directory containing trained models
        output_dir: Directory to save visualizations
        X_val: Validation features
        y_val: Validation labels
        feature_names: Feature names
    
    Returns:
        Dictionary of output file paths
    """
    
    logger.info("=" * 70)
    logger.info("GENERATING MODEL VISUALIZATIONS")
    logger.info("=" * 70)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_files = {}
    
    try:
        # Load saved models
        with open(f'{models_dir}/flood_risk_models.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        classifiers = artifacts['classifiers']
        preprocessor = artifacts['preprocessor']
        results = artifacts['results']
        feature_names = feature_names or artifacts.get('feature_names', None)
        
        logger.info(f"Loaded models from {models_dir}")
        
        # Generate predictions dictionary for all models
        predictions_dict = {}
        confusion_dict = {}
        
        for model_name, model in classifiers.models.items():
            if model and X_val is not None and y_val is not None:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                predictions_dict[model_name] = {
                    'y_true': y_val,
                    'y_score': y_pred_proba
                }
                confusion_dict[model_name] = (y_val, y_pred)
        
        # 1. Generate confusion matrices
        logger.info("Generating confusion matrices...")
        cm_viz = ConfusionMatrixVisualizer()
        
        # Individual confusion matrices
        for model_name, (y_true, y_pred) in confusion_dict.items():
            fig = cm_viz.plot_confusion_matrix(
                y_true, y_pred, model_name=model_name,
                labels=['No Flood', 'Flood'],
                output_path=f'{output_dir}/confusion_matrix_{model_name}.png'
            )
            output_files[f'confusion_{model_name}'] = f'{output_dir}/confusion_matrix_{model_name}.png'
            plt.close(fig)
        
        # Multi-model confusion matrices
        fig = cm_viz.plot_multi_confusion_matrices(
            confusion_dict,
            labels=['No Flood', 'Flood'],
            output_path=f'{output_dir}/confusion_matrices_comparison.png'
        )
        output_files['confusion_comparison'] = f'{output_dir}/confusion_matrices_comparison.png'
        plt.close(fig)
        
        # 2. Generate ROC curves
        logger.info("Generating ROC curves...")
        roc_viz = ROCCurveVisualizer()
        
        for model_name, pred_data in predictions_dict.items():
            try:
                y_true = pred_data.get('y_true')
                y_score = pred_data.get('y_score')
                
                if y_true is not None and y_score is not None:
                    # Ensure they are numpy arrays
                    y_true = np.asarray(y_true)
                    y_score = np.asarray(y_score)
                    
                    fig, roc_auc = roc_viz.plot_roc_curve(
                        y_true, y_score, model_name=model_name,
                        output_path=f'{output_dir}/roc_curve_{model_name}.png'
                    )
                    output_files[f'roc_{model_name}'] = f'{output_dir}/roc_curve_{model_name}.png'
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not generate ROC curve for {model_name}: {str(e)}")
        
        # Multi-model ROC curves
        try:
            # Convert predictions_dict to proper format for multi-roc
            multi_roc_data = {}
            for model_name, pred_data in predictions_dict.items():
                y_true = pred_data.get('y_true')
                y_score = pred_data.get('y_score')
                if y_true is not None and y_score is not None:
                    multi_roc_data[model_name] = (np.asarray(y_true), np.asarray(y_score))
            
            if multi_roc_data:
                fig = roc_viz.plot_multi_roc_curves(
                    multi_roc_data,
                    output_path=f'{output_dir}/roc_curves_comparison.png'
                )
                output_files['roc_comparison'] = f'{output_dir}/roc_curves_comparison.png'
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate multi-ROC curves: {str(e)}")
        
        # 3. Generate feature importance
        logger.info("Generating feature importance plots...")
        feat_viz = FeatureImportanceVisualizer()
        
        for model_name, model in classifiers.models.items():
            if hasattr(model, 'feature_importances_') and feature_names:
                fig = feat_viz.plot_feature_importance(
                    feature_names, model.feature_importances_,
                    model_name=model_name, top_n=15,
                    output_path=f'{output_dir}/feature_importance_{model_name}.png'
                )
                output_files[f'features_{model_name}'] = f'{output_dir}/feature_importance_{model_name}.png'
                plt.close(fig)
        
        # 4. Generate mean reversion analysis
        logger.info("Generating mean reversion analysis...")
        mr_analyzer = MeanReversionAnalyzer()
        
        for model_name, pred_data in predictions_dict.items():
            try:
                y_true = pred_data.get('y_true')
                y_score = pred_data.get('y_score')
                
                if y_true is not None and y_score is not None:
                    # Ensure they are numpy arrays
                    y_true = np.asarray(y_true)
                    y_score = np.asarray(y_score)
                    
                    fig = mr_analyzer.plot_mean_reversion_analysis(
                        y_true, y_score, model_name=model_name,
                        output_path=f'{output_dir}/mean_reversion_{model_name}.png'
                    )
                    output_files[f'mean_reversion_{model_name}'] = f'{output_dir}/mean_reversion_{model_name}.png'
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not generate mean reversion analysis for {model_name}: {str(e)}")
        
        # 5. Generate performance dashboard
        logger.info("Generating performance dashboard...")
        dashboard = ModelPerformanceDashboard()
        
        fig = dashboard.create_dashboard(
            results,
            output_path=f'{output_dir}/performance_dashboard.png'
        )
        output_files['dashboard'] = f'{output_dir}/performance_dashboard.png'
        plt.close(fig)
        
        # 6. Generate individual metrics charts
        logger.info("Generating individual metrics charts...")
        metrics_viz = MetricsVisualizer()
        
        fig = metrics_viz.create_metrics_charts(
            predictions_dict,
            output_path=f'{output_dir}/metrics_charts.png'
        )
        output_files['metrics_charts'] = f'{output_dir}/metrics_charts.png'
        plt.close(fig)
        
        # Save summary
        logger.info(f"\n{'='*70}")
        logger.info("VISUALIZATION GENERATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Generated {len(output_files)} visualizations:")
        for key, path in output_files.items():
            logger.info(f"  ✓ {key}: {path}")
        
        # Save manifest
        manifest_path = f'{output_dir}/visualization_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(output_files, f, indent=2)
        logger.info(f"Manifest saved: {manifest_path}")
        
        return output_files
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
        return {}


if __name__ == '__main__':
    generate_all_visualizations()
