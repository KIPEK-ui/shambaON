"""
Explainable AI (XAI) Components
================================

This module provides interpretability for predictions:
- SHAP values for feature importance
- LIME for local explanations
- Feature importance visualization
- Model transparency reports
"""

import os
import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger_warning = "SHAP not installed"

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
# SHAP EXPLAINER
# =====================================================================

class SHAPExplainer:
    """SHAP-based feature importance and prediction explanations."""
    
    def __init__(self):
        logger.info("SHAPExplainer initialized")
        if not HAS_SHAP:
            logger.warning("SHAP not installed - feature importance will use model-native methods")
    
    def explain_predictions(self, model: Any, X: np.ndarray, 
                           feature_names: List[str] = None,
                           num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model: Trained model (tree-based preferred)
            X: Feature matrix
            feature_names: Names of features
            num_samples: Number of samples for explanation
        
        Returns:
            Dictionary with SHAP values and analysis
        """
        
        if not HAS_SHAP:
            logger.warning("SHAP not available, using model feature importance instead")
            return self._fallback_importance(model, feature_names)
        
        try:
            logger.info("Computing SHAP values...")
            
            # Use TreeExplainer for tree-based models
            if hasattr(model, 'estimators_'):  # RandomForest, GradientBoosting
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X[:num_samples])
            
            shap_values = explainer.shap_values(X)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Binary classification - use positive class
            
            # Feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Sort by importance
            importance_idx = np.argsort(feature_importance)[::-1]
            
            results = {
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'feature_importance': feature_importance.tolist(),
                'features_ranked': [
                    {
                        'rank': i + 1,
                        'feature': feature_names[idx] if feature_names else f'Feature_{idx}',
                        'importance': float(feature_importance[idx])
                    }
                    for i, idx in enumerate(importance_idx[:10])
                ]
            }
            
            logger.info("SHAP analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return self._fallback_importance(model, feature_names)
    
    def _fallback_importance(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Fallback to model's native feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            idx = np.argsort(importances)[::-1]
            
            return {
                'method': 'Model Native Feature Importance',
                'features_ranked': [
                    {
                        'rank': i + 1,
                        'feature': feature_names[j] if feature_names else f'Feature_{j}',
                        'importance': float(importances[j])
                    }
                    for i, j in enumerate(idx[:10])
                ]
            }
        
        return {'error': 'No feature importance available'}
    
    def explain_single_prediction(self, model: Any, X_instance: np.ndarray,
                                 feature_names: List[str] = None,
                                 background_data: np.ndarray = None) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP.
        
        Args:
            model: Trained model
            X_instance: Single instance to explain
            feature_names: Feature names
            background_data: Background dataset for KernelExplainer
        
        Returns:
            Explanation for single prediction
        """
        
        if not HAS_SHAP:
            logger.warning("SHAP not available")
            return {}
        
        try:
            if background_data is None:
                background_data = X_instance
            
            if hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, background_data)
            
            shap_values = explainer.shap_values(X_instance.reshape(1, -1))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            shap_values = shap_values[0]
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(X_instance.reshape(1, -1))[0]
                pred_value = prediction[1]
                pred_class = 'High Risk' if pred_value > 0.5 else 'Low Risk'
            else:
                pred_value = model.predict(X_instance.reshape(1, -1))[0]
                pred_class = 'High Risk' if pred_value > 0.5 else 'Low Risk'
            
            # Top contributors
            top_idx = np.argsort(np.abs(shap_values))[::-1][:5]
            
            contributors = [
                {
                    'feature': feature_names[i] if feature_names else f'Feature_{i}',
                    'value': X_instance[i],
                    'shap_value': float(shap_values[i]),
                    'direction': 'Increases Risk' if shap_values[i] > 0 else 'Decreases Risk'
                }
                for i in top_idx
            ]
            
            return {
                'prediction': pred_class,
                'confidence': float(pred_value),
                'top_contributors': contributors,
                'explanation': f"Prediction: {pred_class} with {pred_value:.1%} confidence. "
                               f"Key factors: {', '.join([c['feature'] for c in contributors[:3]])}"
            }
            
        except Exception as e:
            logger.error(f"Single prediction explanation failed: {e}")
            return {}


# =====================================================================
# LIME EXPLAINER
# =====================================================================

class LIMEExplainer:
    """LIME-based local explanations for predictions."""
    
    def __init__(self, X_train: np.ndarray, feature_names: List[str] = None,
                 class_names: List[str] = None):
        """
        Initialize LIME explainer.
        
        Args:
            X_train: Training data for LIME
            feature_names: Names of features
            class_names: Names of classes
        """
        
        self.X_train = X_train
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names or ['Low Risk', 'High Risk']
        
        if HAS_LIME:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )
            logger.info("LIMEExplainer initialized")
        else:
            logger.warning("LIME not installed")
    
    def explain_prediction(self, model: Any, X_instance: np.ndarray,
                          num_features: int = 5) -> Dict[str, Any]:
        """
        Explain prediction using LIME.
        
        Args:
            model: Trained model with predict_proba method
            X_instance: Instance to explain
            num_features: Number of features to highlight
        
        Returns:
            Local explanation
        """
        
        if not HAS_LIME:
            logger.warning("LIME not available")
            return {}
        
        try:
            # LIME explanation
            exp = self.explainer.explain_instance(
                X_instance,
                model.predict_proba,
                num_features=num_features,
                top_labels=1
            )
            
            # Extract feature weights
            exp_list = exp.as_list(label=1)
            
            explanation = {
                'instance_features': X_instance.tolist(),
                'local_explanation': [
                    {
                        'feature': feat.split('<=')[0].strip() if '<=' in feat else feat,
                        'effect': 'Increases Risk' if weight > 0 else 'Decreases Risk',
                        'weight': abs(weight)
                    }
                    for feat, weight in exp_list
                ],
                'prediction_score': model.predict_proba(X_instance.reshape(1, -1))[0][1]
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {}


# =====================================================================
# MODEL TRANSPARENCY REPORT
# =====================================================================

class ModelTransparencyReport:
    """Generate comprehensive model transparency and fairness reports."""
    
    def __init__(self):
        logger.info("ModelTransparencyReport initialized")
    
    def generate_report(self, model: Any, X: np.ndarray, y: np.ndarray,
                       feature_names: List[str] = None,
                       model_name: str = 'Model') -> Dict[str, Any]:
        """
        Generate comprehensive transparency report.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            feature_names: Names of features
            model_name: Name of model
        
        Returns:
            Comprehensive report dictionary
        """
        
        logger.info(f"Generating transparency report for {model_name}...")
        
        report = {
            'model_name': model_name,
            'generated_at': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'num_samples': len(X),
                'num_features': X.shape[1],
                'target_distribution': self._analyze_distribution(y)
            },
            'feature_info': self._analyze_features(X, feature_names),
            'bias_analysis': self._analyze_potential_bias(model, X, y),
            'recommendations': self._generate_recommendations(model)
        }
        
        return report
    
    def _analyze_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze target distribution."""
        unique, counts = np.unique(y, return_counts=True)
        
        return {
            'class_distribution': {str(c): int(cnt) for c, cnt in zip(unique, counts)},
            'class_balance': 'Balanced' if 0.3 < min(counts)/sum(counts) < 0.7 else 'Imbalanced'
        }
    
    def _analyze_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature statistics."""
        
        # Handle mismatch between feature_names and X shape
        num_features = X.shape[1]
        feature_names = feature_names or [f'Feature_{i}' for i in range(num_features)]
        
        # Trim feature_names if it has more names than X has columns
        if len(feature_names) > num_features:
            feature_names = feature_names[:num_features]
        
        # Pad feature_names if it has fewer names than X has columns
        elif len(feature_names) < num_features:
            feature_names = list(feature_names) + [f'Feature_{i}' for i in range(len(feature_names), num_features)]
        
        features = []
        for i, name in enumerate(feature_names):
            if i >= num_features:
                break
            col = X[:, i]
            features.append({
                'name': name,
                'mean': float(np.mean(col)),
                'std': float(np.std(col)),
                'min': float(np.min(col)),
                'max': float(np.max(col)),
                'missing_pct': float(np.sum(np.isnan(col)) / len(col) * 100)
            })
        
        return features
    
    def _analyze_potential_bias(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze potential biases in model."""
        
        bias_analysis = {
            'notes': 'Bias analysis requires demographic information',
            'recommendations': [
                'Stratified analysis by gender and farm size',
                'Compare predictions across demographic groups',
                'Monitor for disparate impact'
            ]
        }
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
        
        # Overall metrics
        bias_analysis['overall_prediction_rate'] = float(np.mean(y_pred))
        bias_analysis['model_preference'] = 'High Risk predictions' if np.mean(y_pred) > 0.5 else 'Low Risk predictions'
        
        return bias_analysis
    
    def _generate_recommendations(self, model: Any) -> List[str]:
        """Generate recommendations for model improvement."""
        
        recommendations = [
            'Regularly retrain model with new data',
            'Monitor predictions for drift',
            'Use ensemble models for robustness',
            'Validate across different counties and seasons',
            'Collect feedback from farmers on recommendations'
        ]
        
        if hasattr(model, 'feature_importances_'):
            recommendations.append('Review top features for domain relevance')
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save report to JSON file."""
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def generate_xai_summary(models_dict: Dict[str, Any], X_val: np.ndarray,
                        feature_names: List[str],
                        output_dir: str = 'models') -> Dict[str, Any]:
    """
    Generate XAI summaries for all models.
    
    Args:
        models_dict: Dictionary of trained models
        X_val: Validation feature matrix
        feature_names: Names of features
        output_dir: Output directory for reports
    
    Returns:
        Summary dictionary
    """
    
    logger.info("Generating XAI summaries...")
    Path(output_dir).mkdir(exist_ok=True)
    
    xai_summary = {}
    shap_exp = SHAPExplainer()
    
    for model_name, model in models_dict.items():
        if model is None:
            continue
        
        logger.info(f"  Explaining {model_name}...")
        
        # SHAP analysis
        shap_result = shap_exp.explain_predictions(model, X_val, feature_names)
        
        xai_summary[model_name] = {
            'shap_analysis': shap_result
        }
    
    # Save XAI summaries
    with open(f'{output_dir}/xai_summary.json', 'w') as f:
        json.dump(xai_summary, f, indent=2, default=str)
    
    logger.info(f"XAI summaries saved to {output_dir}/xai_summary.json")
    
    return xai_summary


if __name__ == '__main__':
    logger.info("XAI module ready for use")
