"""
Flood Risk Prediction Models
=============================

This module implements multiple approaches to flood risk prediction:
1. Time-series forecasting (ARIMA, Prophet)
2. Classification models (Logistic Regression, Random Forest, XGBoost)
3. Hybrid ensemble models combining statistical and ML approaches
4. Geospatial models (CNN for satellite imagery)
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Any, List

import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================

FLOOD_PRONE_COUNTIES = [
    'Tana River', 'Kilifi', 'Mombasa', 'Lamu', 'Siaya', 'Kisumu',
    'Homa Bay', 'Migori', 'Garissa', 'Isiolo', 'Baringo'
]

RISK_THRESHOLDS = {
    'Low': (0.0, 0.33),
    'Medium': (0.33, 0.67),
    'High': (0.67, 1.0)
}


# =====================================================================
# DATA PREPARATION & FEATURE ENGINEERING
# =====================================================================

class FloodRiskDataPreprocessor:
    """Preprocess and engineer features for flood risk prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        logger.info("FloodRiskDataPreprocessor initialized")
    
    def load_and_prepare_data(self, climate_file: str, forecast_file: str, 
                             farmer_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and merge climate, forecast, and farmer data.
        
        Args:
            climate_file: Path to climate/hydrology CSV
            forecast_file: Path to flood forecasts CSV
            farmer_file: Path to farmer profiles CSV
        
        Returns:
            Tuple of (features_df, target_df)
        """
        logger.info("Loading input data...")
        
        # Load datasets
        climate_df = pd.read_csv(climate_file)
        forecast_df = pd.read_csv(forecast_file)
        farmer_df = pd.read_csv(farmer_file)
        
        # Create binary flood risk target (High/Critical = 1, Low/Medium = 0)
        forecast_df['flood_risk_binary'] = (forecast_df['risk_class'] == 'High').astype(int)
        
        logger.info(f"Loaded climate: {len(climate_df)} records")
        logger.info(f"Loaded forecasts: {len(forecast_df)} records")
        logger.info(f"Loaded farmers: {len(farmer_df)} records")
        
        # Aggregate climate features by parcel
        features_df = self._aggregate_climate_features(climate_df, forecast_df)
        
        # Ensure county column exists in features_df
        if 'county' not in features_df.columns:
            features_df['county'] = 'Unknown'
        
        # Merge with farmer characteristics (exclude county to avoid duplication)
        farmer_cols = [col for col in ['soil_type', 'soil_ph', 'irrigation_availability', 
                                       'historical_flood_events'] 
                      if col in farmer_df.columns]
        
        if farmer_cols:
            features_df = features_df.merge(
                farmer_df[['parcel_id'] + farmer_cols],
                on='parcel_id',
                how='left'
            )
        
        # Ensure county column exists and is not null
        if 'county' not in features_df.columns:
            features_df['county'] = 'Unknown'
        features_df['county'] = features_df['county'].fillna('Unknown')
        
        # Add county-level flood risk factor
        features_df['is_flood_prone_county'] = features_df['county'].isin(FLOOD_PRONE_COUNTIES).astype(int)
        
        logger.info(f"Created feature set: {len(features_df)} parcels")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        
        return features_df, forecast_df[['parcel_id', 'flood_risk_binary', 'flood_risk_score', 'risk_class']]
    
    def _aggregate_climate_features(self, climate_df: pd.DataFrame, 
                                   forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate climate data by parcel and time window."""
        
        # Map climate records to parcels via county/coordinates
        climate_features = []
        
        # Ensure climate_df has county column (may have different name)
        if 'county' not in climate_df.columns and 'County' in climate_df.columns:
            climate_df['county'] = climate_df['County']
        
        for parcel_id in forecast_df['parcel_id'].unique():
            forecast_row = forecast_df[forecast_df['parcel_id'] == parcel_id].iloc[0]
            
            # Get county from forecast data
            parcel_county = forecast_row.get('county') or forecast_row.get('County', 'Unknown')
            
            # Get recent climate data (past 30 days)
            if 'county' in climate_df.columns:
                recent_climate = climate_df[
                    (climate_df['county'] == parcel_county)
                ].tail(100)  # Last 100 records per county
            else:
                # Fallback: use all data if county column not available
                recent_climate = climate_df.tail(100)
            
            if len(recent_climate) == 0:
                # Still create a feature record with defaults
                features = {
                    'parcel_id': parcel_id,
                    'county': parcel_county,
                    'rainfall_mean': 0,
                    'rainfall_std': 0,
                    'rainfall_max': 0,
                    'rainfall_total': 0,
                    'river_level_mean': 0,
                    'river_level_max': 0,
                    'soil_moisture_mean': 50,
                    'soil_moisture_max': 70,
                }
                climate_features.append(features)
                continue
            
            # Extract rainfall statistics - handle different element naming
            rainfall_data = recent_climate[recent_climate['element'].str.contains('rainfall', case=False, na=False)]['value']
            river_data = recent_climate[recent_climate['element'].str.contains('river', case=False, na=False)]['value']
            soil_moist_data = recent_climate[recent_climate['element'].str.contains('soil_moisture', case=False, na=False)]['value']
            
            features = {
                'parcel_id': parcel_id,
                'county': parcel_county,
                'rainfall_mean': rainfall_data.mean() if len(rainfall_data) > 0 else 0,
                'rainfall_std': rainfall_data.std() if len(rainfall_data) > 0 else 0,
                'rainfall_max': rainfall_data.max() if len(rainfall_data) > 0 else 0,
                'rainfall_total': rainfall_data.sum() if len(rainfall_data) > 0 else 0,
                'river_level_mean': river_data.mean() if len(river_data) > 0 else 0,
                'river_level_max': river_data.max() if len(river_data) > 0 else 0,
                'soil_moisture_mean': soil_moist_data.mean() if len(soil_moist_data) > 0 else 50,
                'soil_moisture_max': soil_moist_data.max() if len(soil_moist_data) > 0 else 70,
            }
            
            climate_features.append(features)
        
        result_df = pd.DataFrame(climate_features)
        
        # Ensure county column exists
        if 'county' not in result_df.columns:
            result_df['county'] = 'Unknown'
        
        return result_df
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Engineer advanced features for modeling.
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders/scalers (True for training)
        
        Returns:
            Feature-engineered dataframe
        """
        df = df.copy()
        
        # Polynomial features
        df['rainfall_river_interaction'] = df['rainfall_max'] * df['river_level_max']
        df['rainfall_soil_interaction'] = df['rainfall_total'] * df['soil_moisture_mean']
        df['river_soil_interaction'] = df['river_level_max'] * df['soil_moisture_max']
        
        # Ratio features
        df['rainfall_variability'] = df['rainfall_std'] / (df['rainfall_mean'] + 1e-5)
        df['soil_saturation_ratio'] = df['soil_moisture_max'] / 100.0
        
        # Historical flood risk
        df['historical_flood_events'] = df['historical_flood_events'].fillna(0)
        df['flood_history_normalized'] = df['historical_flood_events'] / (df['historical_flood_events'].max() + 1e-5)
        
        # Soil characteristics risk
        df['soil_type'] = df['soil_type'].fillna('unknown')
        high_risk_soils = ['clay', 'clay loam']
        df['high_risk_soil'] = df['soil_type'].isin(high_risk_soils).astype(int)
        
        # Encode categorical variables
        if fit:
            self.encoders['soil_type'] = LabelEncoder()
            df['soil_type_encoded'] = self.encoders['soil_type'].fit_transform(df['soil_type'])
        else:
            df['soil_type_encoded'] = self.encoders['soil_type'].transform(df['soil_type'])
        
        # Handle missing values
        df = df.fillna(0)
        
        # Feature list for tracking
        self.feature_names = [col for col in df.columns if col not in 
                            ['parcel_id', 'county', 'soil_type', 'irrigation_availability']]
        
        logger.info(f"Engineered {len(self.feature_names)} features")
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale numerical features."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.scalers['features'] = StandardScaler()
            scaled = self.scalers['features'].fit_transform(df[numeric_cols])
        else:
            scaled = self.scalers['features'].transform(df[numeric_cols])
        
        return scaled


# =====================================================================
# CLASSIFICATION MODELS
# =====================================================================

class FloodRiskClassificationModels:
    """Collection of classification models for flood risk prediction."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        logger.info("FloodRiskClassificationModels initialized")
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        
        self.models['logistic_regression'] = model
        
        logger.info(f"Logistic Regression - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        metrics['feature_importance'] = model.feature_importances_
        
        self.models['random_forest'] = model
        
        logger.info(f"Random Forest - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        metrics['feature_importance'] = model.feature_importances_
        
        self.models['gradient_boosting'] = model
        
        logger.info(f"Gradient Boosting - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost model (if available)."""
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed, skipping...")
            return {}
        
        logger.info("Training XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            scale_pos_weight=1,
            tree_method='hist'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        metrics['feature_importance'] = model.feature_importances_
        
        self.models['xgboost'] = model
        
        logger.info(f"XGBoost - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM model (if available)."""
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not installed, skipping...")
            return {}
        
        logger.info("Training LightGBM...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            valid_names=['validation']
        )
        
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        y_pred_proba = model.predict(X_val)
        
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        metrics['feature_importance'] = model.feature_importance()
        
        self.models['lightgbm'] = model
        
        logger.info(f"LightGBM - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        return {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }


# =====================================================================
# ENSEMBLE MODEL
# =====================================================================

class FloodRiskEnsembleModel:
    """Hybrid ensemble combining multiple models."""
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.weights = None
        logger.info("FloodRiskEnsembleModel initialized")
    
    def predict_ensemble(self, X: np.ndarray, method: str = 'average') -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Input features
            method: 'average', 'weighted', or 'voting'
        
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            return predictions.mean(axis=0)
        elif method == 'weighted':
            if self.weights is None:
                self.weights = np.ones(len(self.models)) / len(self.models)
            return np.average(predictions, axis=0, weights=self.weights)
        else:  # voting
            return (predictions > 0.5).mean(axis=0)


# =====================================================================
# TIME-SERIES MODELS
# =====================================================================

class FloodRiskTimeSeriesModels:
    """Time-series forecasting models for rainfall and river levels."""
    
    def __init__(self):
        self.arima_models = {}
        self.prophet_models = {}
        logger.info("FloodRiskTimeSeriesModels initialized")
    
    def train_arima(self, ts_data: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """Train ARIMA model."""
        if not HAS_STATSMODELS:
            logger.warning("statsmodels not installed, skipping ARIMA...")
            return {}
        
        logger.info(f"Training ARIMA{order}...")
        
        try:
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            self.arima_models['arima'] = fitted_model
            
            # Get diagnostics
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            logger.info(f"ARIMA trained - AIC: {aic:.2f}, BIC: {bic:.2f}")
            
            return {
                'model': fitted_model,
                'aic': aic,
                'bic': bic
            }
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return {}
    
    def train_prophet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model."""
        if not HAS_PROPHET:
            logger.warning("Prophet not installed, skipping...")
            return {}
        
        logger.info("Training Prophet...")
        
        try:
            # Prophet expects 'ds' and 'y' columns
            prophet_df = df.rename(columns={'timestamp': 'ds', 'value': 'y'})
            
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)
            
            self.prophet_models['prophet'] = model
            
            logger.info("Prophet trained successfully")
            
            return {'model': model}
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            return {}
    
    def forecast_arima(self, steps: int = 30) -> np.ndarray:
        """Generate ARIMA forecasts."""
        if 'arima' not in self.arima_models:
            return np.array([])
        
        forecast = self.arima_models['arima'].get_forecast(steps=steps)
        return forecast.predicted_mean.values
    
    def forecast_prophet(self, periods: int = 30) -> np.ndarray:
        """Generate Prophet forecasts."""
        if 'prophet' not in self.prophet_models:
            return np.array([])
        
        future = self.prophet_models['prophet'].make_future_dataframe(periods=periods, freq='D')
        forecast = self.prophet_models['prophet'].predict(future)
        return forecast['yhat'].tail(periods).values


# =====================================================================
# MAIN TRAINING PIPELINE
# =====================================================================

def train_flood_risk_models(data_dir: str = 'data', output_dir: str = 'models') -> Dict[str, Any]:
    """
    Train all flood risk prediction models.
    
    Args:
        data_dir: Directory containing generated data
        output_dir: Directory to save trained models
    
    Returns:
        Dictionary with all trained models and metrics
    """
    
    logger.info("=" * 70)
    logger.info("FLOOD RISK PREDICTION MODEL TRAINING")
    logger.info("=" * 70)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Data preparation
    preprocessor = FloodRiskDataPreprocessor()
    
    features_df, target_df = preprocessor.load_and_prepare_data(
        f'{data_dir}/climate_hydrology_environmental.csv',
        f'{data_dir}/flood_risk_forecasts.csv',
        f'{data_dir}/farmer_profiles.csv'
    )
    
    # Merge features with target
    data = features_df.merge(target_df, on='parcel_id', how='left')
    data = data.dropna(subset=['flood_risk_binary'])
    
    logger.info(f"Dataset size: {len(data)} samples")
    logger.info(f"Positive class rate: {data['flood_risk_binary'].mean():.2%}")
    
    # 2. Feature engineering
    data = preprocessor.engineer_features(data, fit=True)
    
    # 3. Train-test split (time-series aware)
    X = preprocessor.scale_features(data, fit=True)
    y = data['flood_risk_binary'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # 4. Train classification models
    results = {}
    classifiers = FloodRiskClassificationModels()
    
    results['logistic_regression'] = classifiers.train_logistic_regression(X_train, y_train, X_val, y_val)
    results['random_forest'] = classifiers.train_random_forest(X_train, y_train, X_val, y_val)
    results['gradient_boosting'] = classifiers.train_gradient_boosting(X_train, y_train, X_val, y_val)
    
    if HAS_XGBOOST:
        results['xgboost'] = classifiers.train_xgboost(X_train, y_train, X_val, y_val)
    
    if HAS_LIGHTGBM:
        results['lightgbm'] = classifiers.train_lightgbm(X_train, y_train, X_val, y_val)
    
    # 5. Ensemble model
    ensemble = FloodRiskEnsembleModel(classifiers.models)
    ensemble_pred = ensemble.predict_ensemble(X_val, method='average')
    ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
    results['ensemble'] = {
        'roc_auc': roc_auc_score(y_val, ensemble_pred),
        'f1': f1_score(y_val, ensemble_pred_binary),
        'precision': precision_score(y_val, ensemble_pred_binary),
        'recall': recall_score(y_val, ensemble_pred_binary)
    }
    
    logger.info(f"Ensemble - ROC-AUC: {results['ensemble']['roc_auc']:.4f}, F1: {results['ensemble']['f1']:.4f}")
    
    # 6. Save models and artifacts
    artifacts = {
        'classifiers': classifiers,
        'ensemble': ensemble,
        'preprocessor': preprocessor,
        'results': results,
        'feature_names': preprocessor.feature_names
    }
    
    # Save with pickle
    with open(f'{output_dir}/flood_risk_models.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    logger.info(f"Models saved to {output_dir}/flood_risk_models.pkl")
    
    # 7. Summary report
    logger.info("\n" + "=" * 70)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    for model_name, metrics in results.items():
        if metrics:
            logger.info(f"{model_name.upper()}")
            logger.info(f"  ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
            logger.info(f"  F1-Score: {metrics.get('f1', 0):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall:   {metrics.get('recall', 0):.4f}")
    logger.info("=" * 70 + "\n")
    
    return artifacts


if __name__ == '__main__':
    train_flood_risk_models()
