"""
Inference & Prediction Module
==============================

Provides interfaces for making predictions:
- Flood risk predictions for parcels
- Crop recommendations
- Batch prediction
- Real-time inference API
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
# FLOOD RISK PREDICTOR
# =====================================================================

class FloodRiskPredictor:
    """Make flood risk predictions for parcels."""
    
    def __init__(self, model_path: str = 'models/flood_risk_models.pkl'):
        """
        Load trained flood risk models.
        
        Args:
            model_path: Path to pickled models
        """
        self.model_path = model_path
        self.artifacts = None
        self.load_models()
    
    def load_models(self):
        """Load models from pickle file."""
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            return
        
        with open(self.model_path, 'rb') as f:
            self.artifacts = pickle.load(f)
        
        logger.info(f"Loaded flood risk models from {self.model_path}")
    
    def predict(self, features: Dict[str, float], use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Predict flood risk for a parcel.
        
        Args:
            features: Dictionary with parcel features
            use_ensemble: Whether to use ensemble model
        
        Returns:
            Prediction with confidence and explanation
        """
        
        if self.artifacts is None:
            return {'error': 'Models not loaded'}
        
        try:
            # Convert features to array
            X = self._prepare_features(features)
            
            if X is None:
                return {'error': 'Invalid features'}
            
            # Get predictions
            if use_ensemble:
                prediction = self._predict_ensemble(X)
            else:
                prediction = self._predict_best_model(X)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model input."""
        
        try:
            # Extract required features in order
            feature_values = [
                features.get('rainfall_mean', 0),
                features.get('rainfall_std', 0),
                features.get('rainfall_max', 0),
                features.get('rainfall_total', 0),
                features.get('river_level_mean', 0),
                features.get('river_level_max', 0),
                features.get('soil_moisture_mean', 0),
                features.get('soil_moisture_max', 0),
                features.get('soil_ph', 6.5),
                features.get('historical_flood_events', 0),
                features.get('is_flood_prone_county', 0),
                features.get('high_risk_soil', 0),
                features.get('irrigation_availability', 0),
            ]
            
            X = np.array([feature_values], dtype=float)
            
            # Scale features
            if hasattr(self.artifacts['preprocessor'], 'scalers'):
                scaler = self.artifacts['preprocessor'].scalers.get('features')
                if scaler:
                    X = scaler.transform(X)
            
            return X
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None
    
    def _predict_ensemble(self, X: np.ndarray) -> Dict[str, Any]:
        """Get ensemble prediction."""
        
        ensemble = self.artifacts.get('ensemble')
        if ensemble is None:
            return self._predict_best_model(X)
        
        # Get ensemble probability
        ensemble_pred = ensemble.predict_ensemble(X, method='average')[0]
        
        # Get individual model predictions for comparison
        individual_preds = {}
        classifiers = self.artifacts.get('classifiers')
        
        if classifiers:
            for name, model in classifiers.models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0][1]
                    individual_preds[name] = float(prob)
        
        # Risk classification
        risk_class = self._classify_risk(ensemble_pred)
        
        return {
            'flood_risk_score': float(ensemble_pred),
            'risk_class': risk_class,
            'confidence': float(abs(ensemble_pred - 0.5) * 2),  # 0-1 scale
            'individual_models': individual_preds,
            'prediction_time': datetime.now().isoformat(),
            'recommendation': self._get_recommendation(ensemble_pred, risk_class)
        }
    
    def _predict_best_model(self, X: np.ndarray) -> Dict[str, Any]:
        """Use best performing model (Ensemble or Gradient Boosting)."""
        
        classifiers = self.artifacts.get('classifiers')
        
        # Try ensemble first, then gradient boosting
        model = classifiers.models.get('ensemble') or classifiers.models.get('gradient_boosting')
        
        if model is None:
            return {'error': 'No model available'}
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0][1]
        else:
            prob = model.predict(X)[0]
        
        risk_class = self._classify_risk(prob)
        
        return {
            'flood_risk_score': float(prob),
            'risk_class': risk_class,
            'confidence': float(abs(prob - 0.5) * 2),
            'prediction_time': datetime.now().isoformat(),
            'recommendation': self._get_recommendation(prob, risk_class)
        }
    
    def _classify_risk(self, score: float) -> str:
        """Classify risk level from score."""
        
        if score < 0.33:
            return 'Low'
        elif score < 0.67:
            return 'Medium'
        else:
            return 'High'
    
    def _get_recommendation(self, score: float, risk_class: str) -> str:
        """Generate recommendation based on prediction."""
        
        if risk_class == 'High':
            return 'High flood risk detected. Prioritize drainage, plant flood-tolerant crops (arrowroot, rice), secure livestock.'
        elif risk_class == 'Medium':
            return 'Moderate flood risk. Prepare drainage systems, diversify crop selection, monitor weather alerts.'
        else:
            return 'Low flood risk. Standard management practices. Continue crop diversification for resilience.'


# =====================================================================
# CROP RECOMMENDATION ENGINE
# =====================================================================

class CropRecommender:
    """Generate crop recommendations."""
    
    def __init__(self, model_path: str = 'models/crop_recommendation_models.pkl'):
        """
        Load trained crop recommendation models.
        
        Args:
            model_path: Path to pickled models
        """
        self.model_path = model_path
        self.artifacts = None
        self.load_models()
    
    def load_models(self):
        """Load models from pickle file."""
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            return
        
        with open(self.model_path, 'rb') as f:
            self.artifacts = pickle.load(f)
        
        logger.info(f"Loaded crop recommendation models from {self.model_path}")
    
    def recommend(self, parcel_data: Dict[str, Any], 
                 flood_risk_score: float = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate crop recommendations for a parcel.
        
        Args:
            parcel_data: Dictionary with parcel characteristics
            flood_risk_score: Flood risk score (0-1) if available
        
        Returns:
            Recommendations from multiple models
        """
        
        if self.artifacts is None:
            return {'error': 'Models not loaded'}
        
        try:
            recommendations = {}
            
            # Knowledge graph recommendations
            kg = self.artifacts.get('knowledge_graph')
            if kg:
                # Determine flood risk class
                if flood_risk_score is not None:
                    if flood_risk_score < 0.33:
                        flood_risk = 'Low'
                    elif flood_risk_score < 0.67:
                        flood_risk = 'Medium'
                    else:
                        flood_risk = 'High'
                else:
                    flood_risk = parcel_data.get('flood_risk_class', 'Medium')
                
                # Rule-based recommendations
                kg_recs = kg.recommend_by_flood_tolerance(flood_risk, top_n=3)
                recommendations['knowledge_graph'] = kg_recs
                
                # Soil-based recommendations
                soil_recs = kg.recommend_by_soil_type(
                    parcel_data.get('soil_type', 'loam'), top_n=3
                )
                recommendations['soil_based'] = soil_recs
                
                # pH-based recommendations
                ph_recs = kg.recommend_by_soil_ph(
                    parcel_data.get('soil_ph', 6.5), top_n=3
                )
                recommendations['ph_based'] = ph_recs
            
            # Multi-objective optimization
            mo = self.artifacts.get('multi_objective')
            if mo:
                drought_risk = parcel_data.get('drought_risk_class', 'Medium')
                mo_recs = mo.recommend(
                    flood_risk if 'flood_risk' in locals() else 'Medium',
                    drought_risk,
                    top_n=3
                )
                recommendations['optimized'] = mo_recs
            
            # Consolidated recommendations (top across all models)
            all_crops = {}
            for model_recs in recommendations.values():
                if isinstance(model_recs, list):
                    for rec in model_recs:
                        crop_name = rec.get('crop', 'Unknown')
                        if crop_name not in all_crops:
                            all_crops[crop_name] = 0
                        all_crops[crop_name] += 1
            
            consolidated = sorted(
                [{'crop': crop, 'votes': count} for crop, count in all_crops.items()],
                key=lambda x: x['votes'],
                reverse=True
            )[:3]
            
            recommendations['consolidated'] = consolidated
            
            return {
                'parcel_id': parcel_data.get('parcel_id', 'unknown'),
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return {'error': str(e)}


# =====================================================================
# BATCH PREDICTION
# =====================================================================

class BatchPredictor:
    """Batch predictions for multiple parcels."""
    
    def __init__(self):
        self.flood_predictor = FloodRiskPredictor()
        self.crop_recommender = CropRecommender()
    
    def predict_batch(self, parcels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for multiple parcels.
        
        Args:
            parcels_df: DataFrame with parcel data
        
        Returns:
            DataFrame with predictions
        """
        
        logger.info(f"Running batch predictions for {len(parcels_df)} parcels...")
        
        results = []
        
        for idx, row in parcels_df.iterrows():
            try:
                # Prepare features
                features = row.to_dict()
                
                # Flood risk prediction
                flood_pred = self.flood_predictor.predict(features)
                
                # Crop recommendation
                crop_rec = self.crop_recommender.recommend(
                    features,
                    flood_risk_score=flood_pred.get('flood_risk_score')
                )
                
                # Consolidate result
                result = {
                    'parcel_id': features.get('parcel_id', idx),
                    'county': features.get('county', 'Unknown'),
                    'flood_risk_score': flood_pred.get('flood_risk_score'),
                    'flood_risk_class': flood_pred.get('risk_class'),
                    'top_recommended_crop': crop_rec.get('recommendations', {}).get('consolidated', [{}])[0].get('crop'),
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch prediction failed for parcel {idx}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        logger.info(f"Batch predictions completed: {len(results_df)} successful")
        
        return results_df


# =====================================================================
# REAL-TIME INFERENCE API
# =====================================================================

class InferenceAPI:
    """Real-time inference API for ShambaON system."""
    
    def __init__(self):
        self.flood_predictor = FloodRiskPredictor()
        self.crop_recommender = CropRecommender()
        logger.info("InferenceAPI initialized")
    
    def predict_parcel(self, parcel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete prediction for a parcel.
        
        Args:
            parcel_data: Parcel features
        
        Returns:
            Complete prediction with flood risk and crop recommendations
        """
        
        # Flood risk
        flood_pred = self.flood_predictor.predict(parcel_data)
        
        # Crop recommendations
        crop_recs = self.crop_recommender.recommend(
            parcel_data,
            flood_risk_score=flood_pred.get('flood_risk_score')
        )
        
        # Combined result
        result = {
            'parcel_id': parcel_data.get('parcel_id', 'unknown'),
            'county': parcel_data.get('county'),
            'prediction': {
                'flood_risk': flood_pred,
                'crop_recommendations': crop_recs
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def health_check(self) -> Dict[str, bool]:
        """Check if models are loaded."""
        
        return {
            'flood_risk_model_loaded': self.flood_predictor.artifacts is not None,
            'crop_recommendation_model_loaded': self.crop_recommender.artifacts is not None,
            'status': 'ready'
        }


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

def example_predictions():
    """Example predictions."""
    
    logger.info("Running example predictions...")
    
    # Initialize API
    api = InferenceAPI()
    
    # Example parcel
    parcel = {
        'parcel_id': 'parcel_001',
        'county': 'Tana River',
        'rainfall_mean': 45.0,
        'rainfall_std': 25.0,
        'rainfall_max': 120.0,
        'rainfall_total': 850.0,
        'river_level_mean': 2.5,
        'river_level_max': 5.5,
        'soil_moisture_mean': 55.0,
        'soil_moisture_max': 85.0,
        'soil_ph': 6.5,
        'soil_type': 'clay loam',
        'historical_flood_events': 3,
        'is_flood_prone_county': 1,
        'irrigation_availability': 1,
        'high_risk_soil': 0
    }
    
    # Get prediction
    result = api.predict_parcel(parcel)
    
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE PREDICTION RESULT")
    logger.info("=" * 80)
    logger.info(f"Parcel ID: {result['parcel_id']}")
    logger.info(f"County: {result['county']}")
    logger.info(f"Flood Risk: {result['prediction']['flood_risk'].get('risk_class')} "
               f"(score: {result['prediction']['flood_risk'].get('flood_risk_score'):.3f})")
    logger.info(f"Recommendation: {result['prediction']['flood_risk'].get('recommendation')}")
    logger.info("=" * 80 + "\n")
    
    return result


if __name__ == '__main__':
    example_predictions()
