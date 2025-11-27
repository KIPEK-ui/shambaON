"""
Crop Recommendation Models
===========================

This module implements crop recommendation systems:
1. Decision tree-based recommenders (Random Forest, Gradient Boosting)
2. Multi-objective optimization balancing yield, resilience, and market demand
3. Knowledge graph / rule-based systems with agro-ecological practices
4. Transfer learning approaches for Kenyan contexts
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# CONSTANTS & RULES
# =====================================================================

FLOOD_TOLERANCE_PRIORITY = {
    'High': 3,
    'Medium': 2,
    'Low': 1
}

DROUGHT_TOLERANCE_PRIORITY = {
    'High': 3,
    'Medium': 2,
    'Low': 1
}

# Agro-ecological zone classifications
AEZ_FLOOD_RISK = {
    'LH1': 'Low', 'LH2': 'Low', 'LH3': 'Low',
    'UM1': 'Low', 'UM2': 'Medium', 'UM3': 'High',
    'LM1': 'Medium', 'LM2': 'High', 'LM3': 'High',
    'AL1': 'Very High', 'AL2': 'Very High'
}

SOIL_CROP_COMPATIBILITY = {
    'clay': ['rice', 'arrowroot', 'beans', 'maize'],
    'sandy': ['millet', 'cassava', 'groundnuts', 'watermelon'],
    'loam': ['maize', 'beans', 'sorghum', 'cassava'],
    'clay loam': ['maize', 'beans', 'rice', 'arrowroot'],
    'sandy loam': ['millet', 'sorghum', 'cassava', 'groundnuts'],
    'silt loam': ['rice', 'maize', 'beans', 'cassava']
}

CROP_SOIL_PH_RANGE = {
    'Sorghum': (5.5, 8.0),
    'Millet': (5.0, 7.5),
    'Arrowroot': (5.5, 7.5),
    'Rice': (5.5, 8.0),
    'Maize': (5.5, 8.0),
    'Beans': (5.5, 7.5),
    'Cassava': (4.5, 7.0),
    'Finger Millet': (5.0, 7.5)
}

CROP_TEMPERATURE_RANGE = {
    'Sorghum': (20, 40),
    'Millet': (20, 35),
    'Arrowroot': (20, 30),
    'Rice': (20, 32),
    'Maize': (18, 30),
    'Beans': (15, 30),
    'Cassava': (18, 32),
    'Finger Millet': (18, 30)
}


# =====================================================================
# KNOWLEDGE GRAPH / RULE-BASED SYSTEM
# =====================================================================

class CropEcologicalKnowledgeGraph:
    """Knowledge graph encoding agro-ecological practices and rules."""
    
    def __init__(self):
        self.rules = {}
        self.practices = {}
        logger.info("CropEcologicalKnowledgeGraph initialized")
    
    def load_crop_catalog(self, crop_file: str) -> pd.DataFrame:
        """Load and index crop catalog."""
        self.crop_df = pd.read_csv(crop_file)
        logger.info(f"Loaded {len(self.crop_df)} crop records")
        return self.crop_df
    
    def recommend_by_flood_tolerance(self, flood_risk_class: str, 
                                     top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend crops based on flood risk using rules.
        
        Args:
            flood_risk_class: 'Low', 'Medium', 'High'
            top_n: Number of recommendations
        
        Returns:
            List of crop recommendations
        """
        if flood_risk_class == 'High':
            preferred_tolerance = 'High'
            acceptable_tolerance = 'Medium'
        elif flood_risk_class == 'Medium':
            preferred_tolerance = 'Medium'
            acceptable_tolerance = 'Low'
        else:
            preferred_tolerance = 'Low'
            acceptable_tolerance = 'Medium'
        
        # Filter crops
        suitable_crops = self.crop_df[
            (self.crop_df['flood_tolerance_class'].isin([preferred_tolerance, acceptable_tolerance]))
        ].copy()
        
        # Score by tolerance match
        suitable_crops['tolerance_score'] = suitable_crops['flood_tolerance_class'].apply(
            lambda x: FLOOD_TOLERANCE_PRIORITY.get(x, 1)
        )
        
        recommendations = suitable_crops.nlargest(top_n, 'tolerance_score')[
            ['crop', 'crop_id', 'flood_tolerance_class', 'yield_potential_t_ha', 
             'soil_preference', 'market_notes']
        ].to_dict('records')
        
        return recommendations
    
    def recommend_by_soil_type(self, soil_type: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend crops based on soil type using compatibility rules.
        
        Args:
            soil_type: Soil type classification
            top_n: Number of recommendations
        
        Returns:
            List of crop recommendations
        """
        compatible_crops = SOIL_CROP_COMPATIBILITY.get(soil_type, [])
        
        if not compatible_crops:
            logger.warning(f"No soil compatibility rules for {soil_type}")
            return []
        
        suitable_crops = self.crop_df[
            self.crop_df['crop'].str.lower().isin(compatible_crops)
        ].copy()
        
        recommendations = suitable_crops.head(top_n)[
            ['crop', 'crop_id', 'soil_preference', 'yield_potential_t_ha']
        ].to_dict('records')
        
        return recommendations
    
    def recommend_by_soil_ph(self, soil_ph: float, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend crops based on soil pH tolerance.
        
        Args:
            soil_ph: Soil pH value
            top_n: Number of recommendations
        
        Returns:
            List of crop recommendations
        """
        suitable_crops = []
        
        for crop, (min_ph, max_ph) in CROP_SOIL_PH_RANGE.items():
            if min_ph <= soil_ph <= max_ph:
                crop_record = self.crop_df[self.crop_df['crop'] == crop]
                if not crop_record.empty:
                    suitable_crops.append({
                        'crop': crop,
                        'crop_id': crop_record['crop_id'].values[0],
                        'compatibility': 'Optimal' if abs(soil_ph - (min_ph + max_ph) / 2) < 0.5 else 'Suitable'
                    })
        
        return suitable_crops[:top_n]
    
    def get_agroecological_practices(self, crop: str, aez_code: str) -> Dict[str, Any]:
        """
        Get recommended practices for a crop in a specific AEZ.
        
        Args:
            crop: Crop name
            aez_code: Agroecological zone code
        
        Returns:
            Dictionary with practices and guidance
        """
        crop_aez = self.crop_df[
            (self.crop_df['crop'] == crop) & 
            (self.crop_df['agroecological_zone'] == aez_code)
        ]
        
        if crop_aez.empty:
            return {}
        
        record = crop_aez.iloc[0]
        
        try:
            practices = json.loads(record['agroecological_zone_practices'])
        except:
            practices = []
        
        return {
            'crop': crop,
            'aez': aez_code,
            'practices': practices,
            'sowing_period': f"{record['early_sowing_month']}-{record['later_sowing_month']}",
            'growing_period': f"{record['growing_period_value']} {record['growing_period_period']}",
            'market_notes': record['market_notes']
        }


# =====================================================================
# DECISION TREE-BASED RECOMMENDER
# =====================================================================

class CropRecommendationDecisionTreeModel:
    """Multi-output decision tree recommender for multiple crops."""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.crop_names = []
        self.scaler = StandardScaler()
        logger.info("CropRecommendationDecisionTreeModel initialized")
    
    def prepare_data(self, farmer_df: str, crop_df: str, 
                     recommendations_df: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for crop recommendation.
        
        Args:
            farmer_df: Path to farmer profiles
            crop_df: Path to crop catalog
            recommendations_df: Path to recommendations
        
        Returns:
            Tuple of (X, y) for training
        """
        logger.info("Preparing crop recommendation data...")
        
        # Load data
        farmers = pd.read_csv(farmer_df)
        crops = pd.read_csv(crop_df)
        recommendations = pd.read_csv(recommendations_df)
        
        # Create training samples
        X_data = []
        y_data = []
        
        for _, rec in recommendations.iterrows():
            parcel_id = rec['parcel_id']
            recommended_crop = rec['recommended_crop_name']
            
            # Get farmer/parcel info
            farmer = farmers[farmers['parcel_id'] == parcel_id]
            
            if farmer.empty:
                continue
            
            farmer = farmer.iloc[0]
            
            # Extract soil pH (or use default)
            soil_ph = farmer.get('soil_ph', 6.5) if isinstance(farmer.get('soil_ph'), (int, float)) else 6.5
            if pd.isna(soil_ph):
                soil_ph = 6.5
            
            # Get irrigation availability
            irrigation = 1 if farmer.get('irrigation_availability') else 0
            
            # Create feature vector
            X_row = [
                soil_ph,
                irrigation,
                1 if farmer.get('county') in ['Tana River', 'Kilifi', 'Siaya'] else 0,  # flood prone
                farmer.get('historical_flood_events', 0) if isinstance(farmer.get('historical_flood_events'), (int, float)) else 0,
                1 if farmer.get('parcel_area_ha', 0) > 1 else 0 if isinstance(farmer.get('parcel_area_ha', 0), (int, float)) else 0
            ]
            
            X_data.append(X_row)
            
            # Create target (crops recommended)
            y_row = [1 if crop.lower() == recommended_crop.lower() else 0 
                    for crop in ['Sorghum', 'Millet', 'Arrowroot', 'Rice', 'Maize', 'Beans', 'Cassava', 'Finger Millet']]
            y_data.append(y_row)
            
            if recommended_crop not in self.crop_names:
                self.crop_names.append(recommended_crop)
        
        X = np.array(X_data, dtype=float)
        y = np.array(y_data, dtype=int)
        
        logger.info(f"Created training set: {len(X)} samples, {len(self.crop_names)} unique crops")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train multi-output decision tree model."""
        logger.info("Training crop recommendation model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train multi-output classifier
        self.model = MultiOutputClassifier(
            GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        hamming = hamming_loss(y_val, y_pred)
        
        logger.info(f"Model trained - Accuracy: {accuracy:.4f}, Hamming Loss: {hamming:.4f}")
        
        return {
            'accuracy': accuracy,
            'hamming_loss': hamming
        }
    
    def recommend(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate crop recommendations for a parcel.
        
        Args:
            features: Dictionary with parcel features (soil_ph, irrigation, etc.)
        
        Returns:
            List of recommended crops with scores
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return []
        
        # Prepare feature vector
        X = np.array([[
            features.get('soil_ph', 6.5),
            features.get('irrigation', 0),
            features.get('flood_prone', 0),
            features.get('historical_floods', 0),
            features.get('large_parcel', 0)
        ]], dtype=float)
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict_proba(X_scaled)
        
        recommendations = []
        crops = ['Sorghum', 'Millet', 'Arrowroot', 'Rice', 'Maize', 'Beans', 'Cassava', 'Finger Millet']
        
        for i, crop in enumerate(crops):
            if i < len(predictions):
                # Get probability from the i-th estimator
                proba = predictions[i][0, 1] if hasattr(predictions[i], 'shape') else predictions[i][1]
                recommendations.append({
                    'crop': crop,
                    'confidence': float(proba)
                })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations[:3]


# =====================================================================
# MULTI-OBJECTIVE OPTIMIZATION
# =====================================================================

class CropRecommendationMultiObjective:
    """Multi-objective optimizer balancing yield, resilience, and market demand."""
    
    def __init__(self, crop_df: str):
        self.crops = pd.read_csv(crop_df)
        logger.info("CropRecommendationMultiObjective initialized")
    
    def score_crop(self, crop_name: str, flood_risk: str, drought_risk: str,
                   market_demand: str = 'Medium') -> float:
        """
        Calculate multi-objective score for a crop.
        
        Args:
            crop_name: Crop to evaluate
            flood_risk: 'Low', 'Medium', 'High'
            drought_risk: 'Low', 'Medium', 'High'
            market_demand: 'Low', 'Medium', 'High'
        
        Returns:
            Composite score (0-100)
        """
        crop = self.crops[self.crops['crop'] == crop_name]
        if crop.empty:
            return 0
        
        crop = crop.iloc[0]
        
        # Parse yield potential
        try:
            yield_parts = str(crop.get('yield_potential_t_ha', '1.0-5.0')).split('-')
            yield_score = (float(yield_parts[0]) + float(yield_parts[1])) / 2 / 10
        except:
            yield_score = 0.5
        
        # Flood tolerance score
        flood_score = 0
        crop_flood_tol = crop.get('flood_tolerance_class', 'Low')
        if flood_risk == 'High' and crop_flood_tol == 'High':
            flood_score = 1.0
        elif flood_risk == 'High' and crop_flood_tol == 'Medium':
            flood_score = 0.6
        elif flood_risk == 'Medium' and crop_flood_tol in ['Medium', 'High']:
            flood_score = 0.9
        elif flood_risk == 'Low':
            flood_score = 0.8
        
        # Drought tolerance score
        drought_score = 0
        crop_drought_tol = crop.get('drought_tolerance_class', 'Low')
        if drought_risk == 'High' and crop_drought_tol == 'High':
            drought_score = 1.0
        elif drought_risk == 'High' and crop_drought_tol == 'Medium':
            drought_score = 0.6
        elif drought_risk == 'Medium' and crop_drought_tol in ['Medium', 'High']:
            drought_score = 0.9
        elif drought_risk == 'Low':
            drought_score = 0.8
        
        # Market demand score
        market_score = 0.8 if market_demand == 'High' else 0.6 if market_demand == 'Medium' else 0.4
        
        # Composite score with weights
        composite = (
            yield_score * 0.3 +           # Yield potential
            flood_score * 0.35 +          # Flood resilience (highest priority)
            drought_score * 0.20 +        # Drought resilience
            market_score * 0.15           # Market demand
        ) * 100
        
        return round(composite, 2)
    
    def recommend(self, flood_risk: str, drought_risk: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get top crop recommendations based on multi-objective optimization.
        
        Args:
            flood_risk: Current flood risk level
            drought_risk: Current drought risk level
            top_n: Number of recommendations
        
        Returns:
            Sorted list of recommended crops with scores
        """
        recommendations = []
        
        for crop in self.crops['crop'].unique():
            score = self.score_crop(crop, flood_risk, drought_risk)
            recommendations.append({
                'crop': crop,
                'score': score,
                'flood_risk': flood_risk,
                'drought_risk': drought_risk
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_n]


# =====================================================================
# MAIN TRAINING PIPELINE
# =====================================================================

def train_crop_recommendation_models(data_dir: str = 'data', 
                                     output_dir: str = 'models') -> Dict[str, Any]:
    """
    Train all crop recommendation models.
    
    Args:
        data_dir: Directory containing generated data
        output_dir: Directory to save trained models
    
    Returns:
        Dictionary with all trained models and results
    """
    
    logger.info("=" * 70)
    logger.info("CROP RECOMMENDATION MODEL TRAINING")
    logger.info("=" * 70)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Knowledge Graph
    kg = CropEcologicalKnowledgeGraph()
    kg.load_crop_catalog(f'{data_dir}/crop_catalog.csv')
    
    # Test rule-based recommendations
    logger.info("\n📋 RULE-BASED RECOMMENDATIONS (Knowledge Graph)")
    logger.info("-" * 70)
    
    high_flood_recs = kg.recommend_by_flood_tolerance('High', top_n=3)
    logger.info(f"High flood risk recommendations: {[r['crop'] for r in high_flood_recs]}")
    
    # 2. Decision Tree Recommender
    logger.info("\n🌳 DECISION TREE RECOMMENDER")
    logger.info("-" * 70)
    
    dt_model = CropRecommendationDecisionTreeModel()
    X, y = dt_model.prepare_data(
        f'{data_dir}/farmer_profiles.csv',
        f'{data_dir}/crop_catalog.csv',
        f'{data_dir}/crop_recommendations.csv'
    )
    
    if len(X) > 0:
        dt_metrics = dt_model.train(X, y)
        logger.info(f"Decision Tree Metrics: {dt_metrics}")
    else:
        logger.warning("Insufficient data for decision tree training")
    
    # 3. Multi-Objective Optimization
    logger.info("\n⚖️  MULTI-OBJECTIVE OPTIMIZATION")
    logger.info("-" * 70)
    
    mo_model = CropRecommendationMultiObjective(f'{data_dir}/crop_catalog.csv')
    
    # Test scenarios
    scenarios = [
        ('High', 'Low'),   # High flood, low drought
        ('Medium', 'Medium'),  # Balanced risk
        ('Low', 'High')    # Low flood, high drought
    ]
    
    for flood_risk, drought_risk in scenarios:
        recs = mo_model.recommend(flood_risk, drought_risk)
        logger.info(f"Flood:{flood_risk}/Drought:{drought_risk} -> {recs[0]['crop']} (score: {recs[0]['score']})")
    
    # 4. Save artifacts
    artifacts = {
        'knowledge_graph': kg,
        'decision_tree': dt_model,
        'multi_objective': mo_model,
        'crop_names': dt_model.crop_names if dt_model.crop_names else []
    }
    
    with open(f'{output_dir}/crop_recommendation_models.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    logger.info(f"\n✓ Models saved to {output_dir}/crop_recommendation_models.pkl")
    logger.info("=" * 70 + "\n")
    
    return artifacts


if __name__ == '__main__':
    train_crop_recommendation_models()
