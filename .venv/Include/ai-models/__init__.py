"""
ShambaON ML Package
===================

AI-powered flood risk prediction and crop recommendation system for smallholder farmers in Kenya.
"""

__version__ = '1.0.0'
__author__ = 'ShambaON Team'

from models.flood_risk_model import train_flood_risk_models, FloodRiskClassificationModels, FloodRiskDataPreprocessor
from models.crop_recommendation_model import train_crop_recommendation_models, CropEcologicalKnowledgeGraph, CropRecommendationDecisionTreeModel
from models.explainable_ai import SHAPExplainer, LIMEExplainer, ModelTransparencyReport
from models.simulation_engine import SimulationEngine, SyntheticDataGenerator
from inference import FloodRiskPredictor, CropRecommender, InferenceAPI, BatchPredictor

__all__ = [
    'train_flood_risk_models',
    'train_crop_recommendation_models',
    'FloodRiskClassificationModels',
    'CropEcologicalKnowledgeGraph',
    'SHAPExplainer',
    'LIMEExplainer',
    'SimulationEngine',
    'InferenceAPI',
    'FloodRiskPredictor',
    'CropRecommender',
]
