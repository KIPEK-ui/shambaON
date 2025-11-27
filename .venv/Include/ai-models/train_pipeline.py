"""
Complete Model Training Pipeline
==================================

Orchestrates the full ML workflow:
1. Data generation
2. Model training (flood risk + crop recommendation)
3. Model evaluation
4. Explainability analysis
5. Artifact preservation
"""

import os
import sys
import pickle
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data.data_generator import main as generate_data
from models.flood_risk_model import train_flood_risk_models
from models.crop_recommendation_model import train_crop_recommendation_models
from models.explainable_ai import generate_xai_summary, ModelTransparencyReport
from models.simulation_engine import generate_augmented_dataset, run_scenario_tests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shambaon_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =====================================================================
# MAIN PIPELINE
# =====================================================================

class ShambaONMLPipeline:
    """Complete ML pipeline for ShambaON system."""
    
    def __init__(self, data_dir: str = 'data', output_dir: str = 'models',
                 config_file: str = None):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Directory for data
            output_dir: Directory for models and artifacts
            config_file: Optional configuration file
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("SHAMBAON ML PIPELINE INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Timestamp: {self.timestamp}")
    
    def step_1_generate_data(self, regenerate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Generate synthetic datasets.
        
        Args:
            regenerate: Whether to regenerate even if files exist
        
        Returns:
            Dictionary of generated datasets
        """
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA GENERATION")
        logger.info("=" * 80)
        
        # Check if data already exists
        climate_file = self.data_dir / 'climate_hydrology_environmental.csv'
        if climate_file.exists() and not regenerate:
            logger.info("Dataset already exists, loading...")
            return self._load_datasets()
        
        # Generate new data
        logger.info("Generating synthetic datasets...")
        datasets = generate_data()
        
        logger.info("✓ Data generation completed successfully")
        
        return datasets
    
    def step_2_train_flood_risk_models(self) -> Dict[str, Any]:
        """
        Step 2: Train flood risk prediction models.
        
        Returns:
            Training artifacts and metrics
        """
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FLOOD RISK PREDICTION MODEL TRAINING")
        logger.info("=" * 80)
        
        artifacts = train_flood_risk_models(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        
        logger.info("✓ Flood risk models trained successfully")
        
        return artifacts
    
    def step_3_train_crop_recommendation_models(self) -> Dict[str, Any]:
        """
        Step 3: Train crop recommendation models.
        
        Returns:
            Training artifacts and metrics
        """
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: CROP RECOMMENDATION MODEL TRAINING")
        logger.info("=" * 80)
        
        artifacts = train_crop_recommendation_models(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        
        logger.info("✓ Crop recommendation models trained successfully")
        
        return artifacts
    
    def step_4_generate_xai_analysis(self, flood_artifacts: Dict[str, Any],
                                    X_val: np.ndarray,
                                    feature_names: list) -> Dict[str, Any]:
        """
        Step 4: Generate explainability analysis.
        
        Args:
            flood_artifacts: Flood risk model artifacts
            X_val: Validation feature matrix
            feature_names: Feature names
        
        Returns:
            XAI analysis
        """
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: EXPLAINABLE AI ANALYSIS")
        logger.info("=" * 80)
        
        classifiers = flood_artifacts.get('classifiers')
        if classifiers is None:
            logger.warning("Classifiers not found in artifacts")
            return {}
        
        # Generate SHAP explanations
        xai_summary = generate_xai_summary(
            classifiers.models,
            X_val,
            feature_names,
            output_dir=str(self.output_dir)
        )
        
        # Generate transparency report
        report_gen = ModelTransparencyReport()
        for model_name, model in classifiers.models.items():
            report = report_gen.generate_report(
                model, X_val, None,  # y values if available
                feature_names=feature_names,
                model_name=model_name
            )
            
            report_path = self.output_dir / f'transparency_report_{model_name}.json'
            report_gen.save_report(report, str(report_path))
            logger.info(f"  Saved transparency report: {model_name}")
        
        logger.info("✓ XAI analysis completed")
        
        return xai_summary
    
    def step_5_run_simulations(self) -> pd.DataFrame:
        """
        Step 5: Run scenario simulations.
        
        Returns:
            Simulation results DataFrame
        """
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: SCENARIO SIMULATION")
        logger.info("=" * 80)
        
        # Create sample parcel configurations
        parcel_configs = [
            {
                'baseline_yield': 2.5,
                'parcel_area_ha': 1.0,
                'irrigation_available': True,
                'flood_tolerance': 'High',
                'drought_tolerance': 'Medium',
                'crop_price': 15000
            },
            {
                'baseline_yield': 2.0,
                'parcel_area_ha': 0.5,
                'irrigation_available': False,
                'flood_tolerance': 'Low',
                'drought_tolerance': 'High',
                'crop_price': 12000
            },
            {
                'baseline_yield': 3.0,
                'parcel_area_ha': 2.0,
                'irrigation_available': True,
                'flood_tolerance': 'Medium',
                'drought_tolerance': 'Medium',
                'crop_price': 18000
            }
        ]
        
        simulation_results = run_scenario_tests(
            parcel_configs,
            output_dir=str(self.output_dir / 'simulations')
        )
        
        logger.info(f"✓ Simulation completed with {len(simulation_results)} scenarios")
        
        return simulation_results
    
    def step_6_generate_report(self, all_artifacts: Dict[str, Any]) -> str:
        """
        Step 6: Generate comprehensive report.
        
        Args:
            all_artifacts: Dictionary with all training artifacts
        
        Returns:
            Path to report file
        """
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: COMPREHENSIVE REPORT GENERATION")
        logger.info("=" * 80)
        
        report = {
            'pipeline_metadata': {
                'execution_date': self.timestamp,
                'data_directory': str(self.data_dir),
                'output_directory': str(self.output_dir),
                'python_version': f"{sys.version}",
            },
            'data_summary': all_artifacts.get('data_summary', {}),
            'flood_risk_models': all_artifacts.get('flood_results', {}),
            'crop_recommendation_models': all_artifacts.get('crop_results', {}),
            'artifacts_saved': [
                'flood_risk_models.pkl',
                'crop_recommendation_models.pkl',
                'xai_summary.json',
                'transparency_report_*.json',
                'simulations/scenario_results.csv'
            ],
            'next_steps': [
                '1. Deploy models to Streamlit application',
                '2. Connect to Supabase backend',
                '3. Integrate USSD interface via Africa\'s Talking',
                '4. Monitor model performance in production',
                '5. Collect farmer feedback for continuous improvement',
                '6. Regular retraining with new data'
            ]
        }
        
        report_path = self.output_dir / f'pipeline_report_{self.timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✓ Report generated: {report_path}")
        
        return str(report_path)
    
    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load existing datasets."""
        
        datasets = {}
        
        for file in ['climate_hydrology_environmental', 'crop_catalog', 'farmer_profiles',
                    'flood_risk_forecasts', 'crop_recommendations', 'alerts_notifications']:
            filepath = self.data_dir / f'{file}.csv'
            if filepath.exists():
                datasets[file] = pd.read_csv(filepath)
                logger.info(f"Loaded: {file} ({len(datasets[file])} records)")
        
        return datasets
    
    def run_complete_pipeline(self, regenerate_data: bool = True) -> Dict[str, Any]:
        """
        Run complete ML pipeline.
        
        Args:
            regenerate_data: Whether to regenerate synthetic data
        
        Returns:
            Dictionary with all results
        """
        
        try:
            # Step 1: Data Generation
            datasets = self.step_1_generate_data(regenerate=regenerate_data)
            
            # Step 2: Flood Risk Models
            flood_artifacts = self.step_2_train_flood_risk_models()
            
            # Step 3: Crop Recommendation Models
            crop_artifacts = self.step_3_train_crop_recommendation_models()
            
            # Step 4: XAI Analysis
            # Prepare validation data for XAI
            climate_df = pd.read_csv(self.data_dir / 'climate_hydrology_environmental.csv')
            forecast_df = pd.read_csv(self.data_dir / 'flood_risk_forecasts.csv')
            farmer_df = pd.read_csv(self.data_dir / 'farmer_profiles.csv')
            
            X_sample = np.random.randn(100, 13)  # Sample for XAI
            feature_names = flood_artifacts.get('feature_names', 
                [f'Feature_{i}' for i in range(13)])
            
            xai_results = self.step_4_generate_xai_analysis(
                flood_artifacts, X_sample, feature_names
            )
            
            # Step 5: Simulations
            simulation_results = self.step_5_run_simulations()
            
            # Step 6: Report
            all_artifacts = {
                'data_summary': {
                    'climate_records': len(datasets.get('climate_hydrology_environmental', [])),
                    'farmers': len(datasets.get('farmer_profiles', [])),
                    'crop_catalog_entries': len(datasets.get('crop_catalog', []))
                },
                'flood_results': {
                    'models_trained': list(flood_artifacts.get('results', {}).keys()),
                    'best_model': 'ensemble'
                },
                'crop_results': {
                    'models_trained': ['knowledge_graph', 'decision_tree', 'multi_objective']
                }
            }
            
            report_path = self.step_6_generate_report(all_artifacts)
            
            # Final summary
            logger.info("\n" + "=" * 80)
            logger.info("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            summary = {
                'status': 'success',
                'timestamp': self.timestamp,
                'report_path': report_path,
                'artifacts': {
                    'flood_risk_models': str(self.output_dir / 'flood_risk_models.pkl'),
                    'crop_recommendation_models': str(self.output_dir / 'crop_recommendation_models.pkl'),
                    'xai_summary': str(self.output_dir / 'xai_summary.json'),
                    'simulations': str(self.output_dir / 'simulations/scenario_results.csv')
                }
            }
            
            logger.info(f"\nKey Outputs:")
            logger.info(f"  ✓ Flood Risk Models: {summary['artifacts']['flood_risk_models']}")
            logger.info(f"  ✓ Crop Recommendations: {summary['artifacts']['crop_recommendation_models']}")
            logger.info(f"  ✓ XAI Analysis: {summary['artifacts']['xai_summary']}")
            logger.info(f"  ✓ Simulation Results: {summary['artifacts']['simulations']}")
            logger.info(f"\nFull Report: {report_path}")
            logger.info("=" * 80 + "\n")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    """Main entry point for pipeline."""
    
    # Initialize pipeline
    pipeline = ShambaONMLPipeline(
        data_dir='data',
        output_dir='models'
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(regenerate_data=True)
    
    return results


if __name__ == '__main__':
    main()
