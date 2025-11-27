"""
Synthetic Data Generation & Simulation Engine
==============================================

This module provides:
1. Synthetic data augmentation using SDV and sklearn
2. Scenario simulation for what-if testing
3. Extreme event simulation (floods, droughts)
4. Market shock simulation
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

try:
    from sdv.tabular import CTGAN, TVAE
    HAS_SDV = True
except ImportError:
    HAS_SDV = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
# SYNTHETIC DATA GENERATION
# =====================================================================

class SyntheticDataGenerator:
    """Generate synthetic datasets for model augmentation."""
    
    def __init__(self):
        logger.info("SyntheticDataGenerator initialized")
    
    def generate_climate_data(self, n_samples: int = 1000, 
                             num_features: int = 8) -> pd.DataFrame:
        """
        Generate synthetic climate/hydrology data.
        
        Args:
            n_samples: Number of synthetic records
            num_features: Number of features
        
        Returns:
            DataFrame with synthetic climate data
        """
        logger.info(f"Generating {n_samples} synthetic climate records...")
        
        # Create features with realistic distributions
        data = {
            'rainfall_mean': np.random.gamma(shape=2, scale=20, size=n_samples),
            'rainfall_std': np.random.exponential(scale=15, size=n_samples),
            'rainfall_max': np.random.exponential(scale=50, size=n_samples),
            'rainfall_total': np.random.gamma(shape=5, scale=30, size=n_samples),
            'river_level_mean': np.random.gamma(shape=2, scale=1.5, size=n_samples),
            'river_level_max': np.random.gamma(shape=3, scale=1.2, size=n_samples),
            'soil_moisture_mean': np.random.uniform(20, 80, n_samples),
            'soil_moisture_max': np.random.uniform(40, 100, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['rainfall_variability'] = df['rainfall_std'] / (df['rainfall_mean'] + 1e-5)
        df['soil_saturation_ratio'] = df['soil_moisture_max'] / 100.0
        df['rainfall_river_interaction'] = df['rainfall_max'] * df['river_level_max']
        
        logger.info(f"Generated synthetic climate data: {df.shape}")
        
        return df
    
    def generate_farmer_data(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Generate synthetic farmer profiles.
        
        Args:
            n_samples: Number of synthetic farmer records
        
        Returns:
            DataFrame with synthetic farmer data
        """
        logger.info(f"Generating {n_samples} synthetic farmer profiles...")
        
        counties = ['Tana River', 'Kilifi', 'Siaya', 'Kisumu', 'Nairobi', 
                   'Nakuru', 'Kisii', 'Migori', 'Homa Bay', 'Isiolo']
        soil_types = ['clay', 'sandy', 'loam', 'clay loam', 'sandy loam']
        
        data = {
            'county': np.random.choice(counties, n_samples),
            'soil_ph': np.random.uniform(5.0, 8.0, n_samples),
            'parcel_area_ha': np.random.lognormal(mean=0, sigma=0.8, size=n_samples),
            'irrigation_availability': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'soil_type': np.random.choice(soil_types, n_samples),
            'historical_flood_events': np.random.poisson(lam=2, size=n_samples),
            'access_to_credit': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'cooperative_membership': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure parcel area is reasonable
        df['parcel_area_ha'] = df['parcel_area_ha'].clip(0.25, 5.0)
        
        logger.info(f"Generated synthetic farmer data: {df.shape}")
        
        return df
    
    def generate_using_sdv(self, real_data: pd.DataFrame, 
                          n_samples: int = None,
                          model_type: str = 'ctgan') -> pd.DataFrame:
        """
        Generate synthetic data using SDV (if available).
        
        Args:
            real_data: Real dataset to learn from
            n_samples: Number of synthetic samples (default: size of real_data)
            model_type: 'ctgan' or 'tvae'
        
        Returns:
            Synthetic DataFrame
        """
        
        if not HAS_SDV:
            logger.warning("SDV not installed, falling back to sklearn-based generation")
            return self._generate_using_sklearn(real_data, n_samples)
        
        if n_samples is None:
            n_samples = len(real_data)
        
        logger.info(f"Training SDV {model_type.upper()} on {len(real_data)} samples...")
        
        try:
            if model_type == 'ctgan':
                model = CTGAN()
            else:
                model = TVAE()
            
            model.fit(real_data)
            synthetic_data = model.sample(n_samples)
            
            logger.info(f"Generated {len(synthetic_data)} synthetic samples using {model_type.upper()}")
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"SDV generation failed: {e}")
            return self._generate_using_sklearn(real_data, n_samples)
    
    def _generate_using_sklearn(self, real_data: pd.DataFrame, 
                               n_samples: int = None) -> pd.DataFrame:
        """
        Generate synthetic data using sklearn (simple fallback).
        
        Args:
            real_data: Real dataset statistics
            n_samples: Number of synthetic samples
        
        Returns:
            Synthetic DataFrame
        """
        
        if n_samples is None:
            n_samples = len(real_data)
        
        logger.info(f"Generating {n_samples} synthetic samples using sklearn...")
        
        synthetic_data = {}
        
        for col in real_data.columns:
            if real_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Generate from distribution
                mean = real_data[col].mean()
                std = real_data[col].std()
                synthetic_data[col] = np.random.normal(mean, std, n_samples)
            else:
                # Categorical - resample
                synthetic_data[col] = np.random.choice(real_data[col].unique(), n_samples)
        
        return pd.DataFrame(synthetic_data)


# =====================================================================
# SIMULATION ENGINE
# =====================================================================

class SimulationEngine:
    """Simulate what-if scenarios and extreme events."""
    
    def __init__(self):
        logger.info("SimulationEngine initialized")
    
    def simulate_flood_event(self, parcel_data: Dict[str, Any], 
                            severity: str = 'moderate') -> Dict[str, Any]:
        """
        Simulate flood event impact on parcel.
        
        Args:
            parcel_data: Dictionary with parcel characteristics
            severity: 'low', 'moderate', 'severe'
        
        Returns:
            Simulation results with impacts
        """
        
        logger.info(f"Simulating {severity} flood event...")
        
        # Severity multipliers
        severity_map = {
            'low': 0.3,
            'moderate': 0.6,
            'severe': 0.9
        }
        
        multiplier = severity_map.get(severity, 0.6)
        
        # Get baseline yield
        baseline_yield = parcel_data.get('baseline_yield', 2.5)
        
        # Flood impacts on yield
        yield_loss_percent = multiplier * 0.7  # Up to 70% loss with severe flood
        
        if parcel_data.get('irrigation_available'):
            yield_loss_percent *= 0.7  # Irrigation reduces impact
        
        # Crops with high flood tolerance are less affected
        if parcel_data.get('flood_tolerance') == 'High':
            yield_loss_percent *= 0.4
        elif parcel_data.get('flood_tolerance') == 'Medium':
            yield_loss_percent *= 0.7
        
        actual_yield = baseline_yield * (1 - yield_loss_percent)
        
        # Economic impact
        crop_price = parcel_data.get('crop_price', 15000)  # KES per ton
        economic_loss = (baseline_yield - actual_yield) * crop_price * parcel_data.get('parcel_area_ha', 1)
        
        results = {
            'event_type': 'flood',
            'severity': severity,
            'baseline_yield_t_ha': baseline_yield,
            'yield_loss_percent': yield_loss_percent * 100,
            'actual_yield_t_ha': actual_yield,
            'economic_loss_kes': economic_loss,
            'recommendation': 'Switch to flood-tolerant crops' if actual_yield < baseline_yield * 0.5 else 'Monitor closely'
        }
        
        return results
    
    def simulate_drought_event(self, parcel_data: Dict[str, Any],
                              duration_months: int = 3) -> Dict[str, Any]:
        """
        Simulate drought event.
        
        Args:
            parcel_data: Dictionary with parcel characteristics
            duration_months: Duration of drought
        
        Returns:
            Simulation results
        """
        
        logger.info(f"Simulating {duration_months}-month drought...")
        
        baseline_yield = parcel_data.get('baseline_yield', 2.5)
        
        # Drought severity increases with duration
        drought_factor = min(0.8, duration_months / 12.0)
        
        # Impact reduction with irrigation
        if parcel_data.get('irrigation_available'):
            drought_factor *= 0.3
        
        # Drought-tolerant crops suffer less
        if parcel_data.get('drought_tolerance') == 'High':
            drought_factor *= 0.5
        elif parcel_data.get('drought_tolerance') == 'Medium':
            drought_factor *= 0.75
        
        yield_loss_percent = drought_factor * 0.6  # Up to 60% loss
        actual_yield = baseline_yield * (1 - yield_loss_percent)
        
        results = {
            'event_type': 'drought',
            'duration_months': duration_months,
            'baseline_yield_t_ha': baseline_yield,
            'yield_loss_percent': yield_loss_percent * 100,
            'actual_yield_t_ha': actual_yield,
            'recommendation': 'Prioritize drought-tolerant crops' if yield_loss_percent > 0.3 else 'Standard management'
        }
        
        return results
    
    def simulate_market_shock(self, parcel_data: Dict[str, Any],
                             price_change_percent: float = -30) -> Dict[str, Any]:
        """
        Simulate market price shock.
        
        Args:
            parcel_data: Dictionary with parcel characteristics
            price_change_percent: Percent change in crop price
        
        Returns:
            Economic impact
        """
        
        logger.info(f"Simulating market shock ({price_change_percent:+.0f}%)...")
        
        baseline_revenue = parcel_data.get('baseline_revenue', 50000)
        actual_revenue = baseline_revenue * (1 + price_change_percent / 100)
        
        results = {
            'event_type': 'market_shock',
            'price_change_percent': price_change_percent,
            'baseline_revenue_kes': baseline_revenue,
            'actual_revenue_kes': actual_revenue,
            'revenue_change_kes': actual_revenue - baseline_revenue,
            'recommendation': 'Diversify crop portfolio' if actual_revenue < baseline_revenue * 0.5 else 'Continue current strategy'
        }
        
        return results
    
    def simulate_extreme_scenario(self, parcel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate combined extreme event (flood + drought + market shock).
        
        Args:
            parcel_data: Dictionary with parcel characteristics
        
        Returns:
            Combined impact analysis
        """
        
        logger.info("Simulating extreme scenario (flood + drought + market shock)...")
        
        flood_impact = self.simulate_flood_event(parcel_data, 'severe')
        drought_impact = self.simulate_drought_event(parcel_data, 4)
        market_impact = self.simulate_market_shock(parcel_data, -40)
        
        # Combined yield impact
        combined_yield_loss = min(0.95, 
            flood_impact['yield_loss_percent'] / 100 + 
            drought_impact['yield_loss_percent'] / 100)
        
        baseline_yield = parcel_data.get('baseline_yield', 2.5)
        actual_yield = baseline_yield * (1 - combined_yield_loss)
        
        crop_price_reduced = parcel_data.get('crop_price', 15000) * (1 - 0.4)
        total_economic_loss = (baseline_yield - actual_yield) * crop_price_reduced * parcel_data.get('parcel_area_ha', 1)
        
        results = {
            'scenario_name': 'Extreme Event',
            'components': {
                'flood': flood_impact,
                'drought': drought_impact,
                'market_shock': market_impact
            },
            'combined_yield_loss_percent': combined_yield_loss * 100,
            'final_yield_t_ha': actual_yield,
            'total_economic_loss_kes': total_economic_loss,
            'resilience_recommendation': 'Urgent need for diversification and irrigation investment'
        }
        
        return results


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def generate_augmented_dataset(original_data: pd.DataFrame, 
                               augmentation_factor: int = 3,
                               output_path: str = 'data/augmented_data.csv') -> pd.DataFrame:
    """
    Generate augmented training dataset.
    
    Args:
        original_data: Original dataset
        augmentation_factor: How many times to augment
        output_path: Path to save augmented data
    
    Returns:
        Augmented DataFrame
    """
    
    logger.info(f"Augmenting dataset with factor {augmentation_factor}...")
    
    generator = SyntheticDataGenerator()
    
    all_data = [original_data]
    
    for i in range(augmentation_factor):
        synthetic = generator.generate_using_sdv(original_data, len(original_data))
        all_data.append(synthetic)
    
    augmented = pd.concat(all_data, ignore_index=True)
    
    # Save augmented data
    Path(output_path).parent.mkdir(exist_ok=True)
    augmented.to_csv(output_path, index=False)
    
    logger.info(f"Augmented dataset saved: {len(augmented)} records to {output_path}")
    
    return augmented


def run_scenario_tests(parcel_configs: List[Dict[str, Any]],
                      output_dir: str = 'simulation_results') -> pd.DataFrame:
    """
    Run simulation tests across multiple parcels.
    
    Args:
        parcel_configs: List of parcel configurations
        output_dir: Output directory for results
    
    Returns:
        DataFrame with all scenario results
    """
    
    logger.info(f"Running scenario tests for {len(parcel_configs)} parcels...")
    
    Path(output_dir).mkdir(exist_ok=True)
    engine = SimulationEngine()
    
    all_results = []
    
    for i, config in enumerate(parcel_configs):
        # Test different scenarios
        scenarios = [
            ('low_flood', engine.simulate_flood_event(config, 'low')),
            ('moderate_flood', engine.simulate_flood_event(config, 'moderate')),
            ('severe_flood', engine.simulate_flood_event(config, 'severe')),
            ('drought_3m', engine.simulate_drought_event(config, 3)),
            ('drought_6m', engine.simulate_drought_event(config, 6)),
            ('market_shock', engine.simulate_market_shock(config, -30)),
            ('extreme', engine.simulate_extreme_scenario(config))
        ]
        
        for scenario_name, result in scenarios:
            result['parcel_id'] = i
            result['scenario'] = scenario_name
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(f'{output_dir}/scenario_results.csv', index=False)
    
    logger.info(f"Scenario results saved: {len(results_df)} total scenarios")
    
    return results_df


if __name__ == '__main__':
    # Test synthetic data generation
    generator = SyntheticDataGenerator()
    synthetic_climate = generator.generate_climate_data(n_samples=500)
    synthetic_farmers = generator.generate_farmer_data(n_samples=500)
    
    # Test simulation engine
    engine = SimulationEngine()
    parcel = {
        'baseline_yield': 2.5,
        'parcel_area_ha': 1.5,
        'irrigation_available': True,
        'flood_tolerance': 'High',
        'drought_tolerance': 'Medium',
        'crop_price': 15000
    }
    
    flood_result = engine.simulate_flood_event(parcel, 'severe')
    logger.info(f"Flood simulation: {flood_result}")
