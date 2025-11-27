"""
ShambaON - Flood Risk Prediction & Crop Recommendation Data Generator
=========================================================================

This module generates synthetic datasets for:
1. Climate, hydrology, and environmental data
2. Crop catalog and agro-ecological guidance
3. Smallholder farmer profiles and field parcels
4. Supporting datasets (forecasts, recommendations, alerts)

The synthetic data is designed to be realistic and representative of
Kenyan agricultural contexts with flood risk considerations.
"""

import os
import uuid
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# CONSTANTS & CONFIGURATIONS
# =====================================================================

KENYAN_COUNTIES = [
    'Mombasa', 'Kwale', 'Kilifi', 'Tana River', 'Lamu', 'Taita Taveta',
    'Nairobi', 'Nyeri', 'Nyandarua', 'Murang\'a', 'Kiambu', 'Turkana',
    'West Pokot', 'Samburu', 'Trans Nzoia', 'Uasin Gishu', 'Elgeyo Marakwet',
    'Nandi', 'Baringo', 'Laikipia', 'Nakuru', 'Narok', 'Kajiado', 'Kericho',
    'Bomet', 'Kakamega', 'Vihiga', 'Bungoma', 'Busia', 'Siaya', 'Kisumu',
    'Homa Bay', 'Migori', 'Kisii', 'Nyamira', 'Machakos', 'Makueni',
    'Mbeere', 'Embu', 'Isiolo', 'Meru', 'Tharaka Nithi', 'Garissa', 'Wajir'
]

FLOOD_PRONE_COUNTIES = [
    'Tana River', 'Kilifi', 'Mombasa', 'Lamu', 'Siaya', 'Kisumu',
    'Homa Bay', 'Migori', 'Nyanza', 'Garissa', 'Isiolo', 'Baringo'
]

CROP_TYPES = {
    'Sorghum': {
        'scientific_name': 'Sorghum bicolor',
        'flood_tolerance': 'Medium',
        'drought_tolerance': 'High',
        'soil_preference': 'loam, sandy loam',
        'temp_range': '20-40',
        'yield_potential': '1.5-3.0',
        'waterlogging_tolerance': 'Low'
    },
    'Millet': {
        'scientific_name': 'Eleusine coracana',
        'flood_tolerance': 'Low',
        'drought_tolerance': 'High',
        'soil_preference': 'sandy, loam',
        'temp_range': '20-35',
        'yield_potential': '1.0-2.5',
        'waterlogging_tolerance': 'Low'
    },
    'Arrowroot': {
        'scientific_name': 'Maranta arundinacea',
        'flood_tolerance': 'High',
        'drought_tolerance': 'Low',
        'soil_preference': 'loam, clay loam',
        'temp_range': '20-30',
        'yield_potential': '8.0-15.0',
        'waterlogging_tolerance': 'High'
    },
    'Rice': {
        'scientific_name': 'Oryza sativa',
        'flood_tolerance': 'High',
        'drought_tolerance': 'Low',
        'soil_preference': 'clay, clay loam',
        'temp_range': '20-32',
        'yield_potential': '2.0-4.0',
        'waterlogging_tolerance': 'High'
    },
    'Maize': {
        'scientific_name': 'Zea mays',
        'flood_tolerance': 'Low',
        'drought_tolerance': 'Medium',
        'soil_preference': 'loam, clay loam',
        'temp_range': '18-30',
        'yield_potential': '3.0-8.0',
        'waterlogging_tolerance': 'Low'
    },
    'Beans': {
        'scientific_name': 'Phaseolus vulgaris',
        'flood_tolerance': 'Low',
        'drought_tolerance': 'Medium',
        'soil_preference': 'loam, sandy loam',
        'temp_range': '15-30',
        'yield_potential': '1.0-2.5',
        'waterlogging_tolerance': 'Low'
    },
    'Cassava': {
        'scientific_name': 'Manihot esculenta',
        'flood_tolerance': 'Medium',
        'drought_tolerance': 'High',
        'soil_preference': 'sandy, sandy loam',
        'temp_range': '18-32',
        'yield_potential': '5.0-12.0',
        'waterlogging_tolerance': 'Medium'
    },
    'Finger Millet': {
        'scientific_name': 'Eleusine coracana',
        'flood_tolerance': 'Low',
        'drought_tolerance': 'High',
        'soil_preference': 'loam, sandy loam',
        'temp_range': '18-30',
        'yield_potential': '0.8-2.0',
        'waterlogging_tolerance': 'Low'
    }
}

AGROECOLOGICAL_ZONES = {
    'LH1': 'Maize/Coffee/Tea Highland with high production potential',
    'LH2': 'Maize/Coffee/Tea Highland with medium production potential',
    'LH3': 'Maize/Coffee/Tea Highland with low production potential',
    'UM1': 'Upper midland with high potential for cash crops',
    'UM2': 'Upper midland with medium potential',
    'UM3': 'Upper midland with low potential',
    'LM1': 'Lower midland with high potential for maize/cotton',
    'LM2': 'Lower midland with medium potential',
    'LM3': 'Lower midland with low potential',
    'AL1': 'Arid/Semi-arid with low livestock production',
    'AL2': 'Arid/Semi-arid with very low production'
}

SOIL_TYPES = ['clay', 'sandy', 'loam', 'clay loam', 'sandy loam', 'silt loam']
DOMAINS = ['Climate', 'Hydrology', 'Soil', 'RemoteSensing']
SUBDOMAINS = {
    'Climate': ['Rainfall', 'Temperature', 'Humidity', 'WindSpeed'],
    'Hydrology': ['RiverLevel', 'Streamflow', 'Groundwater', 'WaterQuality'],
    'Soil': ['SoilMoisture', 'SoilPH', 'SoilNutrients', 'SoilTexture'],
    'RemoteSensing': ['NDVI', 'LST', 'Precipitation', 'VegetationIndex']
}

SOURCES = ['KE-MET', 'WRMA', 'NASA', 'FAO', 'KALRO', 'USGS']
QUALITY_FLAGS = ['OK', 'ESTIMATED', 'MISSING', 'CORRECTED']
CHANNELS = ['SMS', 'USSD', 'App', 'WhatsApp']
ALERT_SEVERITIES = ['Info', 'Warning', 'Critical']

# =====================================================================
# DATASET 1: Climate, Hydrology, and Environmental Data
# =====================================================================

def generate_climate_hydrology_data(num_records=5000):
    """
    Generate synthetic climate, hydrology, and environmental data.
    
    Args:
        num_records: Number of records to generate
    
    Returns:
        DataFrame with climate/hydrology data
    """
    logger.info(f"Generating {num_records} climate/hydrology records...")
    
    records = []
    
    # County distribution - flood-prone counties get more rainfall records
    county_weights = [3 if county in FLOOD_PRONE_COUNTIES else 1 
                      for county in KENYAN_COUNTIES]
    
    for _ in range(num_records):
        # Random selection with weights for flood-prone counties
        county = np.random.choice(KENYAN_COUNTIES, p=np.array(county_weights)/sum(county_weights))
        domain = random.choice(DOMAINS)
        subdomain = random.choice(SUBDOMAINS[domain])
        
        # Generate realistic element names
        element_map = {
            'Rainfall': 'rainfall_mm_day',
            'Temperature': 'temperature_celsius',
            'Humidity': 'relative_humidity_percent',
            'WindSpeed': 'wind_speed_kmh',
            'RiverLevel': 'river_level_m',
            'Streamflow': 'streamflow_m3_s',
            'Groundwater': 'groundwater_depth_m',
            'WaterQuality': 'turbidity_ntu',
            'SoilMoisture': 'soil_moisture_percent',
            'SoilPH': 'soil_ph_value',
            'SoilNutrients': 'nitrogen_mg_kg',
            'SoilTexture': 'sand_percent',
            'NDVI': 'ndvi_index',
            'LST': 'land_surface_temp_celsius',
            'Precipitation': 'precipitation_mm',
            'VegetationIndex': 'evi_index'
        }
        element = element_map.get(subdomain, f'{subdomain.lower()}_value')
        
        # Generate realistic values
        value = generate_realistic_value(subdomain)
        
        # Unit mapping
        unit_map = {
            'rainfall_mm_day': 'mm',
            'temperature_celsius': '°C',
            'relative_humidity_percent': '%',
            'wind_speed_kmh': 'km/h',
            'river_level_m': 'm',
            'streamflow_m3_s': 'm³/s',
            'groundwater_depth_m': 'm',
            'turbidity_ntu': 'NTU',
            'soil_moisture_percent': '%',
            'soil_ph_value': 'pH',
            'nitrogen_mg_kg': 'mg/kg',
            'sand_percent': '%',
            'ndvi_index': 'index',
            'land_surface_temp_celsius': '°C',
            'precipitation_mm': 'mm',
            'evi_index': 'index'
        }
        unit = unit_map.get(element, 'unit')
        
        # Random date in past 2 years
        days_back = random.randint(0, 730)
        record_date = datetime.now() - timedelta(days=days_back)
        
        # Geographic coordinates (realistic for Kenya)
        latitude = np.random.uniform(-4.7, 5.0)
        longitude = np.random.uniform(33.9, 41.9)
        elevation = np.random.uniform(0, 5199)
        
        records.append({
            'record_id': str(uuid.uuid4()),
            'county': county,
            'country_name': 'Kenya',
            'domain': domain,
            'subdomain': subdomain,
            'element': element,
            'item': f'station_{random.randint(1, 200):03d}',
            'value': value,
            'unit': unit,
            'year': record_date.year,
            'month': record_date.month,
            'day': record_date.day,
            'timestamp_utc': record_date.isoformat() + 'Z',
            'latitude': latitude,
            'longitude': longitude,
            'elevation_m': elevation,
            'spatial_resolution': random.choice(['10m', '1km', 'county-level']),
            'temporal_resolution': random.choice(['hourly', 'daily', 'monthly']),
            'source': random.choice(SOURCES),
            'flag': random.choices(QUALITY_FLAGS, weights=[80, 10, 5, 5])[0],
            'ingestion_batch_id': f'batch_{datetime.now().strftime("%Y%m%d")}',
            'version': '1.0',
            'comments_en': f'{subdomain} measurement from {random.choice(SOURCES)}',
            'data_license': random.choice(['CC-BY', 'public', 'restricted'])
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} climate/hydrology records")
    return df


def generate_realistic_value(subdomain):
    """Generate realistic values for different measurement types."""
    ranges = {
        'Rainfall': (0, 150),
        'Temperature': (10, 35),
        'Humidity': (20, 95),
        'WindSpeed': (0, 50),
        'RiverLevel': (0.5, 8),
        'Streamflow': (10, 500),
        'Groundwater': (1, 50),
        'WaterQuality': (0, 100),
        'SoilMoisture': (5, 80),
        'SoilPH': (4.5, 8.0),
        'SoilNutrients': (10, 500),
        'SoilTexture': (10, 80),
        'NDVI': (0.2, 0.9),
        'LST': (15, 45),
        'Precipitation': (0, 200),
        'VegetationIndex': (0.1, 0.8)
    }
    
    min_val, max_val = ranges.get(subdomain, (0, 100))
    
    # Add some skew towards realistic distributions
    if subdomain in ['Rainfall', 'Precipitation']:
        # Rainfall is often zero or high
        if random.random() < 0.4:
            return 0
        return np.random.gamma(shape=2, scale=30)
    
    return np.random.uniform(min_val, max_val)


# =====================================================================
# DATASET 2: Crop Catalog and Agro-Ecological Guidance
# =====================================================================

def generate_crop_catalog():
    """
    Generate crop catalog with agro-ecological guidance.
    
    Returns:
        DataFrame with crop information
    """
    logger.info("Generating crop catalog...")
    
    records = []
    
    for crop_name, crop_data in CROP_TYPES.items():
        for aez_code, aez_desc in AGROECOLOGICAL_ZONES.items():
            # Determine suitability
            suitability = random.choice(['Highly Suitable', 'Suitable', 'Moderately Suitable'])
            
            records.append({
                'crop_id': str(uuid.uuid4()),
                'country_name': 'Kenya',
                'crop': crop_name,
                'scientific_name': crop_data['scientific_name'],
                'variety': f'{crop_name}_var_{random.randint(1, 5)}',
                'agroecological_zone': aez_code,
                'agroecological_zone_description': aez_desc,
                'agroecological_zone_practices': json.dumps([
                    'Use improved varieties', 'Conserve moisture', 'Crop rotation',
                    'Integrated pest management', 'Apply organic matter'
                ]),
                'agroecological_zone_units': 'metric',
                'additional_information': f'{suitability} for {aez_code}',
                'early_sowing_day': random.randint(1, 15),
                'early_sowing_month': random.randint(1, 12),
                'later_sowing_day': random.randint(15, 28),
                'later_sowing_month': random.randint(1, 12),
                'all_year': False,
                'sowing_rate_value': np.random.uniform(15, 50),
                'sowing_rate_unit': 'kg/ha',
                'growing_period_value': random.randint(60, 180),
                'growing_period_period': 'days',
                'early_harvest_day': random.randint(1, 15),
                'early_harvest_month': random.randint(1, 12),
                'late_harvest_day': random.randint(15, 28),
                'late_harvest_month': random.randint(1, 12),
                'flood_tolerance_class': crop_data['flood_tolerance'],
                'drought_tolerance_class': crop_data['drought_tolerance'],
                'soil_preference': crop_data['soil_preference'],
                'soil_ph_range': '5.5-7.5',
                'waterlogging_tolerance': crop_data['waterlogging_tolerance'],
                'temperature_range_c': crop_data['temp_range'],
                'yield_potential_t_ha': crop_data['yield_potential'],
                'market_notes': f'Good market demand in flood-prone regions',
                'comments_en': f'Suitable for {aez_code} with {crop_data["flood_tolerance"]} flood tolerance',
                'source': random.choice(['FAO', 'KALRO', 'literature']),
                'version': '1.0',
                'data_license': 'CC-BY'
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated crop catalog with {len(df)} records")
    return df


# =====================================================================
# DATASET 3: Smallholder Farmer Profiles and Field Parcels
# =====================================================================

def generate_farmer_profiles(num_farmers=1000):
    """
    Generate smallholder farmer profiles and field parcels.
    
    Args:
        num_farmers: Number of farmers to generate
    
    Returns:
        DataFrame with farmer profiles
    """
    logger.info(f"Generating {num_farmers} farmer profiles...")
    
    records = []
    
    for i in range(num_farmers):
        county = random.choice(KENYAN_COUNTIES)
        num_parcels = random.randint(1, 3)
        
        for parcel_idx in range(num_parcels):
            records.append({
                'farmer_id': str(uuid.uuid4()),
                'household_id': str(uuid.uuid4()),
                'county': county,
                'sub_county': f'{county}_SC_{random.randint(1, 10)}',
                'ward': f'Ward_{random.randint(1, 5)}',
                'village_locality': f'Village_{random.randint(1, 20)}',
                'latitude': np.random.uniform(-4.7, 5.0),
                'longitude': np.random.uniform(33.9, 41.9),
                'parcel_id': str(uuid.uuid4()),
                'parcel_area_ha': np.random.uniform(0.25, 5.0),
                'irrigation_availability': random.choice([True, False]),
                'soil_type': random.choice(SOIL_TYPES),
                'soil_ph': np.random.uniform(5.0, 8.0),
                'primary_crops': json.dumps(random.sample(list(CROP_TYPES.keys()), k=random.randint(1, 3))),
                'livelihood_mix': json.dumps({
                    'crop': random.randint(40, 90),
                    'livestock': random.randint(5, 40),
                    'off_farm': random.randint(5, 30)
                }),
                'cooperative_membership': random.choice([True, False]),
                'cooperative_name': f'Coop_{random.randint(1, 100)}' if random.random() > 0.4 else None,
                'advisory_interactions': random.randint(0, 20),
                'historical_yields_t_ha': json.dumps({crop: round(np.random.uniform(0.5, 5), 2) 
                                                     for crop in random.sample(list(CROP_TYPES.keys()), 2)}),
                'historical_flood_events': random.randint(0, 5) if county in FLOOD_PRONE_COUNTIES else random.randint(0, 2),
                'access_to_seeds': random.choice([True, False]),
                'access_to_fertilizer': random.choice([True, False]),
                'access_to_credit': random.choice([True, False]),
                'phone_reachable': random.choice([True, False]),
                'preferred_language': random.choice(['English', 'Kiswahili', 'Local']),
                'consent_status': random.choice(['Granted', 'Pending', 'Denied']),
                'source': 'farmer_registration',
                'version': '1.0',
                'comments_en': f'Farmer profile from {county}'
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} farmer profiles")
    return df


# =====================================================================
# DATASET 4: Supporting Datasets - Forecasts, Recommendations, Alerts
# =====================================================================

def generate_flood_risk_forecasts(farmer_df, num_forecasts=2000):
    """
    Generate flood risk forecast records.
    
    Args:
        farmer_df: DataFrame with farmer profiles
        num_forecasts: Number of forecast records
    
    Returns:
        DataFrame with flood risk forecasts
    """
    logger.info(f"Generating {num_forecasts} flood risk forecasts...")
    
    records = []
    
    for _ in range(num_forecasts):
        parcel_row = farmer_df.sample(1).iloc[0]
        forecast_time = datetime.now() - timedelta(days=random.randint(0, 30))
        target_start = forecast_time + timedelta(days=random.randint(1, 7))
        target_end = target_start + timedelta(days=random.randint(7, 30))
        
        # Flood-prone counties have higher risk scores
        is_flood_prone = parcel_row['county'] in FLOOD_PRONE_COUNTIES
        if is_flood_prone:
            risk_score = np.random.beta(5, 2)  # Skewed towards higher values
        else:
            risk_score = np.random.beta(2, 5)  # Skewed towards lower values
        
        risk_class = 'High' if risk_score > 0.7 else ('Medium' if risk_score > 0.4 else 'Low')
        
        records.append({
            'forecast_id': str(uuid.uuid4()),
            'parcel_id': parcel_row['parcel_id'],
            'county': parcel_row['county'],
            'forecast_timestamp_utc': forecast_time.isoformat() + 'Z',
            'target_window_start_utc': target_start.isoformat() + 'Z',
            'target_window_end_utc': target_end.isoformat() + 'Z',
            'flood_risk_score': round(risk_score, 3),
            'risk_class': risk_class,
            'top_features': json.dumps({
                'rainfall_intensity': round(np.random.uniform(0.5, 1.0), 2),
                'river_level': round(np.random.uniform(0.3, 1.0), 2),
                'soil_moisture': round(np.random.uniform(0.2, 0.9), 2),
                'land_slope': round(np.random.uniform(0.1, 0.8), 2),
                'historical_flood_frequency': round(np.random.uniform(0, 1), 2)
            }),
            'model_version': '1.0.0',
            'source': 'flood_risk_model_v1'
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} flood risk forecasts")
    return df


def generate_crop_recommendations(farmer_df, crop_df, num_recommendations=2000):
    """
    Generate crop recommendation records.
    
    Args:
        farmer_df: DataFrame with farmer profiles
        crop_df: DataFrame with crop catalog
        num_recommendations: Number of recommendation records
    
    Returns:
        DataFrame with crop recommendations
    """
    logger.info(f"Generating {num_recommendations} crop recommendations...")
    
    records = []
    
    for _ in range(num_recommendations):
        parcel_row = farmer_df.sample(1).iloc[0]
        recommend_time = datetime.now() - timedelta(days=random.randint(0, 30))
        
        # Filter crops suitable for the county/aez
        suitable_crops = crop_df[
            (crop_df['country_name'] == 'Kenya')
        ].sample(1).iloc[0]
        
        confidence = np.random.uniform(0.6, 0.99)
        
        # Crops with high flood tolerance get recommended in flood-prone areas
        if parcel_row['county'] in FLOOD_PRONE_COUNTIES:
            if suitable_crops['flood_tolerance_class'] == 'High':
                confidence = min(1.0, confidence + 0.2)
        
        season = random.choice(['Long rains (MAM)', 'Short rains (OND)'])
        
        records.append({
            'recommendation_id': str(uuid.uuid4()),
            'parcel_id': parcel_row['parcel_id'],
            'county': parcel_row['county'],
            'recommendation_timestamp_utc': recommend_time.isoformat() + 'Z',
            'recommended_crop_id': suitable_crops['crop_id'],
            'recommended_crop_name': suitable_crops['crop'],
            'confidence_score': round(min(confidence, 1.0), 3),
            'rationale_en': f"Recommended {suitable_crops['crop']} for {parcel_row['county']} due to soil type and climate compatibility",
            'season_window': season,
            'practice_notes': 'Apply recommended practices for AEZ',
            'model_version': '1.0.0',
            'source': 'crop_recommendation_engine_v1'
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} crop recommendations")
    return df


def generate_alerts(farmer_df, num_alerts=1500):
    """
    Generate alert/notification records.
    
    Args:
        farmer_df: DataFrame with farmer profiles
        num_alerts: Number of alert records
    
    Returns:
        DataFrame with alerts
    """
    logger.info(f"Generating {num_alerts} alerts...")
    
    records = []
    alert_messages = {
        'Critical': [
            'Severe flooding expected in your area within 24 hours',
            'Flash flood warning - evacuate livestock immediately',
            'River level critically high - prepare emergency measures'
        ],
        'Warning': [
            'Heavy rainfall forecast - secure drainage systems',
            'Moderate flood risk - monitor weather updates',
            'River levels rising - remain alert for updates'
        ],
        'Info': [
            'Planting season recommendations available',
            'New crop advisory for your region',
            'Weather forecast for next 7 days available'
        ]
    }
    
    for _ in range(num_alerts):
        farmer_row = farmer_df.sample(1).iloc[0]
        severity = random.choices(
            ['Info', 'Warning', 'Critical'],
            weights=[40, 40, 20]
        )[0]
        
        channel = random.choice(CHANNELS)
        sent_time = datetime.now() - timedelta(hours=random.randint(1, 720))
        
        records.append({
            'alert_id': str(uuid.uuid4()),
            'farmer_id': farmer_row['farmer_id'],
            'parcel_id': farmer_row['parcel_id'],
            'channel': channel,
            'language': farmer_row['preferred_language'],
            'severity': severity,
            'message_en': random.choice(alert_messages[severity]),
            'sent_timestamp_utc': sent_time.isoformat() + 'Z',
            'delivery_status': random.choices(
                ['Queued', 'Sent', 'Failed'],
                weights=[10, 80, 10]
            )[0],
            'source': 'alert_service'
        })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} alerts")
    return df


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main function to generate all datasets."""
    
    logger.info("=" * 70)
    logger.info("ShambaON - Flood Risk Prediction & Crop Recommendation")
    logger.info("Data Generator Started")
    logger.info("=" * 70)
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Data directory: {data_dir}")
    
    try:
        # Generate Dataset 1: Climate, Hydrology, Environmental Data
        climate_df = generate_climate_hydrology_data(num_records=5000)
        climate_file = data_dir / 'climate_hydrology_environmental.csv'
        climate_df.to_csv(climate_file, index=False)
        logger.info(f"✓ Saved: {climate_file}")
        
        # Generate Dataset 2: Crop Catalog
        crop_df = generate_crop_catalog()
        crop_file = data_dir / 'crop_catalog.csv'
        crop_df.to_csv(crop_file, index=False)
        logger.info(f"✓ Saved: {crop_file}")
        
        # Generate Dataset 3: Farmer Profiles
        farmer_df = generate_farmer_profiles(num_farmers=500)
        farmer_file = data_dir / 'farmer_profiles.csv'
        farmer_df.to_csv(farmer_file, index=False)
        logger.info(f"✓ Saved: {farmer_file}")
        
        # Generate Dataset 4a: Flood Risk Forecasts
        forecasts_df = generate_flood_risk_forecasts(farmer_df, num_forecasts=1000)
        forecasts_file = data_dir / 'flood_risk_forecasts.csv'
        forecasts_df.to_csv(forecasts_file, index=False)
        logger.info(f"✓ Saved: {forecasts_file}")
        
        # Generate Dataset 4b: Crop Recommendations
        recommendations_df = generate_crop_recommendations(farmer_df, crop_df, num_recommendations=1000)
        recommendations_file = data_dir / 'crop_recommendations.csv'
        recommendations_df.to_csv(recommendations_file, index=False)
        logger.info(f"✓ Saved: {recommendations_file}")
        
        # Generate Dataset 4c: Alerts
        alerts_df = generate_alerts(farmer_df, num_alerts=1500)
        alerts_file = data_dir / 'alerts_notifications.csv'
        alerts_df.to_csv(alerts_file, index=False)
        logger.info(f"✓ Saved: {alerts_file}")
        
        # Summary Report
        logger.info("\n" + "=" * 70)
        logger.info("DATA GENERATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Climate/Hydrology records:    {len(climate_df):,}")
        logger.info(f"Crop catalog records:         {len(crop_df):,}")
        logger.info(f"Farmer profiles:              {len(farmer_df):,}")
        logger.info(f"Flood risk forecasts:         {len(forecasts_df):,}")
        logger.info(f"Crop recommendations:         {len(recommendations_df):,}")
        logger.info(f"Alerts/notifications:         {len(alerts_df):,}")
        logger.info("=" * 70)
        logger.info("✓ All datasets generated successfully!")
        logger.info("=" * 70 + "\n")
        
        return {
            'climate': climate_df,
            'crops': crop_df,
            'farmers': farmer_df,
            'forecasts': forecasts_df,
            'recommendations': recommendations_df,
            'alerts': alerts_df
        }
        
    except Exception as e:
        logger.error(f"Error during data generation: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    datasets = main()
