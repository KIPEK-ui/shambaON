"""
Quick Start Script for ShambaON ML System
==========================================

This script demonstrates how to:
1. Generate synthetic data
2. Train models
3. Make predictions
4. Run simulations
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run ShambaON ML system."""
    
    print("\n" + "=" * 80)
    print("🌾 SHAMBAON - FLOOD RISK & CROP RECOMMENDATION ML SYSTEM")
    print("=" * 80)
    print()
    
    try:
        # Step 1: Generate Data
        print("📊 STEP 1: Generating synthetic datasets...")
        print("-" * 80)
        
        from data.data_generator import main as generate_data
        datasets = generate_data()
        
        print("\n✅ Data generation complete!\n")
        
        # Step 2: Train Flood Risk Models
        print("🌊 STEP 2: Training flood risk prediction models...")
        print("-" * 80)
        
        from models.flood_risk_model import train_flood_risk_models
        flood_artifacts = train_flood_risk_models(data_dir='data', output_dir='models')
        
        print("\n✅ Flood risk models trained!\n")
        
        # Step 3: Train Crop Recommendation Models
        print("🌱 STEP 3: Training crop recommendation models...")
        print("-" * 80)
        
        from models.crop_recommendation_model import train_crop_recommendation_models
        crop_artifacts = train_crop_recommendation_models(data_dir='data', output_dir='models')
        
        print("\n✅ Crop recommendation models trained!\n")
        
        # Step 4: Make Example Prediction
        print("🔮 STEP 4: Making example predictions...")
        print("-" * 80)
        
        from inference import InferenceAPI
        
        api = InferenceAPI()
        
        # Check models loaded
        health = api.health_check()
        if health.get('status') != 'ready':
            print("⚠️  Models not fully loaded. Skipping predictions.")
        else:
            print("✅ Models loaded and ready!")
            print()
            
            # Example parcel
            example_parcel = {
                'parcel_id': 'DEMO_001',
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
                'irrigation_availability': 1
            }
            
            print(f"📍 Predicting for parcel: {example_parcel['parcel_id']}")
            print(f"   Location: {example_parcel['county']}")
            print()
            
            try:
                result = api.predict_parcel(example_parcel)
                
                # Extract key results
                flood_info = result.get('prediction', {}).get('flood_risk', {})
                crop_info = result.get('prediction', {}).get('crop_recommendations', {})
                
                print("🌊 FLOOD RISK ASSESSMENT:")
                print(f"   Risk Score: {flood_info.get('flood_risk_score', 0):.3f} (0-1)")
                print(f"   Risk Class: {flood_info.get('risk_class', 'Unknown')}")
                print(f"   Confidence: {flood_info.get('confidence', 0):.1%}")
                print()
                print(f"📋 Recommendation:")
                print(f"   {flood_info.get('recommendation', 'N/A')}")
                print()
                
                print("🌱 CROP RECOMMENDATIONS:")
                recs = crop_info.get('recommendations', {}).get('consolidated', [])
                for i, rec in enumerate(recs[:3], 1):
                    print(f"   {i}. {rec.get('crop', 'Unknown')} (votes: {rec.get('votes', 0)})")
                
            except Exception as e:
                print(f"⚠️  Prediction error: {e}")
        
        print("\n" + "=" * 80)
        print("✅ QUICK START COMPLETE!")
        print("=" * 80)
        print()
        print("📚 Next steps:")
        print("   1. Review trained models in 'models/' directory")
        print("   2. Check generated datasets in 'data/' directory")
        print("   3. Read README.md for detailed documentation")
        print("   4. Run 'python inference.py' for batch predictions")
        print("   5. Deploy to Streamlit dashboard for visualization")
        print()
        print("🔗 Integration options:")
        print("   - Streamlit: For web dashboard")
        print("   - Supabase: For data backend")
        print("   - Africa's Talking: For USSD/SMS alerts")
        print()
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print("\n❌ Error occurred. Check logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()
