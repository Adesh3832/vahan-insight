"""
Production-Ready EV Adoption Forecasting with Prophet

Features:
- National-level aggregated forecasting
- Confidence intervals (80% and 95%)
- Smooth historical-to-forecast transition
- Seasonality modeling (yearly + monthly)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_and_prepare_data():
    """Load and aggregate data at national level for forecasting"""
    print("📊 Loading and preparing national-level EV data...")
    
    df_agg = pd.read_csv(DATA_DIR / "state_month_fuel_cleaned.csv")
    df_agg['registration_date'] = pd.to_datetime(df_agg['registration_date'])
    
    # Filter for EV data only
    ev_data = df_agg[df_agg['fuel_category'] == 'EV'].copy()
    
    # Aggregate to national monthly level
    national_monthly = ev_data.groupby(
        pd.Grouper(key='registration_date', freq='ME')
    )['vehicleCount'].sum().reset_index()
    
    national_monthly.columns = ['ds', 'y']  # Prophet format
    
    # Remove incomplete months (very low counts)
    national_monthly = national_monthly[national_monthly['y'] > 100]
    
    print(f"✓ Prepared {len(national_monthly)} months of national EV data")
    print(f"  Date range: {national_monthly['ds'].min().date()} to {national_monthly['ds'].max().date()}")
    print(f"  Monthly range: {national_monthly['y'].min()} to {national_monthly['y'].max()}")
    
    return national_monthly


def train_prophet_model(df):
    """Train Prophet model on national EV data"""
    print("\n🔮 Training Prophet forecasting model...")
    
    # Calculate realistic cap for post-processing
    current_max = df['y'].max()
    cap = current_max * 3  # Cap at 3x current maximum (realistic 5-year ceiling)
    
    print(f"  Current max: {current_max:,.0f}, Growth cap: {cap:,.0f}")
    
    # Initialize Prophet with linear growth (more stable)
    model = Prophet(
        growth='linear',  # Linear growth, we'll cap afterward
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.02,  # Conservative trend changes
        seasonality_prior_scale=5,
        interval_width=0.95
    )
    
    # Add monthly seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
    
    # Fit the model
    model.fit(df)
    
    print("✓ Prophet model trained successfully!")
    
    return model, cap


def forecast_future(model, cap, periods=60):
    """Generate forecast with confidence intervals"""
    print(f"\n📈 Generating {periods}-month forecast (2026-2030)...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='ME')
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Apply cap to prevent unrealistic growth
    forecast['yhat'] = forecast['yhat'].clip(lower=0, upper=cap)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0, upper=cap)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0, upper=cap)
    
    print(f"✓ Generated forecast with 95% confidence intervals")
    print(f"  Forecast range: {forecast['yhat'].iloc[-periods:].min():.0f} to {forecast['yhat'].iloc[-periods:].max():.0f}")
    
    return forecast


def save_results(model, forecast, df_historical):
    """Save model and forecast results"""
    print("\n💾 Saving Prophet model and forecast...")
    
    # Save model
    with open(MODELS_DIR / 'prophet_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save forecast (relevant columns only)
    forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_output.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
    forecast_output.to_csv(MODELS_DIR / 'prophet_forecast.csv', index=False)
    
    # Save summary metrics
    historical_end = df_historical['ds'].max()
    forecast_only = forecast[forecast['ds'] > historical_end]
    
    metrics = {
        'model': 'Prophet',
        'training_samples': len(df_historical),
        'forecast_months': len(forecast_only),
        'historical_end': str(historical_end.date()),
        'forecast_start': str(forecast_only['ds'].min().date()),
        'forecast_end': str(forecast_only['ds'].max().date()),
        '2026_avg_monthly': round(forecast_only[forecast_only['ds'].dt.year == 2026]['yhat'].mean(), 0),
        '2030_avg_monthly': round(forecast_only[forecast_only['ds'].dt.year == 2030]['yhat'].mean(), 0),
        'growth_rate_5yr': round(
            (forecast_only[forecast_only['ds'].dt.year == 2030]['yhat'].mean() / 
             forecast_only[forecast_only['ds'].dt.year == 2026]['yhat'].mean() - 1) * 100, 1
        )
    }
    
    with open(MODELS_DIR / 'prophet_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✓ Saved: prophet_model.pkl, prophet_forecast.csv, prophet_metrics.json")
    
    return metrics


def main():
    """Main Prophet forecasting pipeline"""
    print("=" * 60)
    print("🔮 Vahan-Insight: Production Prophet Forecasting")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Train Prophet with logistic growth
    model, cap = train_prophet_model(df)
    
    # Generate forecast
    forecast = forecast_future(model, cap, periods=60)
    
    # Save results
    metrics = save_results(model, forecast, df)
    
    print("\n" + "=" * 60)
    print("✅ Prophet Forecasting Complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  Training data: {metrics['training_samples']} months")
    print(f"  Forecast: {metrics['forecast_start']} to {metrics['forecast_end']}")
    print(f"  2026 avg: {metrics['2026_avg_monthly']:,.0f} EVs/month")
    print(f"  2030 avg: {metrics['2030_avg_monthly']:,.0f} EVs/month")
    print(f"  5-year growth: {metrics['growth_rate_5yr']}%")


if __name__ == "__main__":
    main()
