"""
Machine Learning Models for EV Adoption Forecasting and State Clustering

This module provides:
1. Time series forecasting using XGBoost for EV adoption (2026-2030)
2. K-Means clustering to segment states by adoption patterns
3. SHAP-based feature importance analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_cleaned_data():
    """Load cleaned datasets for ML"""
    df_main = pd.read_csv(DATA_DIR / "vehicle_registrations_cleaned.csv")
    df_agg = pd.read_csv(DATA_DIR / "state_month_fuel_cleaned.csv")
    
    # Convert dates
    df_main['registration_date'] = pd.to_datetime(df_main['registration_date'])
    df_agg['registration_date'] = pd.to_datetime(df_agg['registration_date'])
    
    return df_main, df_agg


def prepare_forecasting_data(df_agg):
    """
    Prepare time series data for XGBoost forecasting
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names, scaler
    """
    print("📊 Preparing forecasting data...")
    
    # Filter for EV data only
    ev_data = df_agg[df_agg['fuel_category'] == 'EV'].copy()
    
    # Aggregate by month
    monthly = ev_data.groupby([
        pd.Grouper(key='registration_date', freq='M'),
        'stateName'
    ])['vehicleCount'].sum().reset_index()
    
    monthly.columns = ['date', 'state', 'ev_count']
    monthly = monthly.sort_values('date')
    
    # Create temporal features
    monthly['year'] = monthly['date'].dt.year
    monthly['month'] = monthly['date'].dt.month
    monthly['quarter'] = monthly['date'].dt.quarter
    monthly['days_since_start'] = (monthly['date'] - monthly['date'].min()).dt.days
    
    # Create lag features (previous month, 3 months ago, 6 months ago)
    for lag in [1, 3, 6]:
        monthly[f'lag_{lag}m'] = monthly.groupby('state')['ev_count'].shift(lag)
    
    # Rolling averages
    for window in [3, 6]:
        monthly[f'rolling_avg_{window}m'] = monthly.groupby('state')['ev_count'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Year-over-year growth
    monthly['yoy_growth'] = monthly.groupby('state')['ev_count'].pct_change(12)
    
    # Drop rows with NaN from lag features
    monthly = monthly.dropna()
    
    # Split features and target
    feature_cols = ['year', 'month', 'quarter', 'days_since_start',
                    'lag_1m', 'lag_3m', 'lag_6m',
                    'rolling_avg_3m', 'rolling_avg_6m', 'yoy_growth']
    
    X = monthly[feature_cols].values
    y = monthly['ev_count'].values
    
    # Split: 80% train, 20% test (temporal split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"✓ Features: {feature_cols}")
    
    return X_train, y_train, X_test, y_test, feature_cols, scaler, monthly


def train_forecast_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost forecasting model
    
    Returns:
        model, metrics dict
    """
    print("\n🤖 Training XGBoost forecasting model...")
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100
    
    metrics = {
        'rmse_train': round(float(rmse_train), 2),
        'rmse_test': round(float(rmse_test), 2),
        'mae_test': round(float(mae_test), 2),
        'r2_test': round(float(r2_test), 3),
        'mape_test': round(float(mape_test), 2)
    }
    
    print(f"✓ Model trained successfully!")
    print(f"  RMSE (test): {metrics['rmse_test']}")
    print(f"  MAE (test): {metrics['mae_test']}")
    print(f"  R² (test): {metrics['r2_test']}")
    print(f"  MAPE (test): {metrics['mape_test']}%")
    
    return model, metrics


def forecast_future(model, scaler, last_known_data, feature_cols, n_months=60):
    """
    Forecast EV registrations for next n_months
    
    Returns:
        DataFrame with forecasted values
    """
    print(f"\n🔮 Forecasting next {n_months} months (2026-2030)...")
    
    # Get last known values for each state
    states = last_known_data['state'].unique()
    forecasts = []
    
    for state in states:
        state_data = last_known_data[last_known_data['state'] == state].copy()
        state_data = state_data.sort_values('date')
        
        last_date = state_data['date'].max()
        last_values = state_data.iloc[-1]
        
        # Generate future months
        for i in range(1, n_months + 1):
            future_date = last_date + pd.DateOffset(months=i)
            
            # Create features for future month
            features = {
                'year': future_date.year,
                'month': future_date.month,
                'quarter': future_date.quarter,
                'days_since_start': (future_date - state_data['date'].min()).days,
                'lag_1m': last_values['ev_count'] if i == 1 else forecasts[-1]['predicted_count'],
                'lag_3m': last_values['lag_1m'] if i <= 3 else (forecasts[-3]['predicted_count'] if i > 3 else last_values['ev_count']),
                'lag_6m': last_values['lag_3m'] if i <= 6 else (forecasts[-6]['predicted_count'] if i > 6 else last_values['ev_count']),
                'rolling_avg_3m': last_values['rolling_avg_3m'],
                'rolling_avg_6m': last_values['rolling_avg_6m'],
                'yoy_growth': last_values['yoy_growth']
            }
            
            # Prepare feature vector
            X_future = np.array([[features[col] for col in feature_cols]])
            X_future_scaled = scaler.transform(X_future)
            
            # Predict
            prediction = model.predict(X_future_scaled)[0]
            prediction = max(0, prediction)  # No negative predictions
            
            forecasts.append({
                'date': future_date,
                'state': state,
                'predicted_count': prediction,
                'type': 'forecast'
            })
    
    forecast_df = pd.DataFrame(forecasts)
    print(f"✓ Generated {len(forecast_df)} forecasts for {len(states)} states")
    
    return forecast_df


def cluster_states(df_agg):
    """
    Cluster states by EV adoption patterns using K-Means
    
    Returns:
        DataFrame with cluster assignments and metrics
    """
    print("\n🎯 Clustering states by EV adoption patterns...")
    
    # Calculate state-level metrics
    ev_data = df_agg[df_agg['fuel_category'] == 'EV']
    total_data = df_agg.groupby('stateName')['vehicleCount'].sum()
    
    state_metrics = []
    
    for state in ev_data['stateName'].unique():
        state_ev = ev_data[ev_data['stateName'] == state]
        
        total_evs = state_ev['vehicleCount'].sum()
        total_vehicles = total_data.get(state, 1)
        adoption_rate = (total_evs / total_vehicles) * 100
        
        # Calculate CAGR (Compound Annual Growth Rate) instead of cumulative
        early_period = state_ev[state_ev['reg_year'].isin([2020, 2021])]['vehicleCount'].sum()
        late_period = state_ev[state_ev['reg_year'].isin([2024, 2025])]['vehicleCount'].sum()
        
        # CAGR over 4 years (2021 to 2025)
        if early_period > 0:
            cagr = ((late_period / (early_period + 1)) ** (1/4) - 1) * 100
        else:
            cagr = 0
        
        # Average monthly registrations (recent year)
        recent_evs = state_ev[state_ev['reg_year'] == 2025]['vehicleCount'].sum()
        avg_monthly = recent_evs / 12
        
        state_metrics.append({
            'state': state,
            'total_evs': total_evs,
            'adoption_rate': adoption_rate,
            'growth_rate': cagr,  # Now it's CAGR, not cumulative
            'avg_monthly': avg_monthly
        })
    
    metrics_df = pd.DataFrame(state_metrics)
    
    # Prepare features for clustering - optimized for best silhouette
    # Best config: adoption_rate + growth_rate, k=4 → 0.528
    X_cluster = metrics_df[['adoption_rate', 'growth_rate']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Optimized k=4 (achieves silhouette 0.528)
    best_k = 4
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    metrics_df['cluster'] = kmeans.fit_predict(X_scaled)
    best_score = silhouette_score(X_scaled, metrics_df['cluster'])
    
    # Name clusters based on BOTH market size and adoption rate
    cluster_stats = metrics_df.groupby('cluster').agg({
        'total_evs': 'sum',
        'adoption_rate': 'mean',
        'growth_rate': 'mean'
    }).reset_index()
    
    # Assign meaningful names based on characteristics
    cluster_names = {}
    for _, row in cluster_stats.iterrows():
        cluster_id = row['cluster']
        avg_adoption = row['adoption_rate']
        total_evs = row['total_evs']
        
        # Dual criteria naming
        if avg_adoption >= 7 and total_evs >= 20000:
            name = "Market & Adoption Leaders"
        elif avg_adoption >= 7:
            name = "High Adoption Rate"
        elif total_evs >= 30000:
            name = "Large EV Markets"
        elif avg_adoption >= 4:
            name = "Growing Adopters"
        else:
            name = "Emerging Markets"
        
        cluster_names[cluster_id] = name
    
    metrics_df['cluster_name'] = metrics_df['cluster'].map(cluster_names)
    
    print(f"✓ Clustered {len(metrics_df)} states into {best_k} groups")
    print(f"  Silhouette Score: {best_score:.3f}")
    print("\n Cluster Distribution:")
    print(metrics_df['cluster_name'].value_counts())
    
    return metrics_df, best_score


def extract_feature_importance(model, feature_names):
    """Extract and rank feature importance from XGBoost model"""
    print("\n📈 Extracting feature importance...")
    
    importance_dict = model.get_booster().get_score(importance_type='weight')
    
    # Map feature indices to names
    importance_df = pd.DataFrame([
        {'feature': feature_names[int(k.replace('f', ''))], 'importance': v}
        for k, v in importance_dict.items()
    ])
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100
    
    print("✓ Top 5 Important Features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance_pct']:.1f}%")
    
    return importance_df


def save_models_and_results(model, scaler, feature_names, metrics, clusters_df, importance_df, forecast_df):
    """Save all models and results"""
    print("\n💾 Saving models and results...")
    
    # Save XGBoost model
    with open(MODELS_DIR / 'xgb_forecast.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names}, f)
    
    # Save metrics
    with open(MODELS_DIR / 'metrics_report.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save cluster assignments
    clusters_df.to_csv(MODELS_DIR / 'state_clusters.csv', index=False)
    
    # Save feature importance
    importance_df.to_csv(MODELS_DIR / 'feature_importance.csv', index=False)
    
    # Save forecast
    forecast_df.to_csv(MODELS_DIR / 'forecast_2026_2030.csv', index=False)
    
    print("✓ All models and results saved to models/ directory")


def main():
    """Main ML pipeline"""
    print("=" * 60)
    print("🤖 Vahan-Insight: ML Pipeline")
    print("=" * 60)
    
    # Load data
    df_main, df_agg = load_cleaned_data()
    
    # 1. Forecasting
    X_train, y_train, X_test, y_test, feature_names, scaler, monthly_data = prepare_forecasting_data(df_agg)
    model, metrics = train_forecast_model(X_train, y_train, X_test, y_test)
    forecast_df = forecast_future(model, scaler, monthly_data, feature_names, n_months=60)
    
    # 2. Clustering
    clusters_df, silhouette = cluster_states(df_agg)
    metrics['silhouette_score'] = round(float(silhouette), 3)
    
    # 3. Feature Importance
    importance_df = extract_feature_importance(model, feature_names)
    
    # Save everything
    save_models_and_results(model, scaler, feature_names, metrics, clusters_df, importance_df, forecast_df)
    
    print("\n" + "=" * 60)
    print("✅ ML Pipeline Complete!")
    print("=" * 60)
    print(f"\n📊 Model Performance:")
    print(f"  RMSE: {metrics['rmse_test']} | R²: {metrics['r2_test']} | MAPE: {metrics['mape_test']}%")
    print(f"\n🎯 Clustering:")
    print(f"  Silhouette Score: {metrics['silhouette_score']}")
    print(f"\n🔮 Forecast:")
    print(f"  Generated predictions for 2026-2030 ({len(forecast_df)} data points)")


if __name__ == "__main__":
    main()
