#!/usr/bin/env python3
"""
Vahan-Insight: Data Cleaning Pipeline
======================================
Transforms raw vehicle registration data into analysis-ready format.

Features:
- Schema enforcement (datetime, categorical types)
- Manufacturer normalization (fuzzy matching)
- State extraction from RTO codes
- Fuel categorization (EV/ICE/Hybrid)

Usage:
    python data_cleaning.py

Author: Vahan-Insight Team
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
INPUT_FILE = DATA_DIR / "vehicle_registrations_500k.csv"
OUTPUT_FILE = DATA_DIR / "vehicle_registrations_cleaned.csv"


def load_config() -> Tuple[Dict, Dict]:
    """Load configuration files for RTO and manufacturer mappings."""
    
    with open(CONFIG_DIR / "rto_state_mapping.json", 'r') as f:
        rto_mapping = json.load(f)
    
    with open(CONFIG_DIR / "manufacturer_mapping.json", 'r') as f:
        manufacturer_config = json.load(f)
    
    return rto_mapping, manufacturer_config


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce proper data types on the dataframe.
    
    - Convert registrationMonthMMYY to proper datetime
    - Set categorical columns
    """
    logger.info("⏳ Enforcing schema...")
    
    # Convert MM-YY format to datetime
    def parse_mmyy(val):
        try:
            if pd.isna(val) or val == '':
                return pd.NaT
            # Format is MM-YY (e.g., "03-23" for March 2023)
            parts = str(val).split('-')
            if len(parts) == 2:
                month, year = int(parts[0]), int(parts[1])
                # Convert 2-digit year to 4-digit (assume 2000s)
                full_year = 2000 + year if year < 100 else year
                return pd.Timestamp(year=full_year, month=month, day=1)
            return pd.NaT
        except (ValueError, TypeError):
            return pd.NaT
    
    df['registration_date'] = df['registrationMonthMMYY'].apply(parse_mmyy)
    
    # Extract year and month for easier analysis
    df['reg_year'] = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month
    
    # Set categorical columns
    categorical_cols = ['fuelName', 'vehicleCategoryName', 'vehicleClassName', 
                        'pollutionNorm', 'saleType', 'stateName']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    logger.info("✓ Schema enforcement complete")
    return df


def extract_state_from_rto(df: pd.DataFrame, rto_mapping: Dict) -> pd.DataFrame:
    """
    Extract state from RTO code prefix.
    
    RTO codes follow pattern: XX## where XX is state code (e.g., MH40 = Maharashtra)
    """
    logger.info("⏳ Extracting state from RTO codes...")
    
    def get_state_from_rto(rto_code):
        if pd.isna(rto_code) or rto_code == '':
            return 'Unknown'
        
        # Extract first 2 letters (state code)
        rto_str = str(rto_code).upper().strip()
        state_code = ''.join(filter(str.isalpha, rto_str[:2]))
        
        return rto_mapping.get(state_code, 'Unknown')
    
    df['state_from_rto'] = df['rtoCode'].apply(get_state_from_rto)
    
    # Count unknown states
    unknown_count = (df['state_from_rto'] == 'Unknown').sum()
    if unknown_count > 0:
        logger.warning(f"   {unknown_count:,} records have unknown state from RTO")
    
    logger.info("✓ State extraction complete")
    return df


def normalize_manufacturers(df: pd.DataFrame, manufacturer_config: Dict) -> pd.DataFrame:
    """
    Normalize manufacturer names using fuzzy matching.
    
    Groups variations like "MARUTI SUZUKI", "MARUTI SUZUKI INDIA", etc. into "MARUTI"
    """
    logger.info("⏳ Normalizing manufacturer names...")
    
    manufacturer_groups = manufacturer_config.get('manufacturer_groups', {})
    
    # Create reverse mapping: variant -> normalized name
    variant_to_normalized = {}
    for normalized_name, variants in manufacturer_groups.items():
        for variant in variants:
            variant_to_normalized[variant.upper()] = normalized_name
    
    # Get all known variants for fuzzy matching
    all_known_variants = list(variant_to_normalized.keys())
    
    def normalize_maker(maker_name):
        if pd.isna(maker_name) or maker_name == '':
            return 'OTHER'
        
        maker_upper = str(maker_name).upper().strip()
        
        # Direct match first
        if maker_upper in variant_to_normalized:
            return variant_to_normalized[maker_upper]
        
        # Fuzzy match with high threshold
        if all_known_variants:
            match = process.extractOne(
                maker_upper, 
                all_known_variants, 
                scorer=fuzz.token_set_ratio,
                score_cutoff=85
            )
            if match:
                return variant_to_normalized[match[0]]
        
        # No match - return original (uppercase) for tracking
        return maker_upper
    
    df['maker_normalized'] = df['makerName'].apply(normalize_maker)
    
    # Report stats
    original_unique = df['makerName'].nunique()
    normalized_unique = df['maker_normalized'].nunique()
    logger.info(f"   Reduced manufacturers: {original_unique:,} → {normalized_unique:,}")
    
    logger.info("✓ Manufacturer normalization complete")
    return df


def categorize_fuel(df: pd.DataFrame, manufacturer_config: Dict) -> pd.DataFrame:
    """
    Categorize fuel types into broader categories.
    
    Categories: EV, HYBRID, CNG, LPG, PETROL, DIESEL, OTHER
    """
    logger.info("⏳ Categorizing fuel types...")
    
    fuel_categories = manufacturer_config.get('fuel_categories', {})
    
    # Create reverse mapping
    fuel_to_category = {}
    for category, fuel_types in fuel_categories.items():
        for fuel_type in fuel_types:
            fuel_to_category[fuel_type.upper()] = category
    
    def get_fuel_category(fuel_name):
        if pd.isna(fuel_name) or fuel_name == '':
            return 'OTHER'
        
        fuel_upper = str(fuel_name).upper().strip()
        return fuel_to_category.get(fuel_upper, 'OTHER')
    
    df['fuel_category'] = df['fuelName'].apply(get_fuel_category)
    
    # Add EV flag for easier filtering
    df['is_ev'] = df['fuel_category'] == 'EV'
    df['is_hybrid'] = df['fuel_category'] == 'HYBRID'
    
    # Report distribution
    fuel_dist = df['fuel_category'].value_counts()
    logger.info("   Fuel category distribution:")
    for cat, count in fuel_dist.head(5).items():
        pct = count / len(df) * 100
        logger.info(f"     {cat}: {count:,} ({pct:.1f}%)")
    
    logger.info("✓ Fuel categorization complete")
    return df


def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """Validate the cleaned dataframe."""
    logger.info("⏳ Validating cleaned data...")
    
    issues = []
    
    # Check for required columns
    required_cols = ['registration_date', 'state_from_rto', 'maker_normalized', 'fuel_category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for null values in key columns
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df) * 100
            if null_pct > 5:
                issues.append(f"{col}: {null_pct:.1f}% null values")
    
    if issues:
        logger.warning("⚠ Validation issues found:")
        for issue in issues:
            logger.warning(f"   - {issue}")
        return False
    
    logger.info("✓ Validation passed")
    return True


def clean_aggregated_dataset(manufacturer_config: Dict) -> pd.DataFrame:
    """
    Clean the state_month_fuel aggregated dataset.
    
    Applies:
    - Schema enforcement (datetime)
    - Fuel categorization
    """
    aggregated_input = DATA_DIR / "state_month_fuel_aggregated.csv"
    aggregated_output = DATA_DIR / "state_month_fuel_cleaned.csv"
    
    logger.info("\n" + "=" * 60)
    logger.info("🧹 Cleaning Aggregated Dataset")
    logger.info("=" * 60)
    logger.info(f"Input: {aggregated_input}")
    
    # Load data
    df = pd.read_csv(aggregated_input)
    logger.info(f"✓ Loaded {len(df):,} records")
    
    # Parse dates
    def parse_mmyy(val):
        try:
            if pd.isna(val) or val == '':
                return pd.NaT
            parts = str(val).split('-')
            if len(parts) == 2:
                month, year = int(parts[0]), int(parts[1])
                full_year = 2000 + year if year < 100 else year
                return pd.Timestamp(year=full_year, month=month, day=1)
            return pd.NaT
        except (ValueError, TypeError):
            return pd.NaT
    
    df['registration_date'] = df['registrationMonthMMYY'].apply(parse_mmyy)
    df['reg_year'] = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month
    logger.info("✓ Schema enforcement complete")
    
    # Categorize fuel
    fuel_categories = manufacturer_config.get('fuel_categories', {})
    fuel_to_category = {}
    for category, fuel_types in fuel_categories.items():
        for fuel_type in fuel_types:
            fuel_to_category[fuel_type.upper()] = category
    
    df['fuel_category'] = df['fuelName'].apply(
        lambda x: fuel_to_category.get(str(x).upper().strip(), 'OTHER') if pd.notna(x) else 'OTHER'
    )
    df['is_ev'] = df['fuel_category'] == 'EV'
    df['is_hybrid'] = df['fuel_category'] == 'HYBRID'
    logger.info("✓ Fuel categorization complete")
    
    # Set categorical columns
    df['stateName'] = df['stateName'].astype('category')
    df['fuelName'] = df['fuelName'].astype('category')
    df['fuel_category'] = df['fuel_category'].astype('category')
    
    # Save
    df.to_csv(aggregated_output, index=False)
    output_size_mb = aggregated_output.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved to {aggregated_output} ({output_size_mb:.2f} MB)")
    
    # Summary
    logger.info(f"\n  Records: {len(df):,}")
    logger.info(f"  Date range: {df['registration_date'].min()} to {df['registration_date'].max()}")
    logger.info(f"  States: {df['stateName'].nunique()}")
    ev_count = df[df['is_ev']]['vehicleCount'].sum()
    total_count = df['vehicleCount'].sum()
    logger.info(f"  EV registrations: {ev_count:,} ({ev_count/total_count*100:.1f}%)")
    logger.info("=" * 60)
    
    return df


def clean_dataset() -> pd.DataFrame:
    """
    Main pipeline orchestrator.
    
    Runs all cleaning steps in sequence for both datasets.
    """
    logger.info("=" * 60)
    logger.info("🧹 Vahan-Insight: Data Cleaning Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info("=" * 60 + "\n")
    
    # Load data
    logger.info("Step 1/5: Loading data...")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Load configs
    logger.info("\nStep 2/5: Loading configuration...")
    rto_mapping, manufacturer_config = load_config()
    logger.info("✓ Configuration loaded")
    
    # Run transformations
    logger.info("\nStep 3/5: Running transformations...")
    df = enforce_schema(df)
    df = extract_state_from_rto(df, rto_mapping)
    df = normalize_manufacturers(df, manufacturer_config)
    df = categorize_fuel(df, manufacturer_config)
    
    # Validate
    logger.info("\nStep 4/5: Validating data...")
    validate_cleaned_data(df)
    
    # Save
    logger.info("\nStep 5/5: Saving cleaned data...")
    df.to_csv(OUTPUT_FILE, index=False)
    output_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved to {OUTPUT_FILE} ({output_size_mb:.2f} MB)")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Vehicle Registrations Cleaning Summary:")
    logger.info("=" * 60)
    logger.info(f"  Records processed: {len(df):,}")
    logger.info(f"  Date range: {df['registration_date'].min()} to {df['registration_date'].max()}")
    logger.info(f"  Unique manufacturers (normalized): {df['maker_normalized'].nunique()}")
    logger.info(f"  States covered: {df['state_from_rto'].nunique()}")
    logger.info(f"  EV registrations: {df['is_ev'].sum():,} ({df['is_ev'].mean()*100:.1f}%)")
    logger.info("=" * 60)
    
    # Also clean aggregated dataset
    df_agg = clean_aggregated_dataset(manufacturer_config)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All data cleaning completed successfully!")
    logger.info("=" * 60)
    
    return df


if __name__ == "__main__":
    try:
        df = clean_dataset()
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Run 'python ingest_data.py' first to download the dataset.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

