#!/usr/bin/env python3
"""
Vahan-Insight: Automated Data Ingestion Pipeline
=================================================
Downloads the Indian Vehicle Registration dataset (2020-25) from Kaggle.

Usage:
    export KAGGLE_API_TOKEN="your_kaggle_token"
    python ingest_data.py

Author: Vahan-Insight Team
"""

import os
import sys
import json
import zipfile
import logging
import stat
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATASET_IDENTIFIER = "aatifahmad123/indian-vehicle-registration-data-202025"
DATA_DIR = Path(__file__).parent / "data"


def setup_kaggle_credentials():
    """
    Set up Kaggle API credentials.
    Supports multiple sources:
    1. Streamlit secrets (for cloud deployment)
    2. Environment variables
    3. .env file
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Try to get credentials from Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'KAGGLE_USERNAME' in st.secrets:
            username = st.secrets['KAGGLE_USERNAME']
            key = st.secrets['KAGGLE_KEY']
            logger.info("✓ Using Kaggle credentials from Streamlit secrets")
            
            # Create kaggle.json for API
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            with open(kaggle_json, 'w') as f:
                json.dump({"username": username, "key": key}, f)
            kaggle_json.chmod(0o600)
            return
    except (ImportError, AttributeError, KeyError):
        pass  # Streamlit not available or secrets not set
    
    # Check if kaggle.json already exists
    if kaggle_json.exists():
        logger.info("✓ Using existing ~/.kaggle/kaggle.json")
        return
    
    # Try environment variables
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY') or os.getenv('KAGGLE_API_TOKEN')
    
    if username and key:
        logger.info("✓ Using Kaggle credentials from environment variables")
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, 'w') as f:
            json.dump({"username": username, "key": key}, f)
        kaggle_json.chmod(0o600)
        return
    
    # Try .env file
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY') or os.getenv('KAGGLE_API_TOKEN')
        
        if username and key:
            logger.info("✓ Using Kaggle credentials from .env file")
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            with open(kaggle_json, 'w') as f:
                json.dump({"username": username, "key": key}, f)
            kaggle_json.chmod(0o600)
            return
    
    raise EnvironmentError(
        "Kaggle credentials not found! Please set up credentials via:\n"
        "1. Streamlit secrets (for cloud): KAGGLE_USERNAME and KAGGLE_KEY\n"
        "2. Environment variables: KAGGLE_USERNAME and KAGGLE_KEY\n"
        "3. .env file with KAGGLE_USERNAME and KAGGLE_KEY"
    )


def download_dataset() -> bool:
    """
    Download the Indian Vehicle Registration dataset from Kaggle.
    
    Returns:
        bool: True if download was successful
    """
    try:
        # Import kaggle here after credentials are set up
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize and authenticate
        api = KaggleApi()
        api.authenticate()
        logger.info("✓ Kaggle API authenticated successfully")
        
        # Create data directory
        DATA_DIR.mkdir(exist_ok=True)
        logger.info(f"✓ Data directory created: {DATA_DIR}")
        
        # Download dataset
        logger.info(f"⏳ Downloading dataset: {DATASET_IDENTIFIER}")
        logger.info("   This may take a few minutes depending on your connection...")
        
        api.dataset_download_files(
            dataset=DATASET_IDENTIFIER,
            path=str(DATA_DIR),
            unzip=False  # We'll handle extraction ourselves for better control
        )
        
        logger.info("✓ Dataset downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download dataset: {str(e)}")
        logger.info("\nTroubleshooting tips:")
        logger.info("  1. Verify your KAGGLE_API_TOKEN is correct")
        logger.info("  2. Check if the dataset exists: https://www.kaggle.com/datasets/" + DATASET_IDENTIFIER)
        logger.info("  3. Ensure you have accepted any dataset terms on Kaggle")
        return False


def extract_and_cleanup() -> bool:
    """
    Extract downloaded zip files and remove the archives.
    
    Returns:
        bool: True if extraction was successful
    """
    try:
        zip_files = list(DATA_DIR.glob("*.zip"))
        
        if not zip_files:
            logger.warning("No zip files found to extract")
            return True
        
        for zip_path in zip_files:
            logger.info(f"⏳ Extracting: {zip_path.name}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            
            # Remove the zip file after extraction
            zip_path.unlink()
            logger.info(f"✓ Extracted and removed: {zip_path.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to extract files: {str(e)}")
        return False


def validate_data() -> bool:
    """
    Validate that the downloaded data files exist and are readable.
    
    Returns:
        bool: True if validation passed
    """
    try:
        # Check for CSV files first, then any other data files
        data_files = list(DATA_DIR.glob("*.csv"))
        if not data_files:
            data_files = list(DATA_DIR.glob("*.*"))
            data_files = [f for f in data_files if f.is_file()]
        
        if not data_files:
            logger.error("✗ No data files found in data directory!")
            return False
        
        logger.info(f"\n{'='*50}")
        logger.info("📊 Downloaded Files Summary:")
        logger.info(f"{'='*50}")
        
        total_size = 0
        for data_file in data_files:
            size_mb = data_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            # Count rows for CSV files
            if data_file.suffix.lower() == '.csv':
                try:
                    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                        row_count = sum(1 for _ in f) - 1  # Subtract header
                    logger.info(f"  📄 {data_file.name}")
                    logger.info(f"     Size: {size_mb:.2f} MB | Rows: {row_count:,}")
                except Exception:
                    logger.info(f"  📄 {data_file.name}")
                    logger.info(f"     Size: {size_mb:.2f} MB")
            else:
                logger.info(f"  📄 {data_file.name}")
                logger.info(f"     Size: {size_mb:.2f} MB")
        
        logger.info(f"{'='*50}")
        logger.info(f"📦 Total: {len(data_files)} file(s), {total_size:.2f} MB")
        logger.info(f"{'='*50}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Validation failed: {str(e)}")
        return False


def main():
    """Main entry point for the data ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("🚗 Vahan-Insight: Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Dataset: {DATASET_IDENTIFIER}")
    logger.info(f"Target Directory: {DATA_DIR}")
    logger.info("=" * 60 + "\n")
    
    # Step 1: Set up credentials
    logger.info("Step 1/4: Setting up Kaggle credentials...")
    if not setup_kaggle_credentials():
        sys.exit(1)
    
    # Step 2: Download dataset
    logger.info("\nStep 2/4: Downloading dataset...")
    if not download_dataset():
        sys.exit(1)
    
    # Step 3: Extract and cleanup
    logger.info("\nStep 3/4: Extracting files...")
    if not extract_and_cleanup():
        sys.exit(1)
    
    # Step 4: Validate data
    logger.info("\nStep 4/4: Validating downloaded data...")
    if not validate_data():
        sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Data ingestion completed successfully!")
    logger.info("=" * 60)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Explore data: pandas.read_csv('{DATA_DIR}/<filename>.csv')")
    logger.info(f"  2. Run Phase 2: EV Revolution Deep Dive analysis")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
