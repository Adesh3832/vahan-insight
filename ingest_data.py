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


def setup_kaggle_credentials() -> bool:
    """
    Set up Kaggle credentials from KAGGLE_API_TOKEN environment variable.
    
    Supports both formats:
    - New format: KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    - Legacy format: username:api_key or just api_key
    
    Returns:
        bool: True if credentials were set up successfully
    """
    token = os.environ.get("KAGGLE_API_TOKEN")
    
    if not token:
        logger.error("KAGGLE_API_TOKEN environment variable not set!")
        logger.info("Set it with: export KAGGLE_API_TOKEN='your_token_here'")
        return False
    
    # Create kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # For KGAT_ tokens (new format from Kaggle 2024+)
    # We need to create a kaggle.json with the token as the key
    # The username can be empty or a placeholder for KGAT_ tokens
    if token.startswith("KGAT_"):
        credentials = {"username": "__token__", "key": token}
        logger.info("✓ Kaggle API token (KGAT_ format) detected")
    elif ":" in token:
        # Legacy format: username:api_key
        username, api_key = token.split(":", 1)
        credentials = {"username": username, "key": api_key}
        logger.info("✓ Kaggle credentials (legacy format) detected")
    else:
        # Assume it's just an API key
        username = os.environ.get("KAGGLE_USERNAME", "__token__")
        credentials = {"username": username, "key": token}
        logger.info("✓ Kaggle API key detected")
    
    # Write credentials to kaggle.json
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)
    
    # Set proper permissions (600) - required by Kaggle API
    os.chmod(kaggle_json, stat.S_IRUSR | stat.S_IWUSR)
    
    logger.info(f"✓ Credentials saved to {kaggle_json}")
    return True


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
