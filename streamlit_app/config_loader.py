"""
Configuration loading utilities.
"""

import os
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "data": {
            "merged_csv": "./merged_final.csv",
            "soil_csv": "./soil_ml_ready.csv",
            "model_dir": "./model_outputs"
        },
        "features": [
            "Month_num", "month_sin", "month_cos", "Season_num",
            "state_freq", "district_id", "lag_1", "lag_7",
            "rolling_3", "rolling_6"
        ],
        "models": {
            "preferred": ["LightGBM", "XGBoost"],
            "ensemble": True
        },
        "app": {
            "cache_ttl": 3600,
            "page_title": "Soil Moisture Predictor",
            "page_layout": "wide"
        }
    }
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults to ensure all keys exist
        merged_config = default_config.copy()
        for key, value in config.items():
            if isinstance(value, dict) and key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        logger.info(f"Configuration loaded from {config_path}")
        return merged_config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}, using defaults")
        return default_config

