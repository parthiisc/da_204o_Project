"""
Model loading and prediction utilities with error handling.
"""

import os
import logging
from typing import Optional, Dict, Any
import joblib
import streamlit as st
from pathlib import Path

# Try to import downloader (optional). If manifest isn't present this is a noop.
try:
    from streamlit_app import model_downloader as _model_downloader
except Exception:
    _model_downloader = None

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model loading and predictions."""
    
    def __init__(self, model_dir: str = "./model_outputs"):
        """
        Initialize ModelManager.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.load_models()
    
    def load_models(self) -> None:
        """Load all available models from model directory."""
        if not os.path.isdir(self.model_dir):
            logger.warning(f"Model directory not found: {self.model_dir}")
            st.warning(f"⚠️ Model directory not found: {self.model_dir}")
            return
        # If directory exists but is empty, attempt to download files from a manifest
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".joblib")]
        if not model_files and _model_downloader is not None:
            try:
                _model_downloader.try_download_all(self.model_dir)
                model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".joblib")]
            except Exception as e:
                logger.warning("Downloader attempt failed: %s", e)
        
        if not model_files:
            logger.warning(f"No model files found in {self.model_dir}")
            st.warning(f"⚠️ No model files (.joblib) found in {self.model_dir}")
            return
        
        for model_file in model_files:
            model_path = os.path.join(self.model_dir, model_file)
            model_name = self._identify_model(model_file)

            # Only load XGBoost and LightGBM models
            if model_name in ("XGBoost", "LightGBM"):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Successfully loaded {model_name} from {model_file}")
                except FileNotFoundError:
                    logger.warning(f"Model file not found: {model_path}")
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {str(e)}")
                    st.warning(f"⚠️ Could not load {model_file}: {str(e)}")
            else:
                logger.info(f"Skipping non-XGBoost/LightGBM model file: {model_file}")
    
    def _identify_model(self, filename: str) -> Optional[str]:
        """
        Identify model type from filename.
        
        Args:
            filename: Model filename
            
        Returns:
            Model name or None
        """
        filename_lower = filename.lower()
        if "light" in filename_lower or "lgbm" in filename_lower:
            return "LightGBM"
        if "xgb" in filename_lower or "xgboost" in filename_lower:
            return "XGBoost"
        return None
    
    def predict(self, features, model_name: Optional[str] = None) -> Optional[float]:
        """
        Make prediction using specified model or ensemble.
        
        Args:
            features: Feature array for prediction
            model_name: Specific model to use, or None for ensemble
            
        Returns:
            Prediction value or None
        """
        if not self.models:
            logger.error("No models available for prediction")
            return None
        
        if model_name:
            if model_name in self.models:
                try:
                    return float(self.models[model_name].predict(features)[0])
                except Exception as e:
                    logger.error(f"Prediction error with {model_name}: {str(e)}")
                    return None
            else:
                logger.warning(f"Model {model_name} not found")
                return None
        
        # Ensemble prediction
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = float(model.predict(features)[0])
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {str(e)}")
        
        if not predictions:
            return None
        
        # Return average of available predictions
        return float(sum(predictions.values()) / len(predictions))
    
    def get_all_predictions(self, features) -> Dict[str, Optional[float]]:
        """
        Get predictions from all available models.
        
        Args:
            features: Feature array for prediction
            
        Returns:
            Dictionary of model_name: prediction
        """
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = float(model.predict(features)[0])
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {str(e)}")
                predictions[name] = None
        
        # Add ensemble if multiple models
        if len([v for v in predictions.values() if v is not None]) > 1:
            valid_preds = [v for v in predictions.values() if v is not None]
            predictions['Ensemble'] = float(sum(valid_preds) / len(valid_preds))
        
        return predictions

