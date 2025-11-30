"""
Data loading and validation module with error handling.
"""

import os
import logging
from typing import Optional, Tuple, List
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and validation of CSV data files."""
    
    def __init__(self, root_dir: str = "."):
        """
        Initialize DataLoader.
        
        Args:
            root_dir: Root directory for data files
        """
        self.root_dir = root_dir
        self.merged_df: Optional[pd.DataFrame] = None
        self.soil_df: Optional[pd.DataFrame] = None
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_csv(_self, path: str) -> Optional[pd.DataFrame]:
        """
        Load CSV file with comprehensive error handling.
        
        Args:
            path: Path to CSV file (relative to root_dir or absolute)
            
        Returns:
            DataFrame if successful, None otherwise
        """
        full_path = os.path.join(_self.root_dir, path) if not os.path.isabs(path) else path
        
        try:
            if not os.path.exists(full_path):
                logger.error(f"File not found: {full_path}")
                st.error(f"❌ Data file not found: {full_path}")
                return None
            
            df = pd.read_csv(full_path, low_memory=False)
            
            if df.empty:
                logger.warning(f"Empty CSV file: {full_path}")
                st.warning(f"⚠️ Empty data file: {full_path}")
                return None
            
            logger.info(f"Successfully loaded {len(df)} rows from {full_path}")
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {full_path}")
            st.error(f"❌ Empty data file: {full_path}")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error in {full_path}: {str(e)}")
            st.error(f"❌ Error parsing CSV file: {str(e)}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error loading {full_path}")
            st.error(f"❌ Error loading data: {str(e)}")
            return None
    
    def find_column(self, df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
        """
        Find column in DataFrame by name (case-insensitive).
        
        Args:
            df: DataFrame to search
            candidates: List of possible column names
            
        Returns:
            First matching column name or None
        """
        if df is None:
            return None
        
        cols = {c.lower(): c for c in df.columns}
        for name in candidates:
            if name.lower() in cols:
                return cols[name.lower()]
        return None
    
    def validate_data(self, df: Optional[pd.DataFrame], required_cols: List[List[str]]) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_cols: List of lists, each containing candidate column names
            
        Returns:
            (is_valid, missing_cols) tuple
        """
        if df is None or df.empty:
            return False, ["DataFrame is None or empty"]
        
        missing = []
        for col_candidates in required_cols:
            found = self.find_column(df, col_candidates)
            if found is None:
                missing.append(f"One of: {col_candidates}")
        
        return len(missing) == 0, missing
    
    def load_all_data(
        self, 
        merged_csv: str = "merged_final.csv",
        soil_csv: str = "soil_ml_ready.csv"
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load all required data files with validation.
        
        Args:
            merged_csv: Path to merged data CSV
            soil_csv: Path to soil data CSV
            
        Returns:
            (merged_df, soil_df) tuple
        """
        logger.info("Loading data files...")
        
        # Load merged data
        self.merged_df = self.load_csv(merged_csv)
        if self.merged_df is None:
            st.error("❌ Failed to load merged_final.csv. App cannot continue.")
            st.stop()
        
        # Validate merged data
        is_valid, missing = self.validate_data(
            self.merged_df,
            [["state", "state_name", "State"], ["district", "districtname", "District"]]
        )
        if not is_valid:
            st.error(f"❌ Missing required columns in merged data: {missing}")
            st.stop()
        
        # Load soil data (optional)
        self.soil_df = self.load_csv(soil_csv)
        if self.soil_df is None:
            logger.warning("Soil data not available - some features may be limited")
            st.warning("⚠️ Soil data file not found - some features may be limited")
        
        logger.info("Data loading complete")
        return self.merged_df, self.soil_df

