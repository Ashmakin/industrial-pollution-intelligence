"""
ETL Pipeline for Water Quality Data Processing

Handles missing value imputation, outlier detection, and feature engineering
for water quality time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityETL:
    """ETL pipeline for water quality data preprocessing"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.outlier_detector = IsolationForest(contamination=contamination, random_state=42)
        
    def load_data(self, db_url: str, days_back: int = 30) -> pd.DataFrame:
        """Load water quality data from database"""
        query = """
        SELECT 
            wqm.*,
            wqs.station_name,
            wqs.province,
            wqs.watershed,
            wqs.area_id
        FROM water_quality_measurements wqm
        JOIN water_quality_stations wqs ON wqm.station_id = wqs.id
        WHERE wqm.monitoring_time >= NOW() - INTERVAL '%s days'
        ORDER BY wqs.station_name, wqm.monitoring_time
        """ % days_back
        
        df = pd.read_sql(query, db_url)
        df['monitoring_time'] = pd.to_datetime(df['monitoring_time'])
        
        logger.info(f"Loaded {len(df)} records from {df['station_name'].nunique()} stations")
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using statistical and ML methods"""
        df = df.copy()
        
        # Numerical columns for outlier detection
        numerical_cols = [
            'temperature', 'ph', 'dissolved_oxygen', 'conductivity', 
            'turbidity', 'permanganate_index', 'ammonia_nitrogen',
            'total_phosphorus', 'total_nitrogen', 'chlorophyll_a', 'algae_density'
        ]
        
        # Remove columns with too many missing values
        valid_cols = [col for col in numerical_cols if col in df.columns and df[col].notna().sum() > len(df) * 0.1]
        
        if len(valid_cols) < 2:
            logger.warning("Not enough valid columns for outlier detection")
            return df
        
        # Method 1: Statistical outliers (IQR method)
        for col in valid_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            df[f'{col}_statistical_outlier'] = outliers_mask
        
        # Method 2: Z-score outliers
        for col in valid_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = 3
            outlier_mask = z_scores > threshold
            df[f'{col}_zscore_outlier'] = False
            df.loc[df[col].notna(), f'{col}_zscore_outlier'] = outlier_mask
        
        # Method 3: Isolation Forest
        try:
            # Prepare data for ML outlier detection
            ml_data = df[valid_cols].fillna(df[valid_cols].median())
            
            if len(ml_data) > 100:  # Need sufficient data for Isolation Forest
                outlier_labels = self.outlier_detector.fit_predict(ml_data)
                df['isolation_forest_outlier'] = outlier_labels == -1
                
                logger.info(f"Isolation Forest detected {df['isolation_forest_outlier'].sum()} outliers")
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
            df['isolation_forest_outlier'] = False
        
        # Combined outlier flag
        outlier_cols = [col for col in df.columns if '_outlier' in col]
        df['is_outlier'] = df[outlier_cols].any(axis=1)
        
        logger.info(f"Total outliers detected: {df['is_outlier'].sum()} ({df['is_outlier'].mean():.2%})")
        return df
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using multiple strategies"""
        df = df.copy()
        
        numerical_cols = [
            'temperature', 'ph', 'dissolved_oxygen', 'conductivity', 
            'turbidity', 'permanganate_index', 'ammonia_nitrogen',
            'total_phosphorus', 'total_nitrogen', 'chlorophyll_a', 'algae_density'
        ]
        
        # Remove columns that don't exist
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        # Strategy 1: Forward fill for time series continuity
        for station in df['station_name'].unique():
            station_mask = df['station_name'] == station
            df.loc[station_mask, numerical_cols] = df.loc[station_mask, numerical_cols].fillna(method='ffill')
        
        # Strategy 2: Seasonal median imputation
        df_grouped = df.groupby(['station_name', df['monitoring_time'].dt.month])
        for col in numerical_cols:
            df[col] = df[col].fillna(df_grouped[col].transform('median'))
        
        # Strategy 3: KNN imputation for remaining missing values
        remaining_missing = df[numerical_cols].isna().sum().sum()
        if remaining_missing > 0:
            logger.info(f"Applying KNN imputation for {remaining_missing} remaining missing values")
            
            # Prepare data for KNN (scale first)
            data_scaled = self.robust_scaler.fit_transform(df[numerical_cols].fillna(0))
            data_imputed = self.imputer.fit_transform(data_scaled)
            data_imputed = self.robust_scaler.inverse_transform(data_imputed)
            
            for i, col in enumerate(numerical_cols):
                df[col] = data_imputed[:, i]
        
        logger.info("Missing value imputation completed")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ML models"""
        df = df.copy()
        
        # Temporal features
        df['hour'] = df['monitoring_time'].dt.hour
        df['day_of_week'] = df['monitoring_time'].dt.dayofweek
        df['month'] = df['monitoring_time'].dt.month
        df['season'] = df['monitoring_time'].dt.month.map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,    # Spring
            6: 2, 7: 2, 8: 2,    # Summer
            9: 3, 10: 3, 11: 3   # Autumn
        })
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Pollution indices
        if 'ammonia_nitrogen' in df.columns and 'total_phosphorus' in df.columns:
            df['nutrient_ratio'] = df['ammonia_nitrogen'] / (df['total_phosphorus'] + 1e-6)
        
        if 'total_nitrogen' in df.columns and 'total_phosphorus' in df.columns:
            df['np_ratio'] = df['total_nitrogen'] / (df['total_phosphorus'] + 1e-6)
        
        # Eutrophication indicators
        if all(col in df.columns for col in ['total_phosphorus', 'total_nitrogen', 'chlorophyll_a']):
            df['eutrophication_index'] = (
                df['total_phosphorus'] * 0.4 + 
                df['total_nitrogen'] * 0.3 + 
                df['chlorophyll_a'] * 0.3
            )
        
        # Water quality grade encoding
        grade_mapping = {'Ⅰ': 1, 'Ⅱ': 2, 'Ⅲ': 3, 'Ⅳ': 4, 'Ⅴ': 5, '劣Ⅴ': 6}
        df['quality_grade_numeric'] = df['water_quality_grade'].map(grade_mapping)
        
        # Rolling statistics (for time series analysis)
        for station in df['station_name'].unique():
            station_mask = df['station_name'] == station
            station_data = df[station_mask].sort_values('monitoring_time')
            
            for col in ['ph', 'dissolved_oxygen', 'ammonia_nitrogen', 'total_phosphorus']:
                if col in station_data.columns:
                    # 24-hour rolling mean
                    df.loc[station_mask, f'{col}_24h_mean'] = (
                        station_data[col].rolling(window=6, min_periods=1).mean()
                    )
                    # 24-hour rolling std
                    df.loc[station_mask, f'{col}_24h_std'] = (
                        station_data[col].rolling(window=6, min_periods=1).std()
                    )
        
        # Spatial lag features (for stations in same watershed)
        for watershed in df['watershed'].unique():
            watershed_data = df[df['watershed'] == watershed]
            for col in ['ph', 'dissolved_oxygen', 'ammonia_nitrogen']:
                if col in watershed_data.columns:
                    watershed_mean = watershed_data.groupby('monitoring_time')[col].mean()
                    df.loc[df['watershed'] == watershed, f'{col}_watershed_mean'] = (
                        df[df['watershed'] == watershed]['monitoring_time'].map(watershed_mean)
                    )
        
        logger.info(f"Created {len(df.columns)} features (including {len(df.columns) - len(df.columns)} engineered)")
        return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous patterns in water quality data"""
        df = df.copy()
        
        # Peak detection for sudden changes
        for station in df['station_name'].unique():
            station_mask = df['station_name'] == station
            station_data = df[station_mask].sort_values('monitoring_time')
            
            for col in ['ammonia_nitrogen', 'total_phosphorus', 'turbidity']:
                if col in station_data.columns:
                    values = station_data[col].dropna()
                    if len(values) > 10:
                        # Detect peaks (sudden increases)
                        peaks, _ = find_peaks(values, height=np.percentile(values, 90))
                        df.loc[station_mask, f'{col}_peak'] = False
                        peak_indices = station_data.index[peaks]
                        df.loc[peak_indices, f'{col}_peak'] = True
        
        # Detect unusual combinations (e.g., high DO with high nutrients)
        if all(col in df.columns for col in ['dissolved_oxygen', 'ammonia_nitrogen', 'total_phosphorus']):
            # Unusual: high DO with high nutrients (possible measurement error)
            df['unusual_do_nutrient'] = (
                (df['dissolved_oxygen'] > df['dissolved_oxygen'].quantile(0.9)) &
                ((df['ammonia_nitrogen'] > df['ammonia_nitrogen'].quantile(0.8)) |
                 (df['total_phosphorus'] > df['total_phosphorus'].quantile(0.8)))
            )
        
        return df
    
    def process_pipeline(self, db_url: str, days_back: int = 30) -> pd.DataFrame:
        """Run complete ETL pipeline"""
        logger.info("Starting ETL pipeline...")
        
        # Load data
        df = self.load_data(db_url, days_back)
        
        # Detect outliers
        df = self.detect_outliers(df)
        
        # Remove extreme outliers but keep moderate ones for analysis
        extreme_outlier_mask = df['isolation_forest_outlier'] | (
            df[[col for col in df.columns if '_statistical_outlier' in col]].sum(axis=1) > 3
        )
        df = df[~extreme_outlier_mask]
        logger.info(f"Removed {extreme_outlier_mask.sum()} extreme outliers")
        
        # Impute missing values
        df = self.impute_missing_values(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Detect anomalies
        df = self.detect_anomalies(df)
        
        logger.info(f"ETL pipeline completed. Final dataset: {len(df)} records, {len(df.columns)} features")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to file"""
        df.to_parquet(output_path, compression='snappy')
        logger.info(f"Processed data saved to {output_path}")

def main():
    """Run ETL pipeline"""
    DB_URL = "postgresql://user:password@localhost:5432/pollution_db"
    
    etl = WaterQualityETL(contamination=0.1)
    processed_df = etl.process_pipeline(DB_URL, days_back=90)
    
    # Save processed data
    output_path = "data/processed_water_quality.parquet"
    etl.save_processed_data(processed_df, output_path)
    
    print(f"ETL pipeline completed. Processed {len(processed_df)} records.")

if __name__ == "__main__":
    main()

