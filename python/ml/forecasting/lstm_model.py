"""
LSTM-based Time Series Forecasting for Water Quality Prediction

Implements deep learning models for multi-step ahead forecasting of water quality parameters.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict, Optional
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityLSTM:
    """LSTM model for water quality time series forecasting"""
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 6):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = None
        self.target_columns = None
        
    def prepare_sequences(self, df: pd.DataFrame, target_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        
        # Select features (exclude target columns and non-numeric)
        exclude_cols = target_columns + ['monitoring_time', 'station_name', 'province', 'watershed', 'area_id', 'water_quality_grade']
        feature_columns = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        
        for station in df['station_name'].unique():
            station_data = df[df['station_name'] == station].sort_values('monitoring_time')
            station_features = scaled_features[df['station_name'] == station]
            station_targets = station_data[target_columns].values
            
            for i in range(len(station_data) - self.sequence_length - self.prediction_horizon + 1):
                X.append(station_features[i:i + self.sequence_length])
                y.append(station_targets[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> Sequential:
        """Build LSTM model architecture"""
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(output_shape)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, target_columns: List[str], validation_split: float = 0.2, epochs: int = 100):
        """Train LSTM model"""
        
        logger.info(f"Training LSTM model for targets: {target_columns}")
        
        # Prepare data
        X, y = self.prepare_sequences(df, target_columns)
        
        if len(X) == 0:
            raise ValueError("No valid sequences found. Check data length and sequence_length.")
        
        self.feature_columns = [col for col in df.columns if col not in target_columns + ['monitoring_time', 'station_name', 'province', 'watershed', 'area_id', 'water_quality_grade'] and df[col].dtype in ['float64', 'int64']]
        self.target_columns = target_columns
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1] * y.shape[2]  # prediction_horizon * n_targets
        self.model = self.build_model(input_shape, output_shape)
        
        # Reshape y for training
        y_reshaped = y.reshape(y.shape[0], -1)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = self.model.fit(
            X, y_reshaped,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM training completed")
        return history
    
    def predict(self, df: pd.DataFrame, steps_ahead: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Make predictions using trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if steps_ahead is None:
            steps_ahead = self.prediction_horizon
        
        predictions = {}
        
        for station in df['station_name'].unique():
            station_data = df[df['station_name'] == station].sort_values('monitoring_time')
            
            if len(station_data) < self.sequence_length:
                continue
            
            # Get last sequence
            station_features = self.scaler.transform(station_data[self.feature_columns].iloc[-self.sequence_length:])
            X = station_features.reshape(1, self.sequence_length, -1)
            
            # Predict
            pred = self.model.predict(X, verbose=0)
            pred = pred.reshape(steps_ahead, len(self.target_columns))
            
            predictions[station] = pred
        
        return predictions
    
    def evaluate(self, df: pd.DataFrame, test_split: float = 0.2) -> Dict[str, float]:
        """Evaluate model performance"""
        
        # Split data temporally
        df_sorted = df.sort_values('monitoring_time')
        split_idx = int(len(df_sorted) * (1 - test_split))
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        # Train on train data
        self.train(train_df, self.target_columns, validation_split=0.2, epochs=50)
        
        # Predict on test data
        predictions = self.predict(test_df)
        
        # Calculate metrics
        metrics = {}
        for target in self.target_columns:
            actual_values = []
            predicted_values = []
            
            for station, pred in predictions.items():
                station_data = test_df[test_df['station_name'] == station].sort_values('monitoring_time')
                if len(station_data) >= self.prediction_horizon:
                    actual = station_data[target].iloc[:self.prediction_horizon].values
                    predicted = pred[:, self.target_columns.index(target)]
                    
                    actual_values.extend(actual)
                    predicted_values.extend(predicted)
            
            if actual_values and predicted_values:
                actual_values = np.array(actual_values)
                predicted_values = np.array(predicted_values)
                
                metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(actual_values, predicted_values))
                metrics[f'{target}_mae'] = mean_absolute_error(actual_values, predicted_values)
                metrics[f'{target}_mape'] = np.mean(np.abs((actual_values - predicted_values) / (actual_values + 1e-6))) * 100
        
        return metrics
    
    def save_model(self, model_path: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(f"{model_path}/lstm_model.h5")
        
        # Save scaler and metadata
        joblib.dump(self.scaler, f"{model_path}/scaler.pkl")
        joblib.dump({
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }, f"{model_path}/model_metadata.pkl")
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and scaler"""
        # Load model
        self.model = tf.keras.models.load_model(f"{model_path}/lstm_model.h5")
        
        # Load scaler and metadata
        self.scaler = joblib.load(f"{model_path}/scaler.pkl")
        metadata = joblib.load(f"{model_path}/model_metadata.pkl")
        
        self.feature_columns = metadata['feature_columns']
        self.target_columns = metadata['target_columns']
        self.sequence_length = metadata['sequence_length']
        self.prediction_horizon = metadata['prediction_horizon']
        
        logger.info(f"Model loaded from {model_path}")

class MultiTargetLSTM:
    """Multi-target LSTM for simultaneous prediction of multiple water quality parameters"""
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 6):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.scalers = {}
        
    def train_multi_target(self, df: pd.DataFrame, target_columns: List[str]):
        """Train separate LSTM models for each target"""
        
        for target in target_columns:
            logger.info(f"Training model for {target}")
            
            model = WaterQualityLSTM(self.sequence_length, self.prediction_horizon)
            model.train(df, [target])
            
            self.models[target] = model
            self.scalers[target] = model.scaler
    
    def predict_multi_target(self, df: pd.DataFrame, target_columns: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Predict multiple targets simultaneously"""
        
        predictions = {}
        for target in target_columns:
            if target in self.models:
                predictions[target] = self.models[target].predict(df)
        
        return predictions
    
    def ensemble_predict(self, df: pd.DataFrame, target_columns: List[str], weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Ensemble predictions from multiple models"""
        
        if weights is None:
            weights = {target: 1.0 / len(target_columns) for target in target_columns}
        
        ensemble_predictions = {}
        
        for station in df['station_name'].unique():
            station_predictions = []
            
            for target in target_columns:
                if target in self.models:
                    pred = self.models[target].predict(df)
                    if station in pred:
                        station_predictions.append(pred[station] * weights.get(target, 1.0))
            
            if station_predictions:
                ensemble_predictions[station] = np.mean(station_predictions, axis=0)
        
        return ensemble_predictions

def main():
    """Example usage"""
    # Load processed data
    df = pd.read_parquet("data/processed_water_quality.parquet")
    
    # Define target variables
    targets = ['ph', 'dissolved_oxygen', 'ammonia_nitrogen', 'total_phosphorus']
    
    # Train multi-target LSTM
    multi_lstm = MultiTargetLSTM(sequence_length=24, prediction_horizon=6)
    multi_lstm.train_multi_target(df, targets)
    
    # Make predictions
    predictions = multi_lstm.predict_multi_target(df, targets)
    
    print("Multi-target LSTM training and prediction completed")

if __name__ == "__main__":
    main()

