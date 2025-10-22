"""
Prophet-based Time Series Forecasting for Water Quality Prediction

Implements Facebook Prophet for decomposing seasonality, trend, and holiday effects
in water quality time series data.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import List, Dict, Tuple, Optional
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityProphet:
    """Prophet model for water quality time series forecasting"""
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0):
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        
    def prepare_prophet_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare data for Prophet (requires 'ds' and 'y' columns)"""
        
        # Group by station and prepare time series
        prophet_data = []
        
        for station in df['station_name'].unique():
            station_data = df[df['station_name'] == station].copy()
            station_data = station_data.sort_values('monitoring_time')
            
            # Remove missing values
            station_data = station_data.dropna(subset=[target_column])
            
            if len(station_data) < 50:  # Need sufficient data for Prophet
                continue
            
            # Create Prophet format
            prophet_df = pd.DataFrame({
                'ds': station_data['monitoring_time'],
                'y': station_data[target_column],
                'station': station
            })
            
            prophet_data.append(prophet_df)
        
        if prophet_data:
            return pd.concat(prophet_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def train_prophet_model(self, df: pd.DataFrame, target_column: str, station: str) -> Prophet:
        """Train Prophet model for a specific station and target"""
        
        # Filter data for specific station
        station_data = df[df['station_name'] == station].copy()
        station_data = station_data.sort_values('monitoring_time')
        station_data = station_data.dropna(subset=[target_column])
        
        if len(station_data) < 50:
            raise ValueError(f"Insufficient data for station {station}")
        
        # Prepare Prophet data
        prophet_df = pd.DataFrame({
            'ds': station_data['monitoring_time'],
            'y': station_data[target_column]
        })
        
        # Create and configure Prophet model
        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=0.95
        )
        
        # Add custom seasonalities for water quality patterns
        # Add monthly seasonality for seasonal variations
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Add quarterly seasonality for broader patterns
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        
        # Add regressors for other water quality parameters
        regressor_columns = ['temperature', 'ph', 'dissolved_oxygen', 'conductivity', 'turbidity']
        available_regressors = [col for col in regressor_columns if col in station_data.columns and col != target_column]
        
        for regressor in available_regressors:
            if station_data[regressor].notna().sum() > len(station_data) * 0.8:  # At least 80% non-null
                # Forward fill missing values
                station_data[regressor] = station_data[regressor].fillna(method='ffill').fillna(method='bfill')
                prophet_df[regressor] = station_data[regressor]
                model.add_regressor(regressor)
        
        # Train model
        model.fit(prophet_df)
        
        return model
    
    def train_multi_station(self, df: pd.DataFrame, target_columns: List[str]):
        """Train Prophet models for multiple stations and targets"""
        
        for target in target_columns:
            logger.info(f"Training Prophet models for {target}")
            
            self.models[target] = {}
            available_stations = df['station_name'].unique()
            
            for station in available_stations:
                try:
                    model = self.train_prophet_model(df, target, station)
                    self.models[target][station] = model
                    logger.info(f"Trained model for {station} - {target}")
                except Exception as e:
                    logger.warning(f"Failed to train model for {station} - {target}: {e}")
    
    def forecast(self, df: pd.DataFrame, target_columns: List[str], periods: int = 24, freq: str = '4H') -> Dict[str, Dict[str, pd.DataFrame]]:
        """Make forecasts for specified periods"""
        
        forecasts = {}
        
        for target in target_columns:
            if target not in self.models:
                continue
                
            forecasts[target] = {}
            
            for station, model in self.models[target].items():
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq=freq)
                
                # Add regressors if available
                station_data = df[df['station_name'] == station].sort_values('monitoring_time')
                
                # Get last known values for regressors
                regressor_columns = ['temperature', 'ph', 'dissolved_oxygen', 'conductivity', 'turbidity']
                available_regressors = [col for col in regressor_columns if col in station_data.columns and col != target]
                
                for regressor in available_regressors:
                    if regressor in future.columns:
                        # Use last known value for future periods
                        last_value = station_data[regressor].dropna().iloc[-1]
                        future[regressor].fillna(last_value, inplace=True)
                
                # Make forecast
                forecast = model.predict(future)
                forecasts[target][station] = forecast
        
        self.forecasts = forecasts
        return forecasts
    
    def evaluate_prophet_model(self, df: pd.DataFrame, target_column: str, station: str, test_days: int = 30) -> Dict[str, float]:
        """Evaluate Prophet model performance"""
        
        if target_column not in self.models or station not in self.models[target_column]:
            return {}
        
        model = self.models[target_column][station]
        
        # Split data temporally
        station_data = df[df['station_name'] == station].sort_values('monitoring_time')
        station_data = station_data.dropna(subset=[target_column])
        
        if len(station_data) < test_days * 6:  # Need at least 6 measurements per day
            return {}
        
        split_date = station_data['monitoring_time'].max() - timedelta(days=test_days)
        train_data = station_data[station_data['monitoring_time'] <= split_date]
        test_data = station_data[station_data['monitoring_time'] > split_date]
        
        if len(train_data) < 50 or len(test_data) < 10:
            return {}
        
        # Train model on training data
        train_prophet_df = pd.DataFrame({
            'ds': train_data['monitoring_time'],
            'y': train_data[target_column]
        })
        
        # Add regressors
        regressor_columns = ['temperature', 'ph', 'dissolved_oxygen', 'conductivity', 'turbidity']
        available_regressors = [col for col in regressor_columns if col in train_data.columns and col != target_column]
        
        for regressor in available_regressors:
            if train_data[regressor].notna().sum() > len(train_data) * 0.8:
                train_prophet_df[regressor] = train_data[regressor].fillna(method='ffill').fillna(method='bfill')
        
        # Create new model for evaluation
        eval_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode
        )
        
        for regressor in available_regressors:
            if regressor in train_prophet_df.columns:
                eval_model.add_regressor(regressor)
        
        eval_model.fit(train_prophet_df)
        
        # Make forecast for test period
        future = eval_model.make_future_dataframe(periods=len(test_data), freq='4H')
        
        # Add regressors for future
        for regressor in available_regressors:
            if regressor in future.columns:
                last_value = train_data[regressor].dropna().iloc[-1]
                future[regressor].fillna(last_value, inplace=True)
        
        forecast = eval_model.predict(future)
        
        # Get predictions for test period
        test_forecast = forecast[forecast['ds'] > split_date]
        
        # Calculate metrics
        actual = test_data[target_column].values
        predicted = test_forecast['yhat'].values[:len(actual)]
        
        if len(actual) == len(predicted) and len(actual) > 0:
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'n_predictions': len(actual)
            }
        
        return {}
    
    def evaluate_all_models(self, df: pd.DataFrame, target_columns: List[str], test_days: int = 30) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate all trained models"""
        
        evaluation_results = {}
        
        for target in target_columns:
            if target not in self.models:
                continue
                
            evaluation_results[target] = {}
            
            for station in self.models[target].keys():
                metrics = self.evaluate_prophet_model(df, target, station, test_days)
                if metrics:
                    evaluation_results[target][station] = metrics
        
        self.metrics = evaluation_results
        return evaluation_results
    
    def plot_forecast(self, target_column: str, station: str, periods: int = 168, figsize: Tuple[int, int] = (15, 10)):
        """Plot Prophet forecast results"""
        
        if target_column not in self.models or station not in self.models[target_column]:
            print(f"No model found for {station} - {target_column}")
            return
        
        model = self.models[target_column][station]
        
        # Make forecast
        future = model.make_future_dataframe(periods=periods, freq='4H')
        forecast = model.predict(future)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Forecast for {station} - {target_column}', 'Components'),
            vertical_spacing=0.1
        )
        
        # Plot forecast
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(0,0,255,0.2)'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Lower Bound',
                line=dict(color='rgba(0,0,255,0.2)'),
                fill='tonexty',
                fillcolor='rgba(0,0,255,0.2)'
            ),
            row=1, col=1
        )
        
        # Plot trend component
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['trend'],
                mode='lines',
                name='Trend',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Prophet Forecast: {station} - {target_column}',
            xaxis_title='Date',
            yaxis_title=target_column,
            height=figsize[1],
            width=figsize[0]
        )
        
        fig.show()
    
    def get_insights(self, target_column: str, station: str) -> Dict[str, any]:
        """Extract insights from Prophet model components"""
        
        if target_column not in self.models or station not in self.models[target_column]:
            return {}
        
        model = self.models[target_column][station]
        
        # Get model parameters
        insights = {
            'changepoints': model.changepoints.tolist(),
            'seasonality_modes': model.seasonality_mode,
            'regressors': list(model.extra_regressors.keys()) if hasattr(model, 'extra_regressors') else []
        }
        
        # Get forecast for analysis
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        # Analyze trend
        trend_data = forecast['trend']
        insights['trend_direction'] = 'increasing' if trend_data.iloc[-1] > trend_data.iloc[0] else 'decreasing'
        insights['trend_magnitude'] = abs(trend_data.iloc[-1] - trend_data.iloc[0])
        
        # Analyze seasonality
        if 'yearly' in forecast.columns:
            yearly_seasonality = forecast['yearly'].std()
            insights['yearly_seasonality_strength'] = yearly_seasonality
        
        if 'weekly' in forecast.columns:
            weekly_seasonality = forecast['weekly'].std()
            insights['weekly_seasonality_strength'] = weekly_seasonality
        
        return insights

def main():
    """Example usage"""
    # Load processed data
    df = pd.read_parquet("data/processed_water_quality.parquet")
    
    # Define target variables
    targets = ['ph', 'dissolved_oxygen', 'ammonia_nitrogen', 'total_phosphorus']
    
    # Initialize Prophet model
    prophet_model = WaterQualityProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    # Train models
    prophet_model.train_multi_station(df, targets)
    
    # Make forecasts
    forecasts = prophet_model.forecast(df, targets, periods=168, freq='4H')
    
    # Evaluate models
    metrics = prophet_model.evaluate_all_models(df, targets, test_days=30)
    
    print("Prophet model training and evaluation completed")
    print(f"Average RMSE across all models: {np.mean([np.mean([m['rmse'] for m in station_metrics.values()]) for station_metrics in metrics.values()])}")

if __name__ == "__main__":
    main()

