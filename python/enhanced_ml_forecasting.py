                      
"""
增强的机器学习预测系统
使用PyTorch深度学习模型进行水质预测
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import sys
import json

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

        
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

             
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

        
torch.manual_seed(42)
np.random.seed(42)

class WaterQualityDataset(Dataset):
    """水质数据数据集"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMNet(nn.Module):
    """LSTM神经网络模型"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
               
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
              
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
                  
        lstm_out, (hidden, cell) = self.lstm(x)
        
                     
        last_output = lstm_out[:, -1, :]
        
              
        output = self.fc(last_output)
        return output

class CNNLSTMNet(nn.Module):
    """CNN-LSTM混合模型"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(CNNLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
               
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
               
        self.lstm = nn.LSTM(64, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
              
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
                                                                                        
        x = x.transpose(1, 2)
        
              
        conv_out = self.conv1d(x)
        conv_out = conv_out.squeeze(-1)            
        
                   
        lstm_input = conv_out.unsqueeze(1)          
        
                  
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
                     
        last_output = lstm_out[:, -1, :]
        
              
        output = self.fc(last_output)
        return output

class TransformerNet(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=3, output_size=1, dropout=0.1):
        super(TransformerNet, self).__init__()
        self.d_model = d_model
        
              
        self.input_projection = nn.Linear(input_size, d_model)
        
              
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
                        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
             
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        
              
        x = self.input_projection(x)
        
                
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
                       
        transformer_out = self.transformer(x)
        
                     
        last_output = transformer_out[:, -1, :]
        
             
        output = self.output_layer(last_output)
        return output

class EnhancedMLForecaster:
    """增强的机器学习预测器"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.scalers = {}                    
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_data(self, station: str, parameter: str, days: int = 30) -> pd.DataFrame:
        """从数据库加载数据"""
        try:
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
                  
            param_mapping = {
                'ph': 'ph',
                'ammonia_nitrogen': 'ammonia_nitrogen',
                'dissolved_oxygen': 'dissolved_oxygen',
                'total_phosphorus': 'total_phosphorus',
                'temperature': 'temperature',
                'conductivity': 'conductivity',
                'turbidity': 'turbidity',
                'permanganate_index': 'permanganate_index',
                'total_nitrogen': 'total_nitrogen',
                'chlorophyll_a': 'chlorophyll_a',
                'algae_density': 'algae_density'
            }
            
            db_param = param_mapping.get(parameter, parameter)
            
                  
            start_date = datetime.now() - timedelta(days=days)
            
            query = f"""
            SELECT monitoring_time, {db_param} as value
            FROM water_quality_data 
            WHERE station_name = %s 
            AND monitoring_time >= %s 
            AND {db_param} IS NOT NULL
            ORDER BY monitoring_time
            """
            
            cur.execute(query, (station, start_date))
            data = cur.fetchall()
            
            if not data:
                logger.warning(f"No data found for station {station}, parameter {parameter}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['monitoring_time'] = pd.to_datetime(df['monitoring_time'])
            df = df.set_index('monitoring_time')
            df = df.sort_index()
            
                       
            df = df.resample('4H').mean()
            df = df.dropna()
            
            logger.info(f"Loaded {len(df)} records for {station} - {parameter}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def create_sequences(self, data: np.ndarray, seq_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列数据"""
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length])
        
        return np.array(sequences), np.array(targets)
    
    def train_lstm_model(self, data: pd.DataFrame, parameter: str, seq_length: int = 24) -> Dict[str, Any]:
        """训练LSTM模型"""
        try:
            values = data['value'].values.reshape(-1, 1)
            
            if len(values) < seq_length + 10:
                logger.warning(f"Insufficient data for LSTM training: {len(values)} records")
                return {"error": "Insufficient data"}
            
                   
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values)
            self.scalers[parameter] = scaler
            
                  
            X, y = self.create_sequences(scaled_values.flatten(), seq_length)
            
            if len(X) < 10:
                logger.warning("Insufficient sequences for training")
                return {"error": "Insufficient sequences"}
            
                  
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
                          
            X_train = torch.FloatTensor(X_train).unsqueeze(-1)
            X_test = torch.FloatTensor(X_test).unsqueeze(-1)
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            
                     
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
                  
            model = LSTMNet(input_size=1, hidden_size=64, num_layers=2, output_size=1)
            model = model.to(self.device)
            
                      
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
                  
            model.train()
            train_losses = []
            
            for epoch in range(100):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.6f}")
            
                  
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(self.device)
                y_pred = model(X_test).squeeze().cpu().numpy()
                y_test_np = y_test.numpy()
                
                mse = mean_squared_error(y_test_np, y_pred)
                mae = mean_absolute_error(y_test_np, y_pred)
                r2 = r2_score(y_test_np, y_pred)
            
                  
            self.models[f'lstm_{parameter}'] = {
                'model': model,
                'scaler': scaler,
                'seq_length': seq_length
            }
            
            return {
                "model_type": "LSTM",
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "training_losses": train_losses[-10:],                 
                "data_points": len(values),
                "sequences": len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {"error": str(e)}
    
    def train_cnn_lstm_model(self, data: pd.DataFrame, parameter: str, seq_length: int = 24) -> Dict[str, Any]:
        """训练CNN-LSTM混合模型"""
        try:
            values = data['value'].values.reshape(-1, 1)
            
            if len(values) < seq_length + 10:
                return {"error": "Insufficient data"}
            
                   
            scaled_values = self.scaler.fit_transform(values)
            
                  
            X, y = self.create_sequences(scaled_values.flatten(), seq_length)
            
            if len(X) < 10:
                return {"error": "Insufficient sequences"}
            
                  
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
                          
            X_train = torch.FloatTensor(X_train).unsqueeze(-1)
            X_test = torch.FloatTensor(X_test).unsqueeze(-1)
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            
                     
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
                  
            model = CNNLSTMNet(input_size=1, hidden_size=64, num_layers=2, output_size=1)
            model = model.to(self.device)
            
                      
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
                  
            model.train()
            train_losses = []
            
            for epoch in range(100):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
            
                  
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(self.device)
                y_pred = model(X_test).squeeze().cpu().numpy()
                y_test_np = y_test.numpy()
                
                mse = mean_squared_error(y_test_np, y_pred)
                mae = mean_absolute_error(y_test_np, y_pred)
                r2 = r2_score(y_test_np, y_pred)
            
                  
            self.models[f'cnn_lstm_{parameter}'] = {
                'model': model,
                'scaler': self.scaler,
                'seq_length': seq_length
            }
            
            return {
                "model_type": "CNN-LSTM",
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "training_losses": train_losses[-10:],
                "data_points": len(values),
                "sequences": len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training CNN-LSTM model: {e}")
            return {"error": str(e)}
    
    def train_transformer_model(self, data: pd.DataFrame, parameter: str, seq_length: int = 24) -> Dict[str, Any]:
        """训练Transformer模型"""
        try:
            values = data['value'].values.reshape(-1, 1)
            
            if len(values) < seq_length + 10:
                return {"error": "Insufficient data"}
            
                   
            scaled_values = self.scaler.fit_transform(values)
            
                  
            X, y = self.create_sequences(scaled_values.flatten(), seq_length)
            
            if len(X) < 10:
                return {"error": "Insufficient sequences"}
            
                  
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
                          
            X_train = torch.FloatTensor(X_train).unsqueeze(-1)
            X_test = torch.FloatTensor(X_test).unsqueeze(-1)
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            
                     
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
                  
            model = TransformerNet(input_size=1, d_model=64, nhead=8, num_layers=3, output_size=1)
            model = model.to(self.device)
            
                      
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
                  
            model.train()
            train_losses = []
            
            for epoch in range(100):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
            
                  
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(self.device)
                y_pred = model(X_test).squeeze().cpu().numpy()
                y_test_np = y_test.numpy()
                
                mse = mean_squared_error(y_test_np, y_pred)
                mae = mean_absolute_error(y_test_np, y_pred)
                r2 = r2_score(y_test_np, y_pred)
            
                  
            self.models[f'transformer_{parameter}'] = {
                'model': model,
                'scaler': self.scaler,
                'seq_length': seq_length
            }
            
            return {
                "model_type": "Transformer",
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "training_losses": train_losses[-10:],
                "data_points": len(values),
                "sequences": len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return {"error": str(e)}
    
    def predict_future(self, station: str, parameter: str, horizon: int = 24, model_type: str = "lstm") -> Dict[str, Any]:
        """预测未来值"""
        try:
                  
            data = self.load_data(station, parameter, days=60)
            if data.empty:
                return {"error": "No data available"}
            
                  
            model_key = f"{model_type}_{parameter}"
            if model_key not in self.models:
                return {"error": f"Model {model_type} not trained for {parameter}"}
            
            model_info = self.models[model_key]
            model = model_info['model']
            scaler = model_info['scaler']
            seq_length = model_info['seq_length']
            
                  
            values = data['value'].values.reshape(-1, 1)
            scaled_values = scaler.transform(values)
            
                     
            last_sequence = scaled_values[-seq_length:].flatten()
            last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1)
            last_sequence = last_sequence.to(self.device)
            
                
            model.eval()
            predictions = []
            
            with torch.no_grad():
                current_sequence = last_sequence
                
                for _ in range(horizon):
                    pred = model(current_sequence)
                    predictions.append(pred.cpu().numpy()[0, 0])
                    
                                
                    pred_reshaped = pred.unsqueeze(1).unsqueeze(-1)
                    new_seq = torch.cat([current_sequence[:, 1:, :], pred_reshaped], dim=1)
                    current_sequence = new_seq
            
                  
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions).flatten()
            
                   
            last_time = data.index[-1]
            future_times = [last_time + timedelta(hours=4*i) for i in range(1, horizon+1)]
            
            return {
                "predictions": predictions.tolist(),
                "timestamps": [t.isoformat() for t in future_times],
                "model_type": model_type,
                "horizon": horizon,
                "station": station,
                "parameter": parameter
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {"error": str(e)}
    
    def ensemble_forecast(self, station: str, parameter: str, horizon: int = 24) -> Dict[str, Any]:
        """集成预测"""
        try:
                    
            lstm_result = self.train_lstm_model(self.load_data(station, parameter), parameter)
            cnn_lstm_result = self.train_cnn_lstm_model(self.load_data(station, parameter), parameter)
            transformer_result = self.train_transformer_model(self.load_data(station, parameter), parameter)
            
                  
            lstm_pred = self.predict_future(station, parameter, horizon, "lstm")
            cnn_lstm_pred = self.predict_future(station, parameter, horizon, "cnn_lstm")
            transformer_pred = self.predict_future(station, parameter, horizon, "transformer")
            
            if "error" in lstm_pred or "error" in cnn_lstm_pred or "error" in transformer_pred:
                return {"error": "Failed to get predictions from all models"}
            
                        
            ensemble_predictions = []
            for i in range(horizon):
                pred = (
                    lstm_pred["predictions"][i] + 
                    cnn_lstm_pred["predictions"][i] + 
                    transformer_pred["predictions"][i]
                ) / 3
                ensemble_predictions.append(pred)
            
            return {
                "predictions": ensemble_predictions,
                "timestamps": lstm_pred["timestamps"],
                "model_type": "Ensemble",
                "horizon": horizon,
                "station": station,
                "parameter": parameter,
                "individual_models": {
                    "lstm": lstm_result,
                    "cnn_lstm": cnn_lstm_result,
                    "transformer": transformer_result
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble forecast: {e}")
            return {"error": str(e)}

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced ML Forecasting')
    parser.add_argument('--station', required=True, help='Station name')
    parser.add_argument('--parameter', required=True, help='Parameter name')
    parser.add_argument('--model-type', choices=['lstm', 'cnn_lstm', 'transformer', 'ensemble'], 
                       default='lstm', help='Model type')
    parser.add_argument('--horizon', type=int, default=24, help='Prediction horizon')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    
    args = parser.parse_args()
    
           
    db_url = "postgres://pollution_user:pollution_pass@localhost:5432/pollution_db"
    
           
    forecaster = EnhancedMLForecaster(db_url)
    
    if args.model_type == 'ensemble':
              
        result = forecaster.ensemble_forecast(args.station, args.parameter, args.horizon)
    else:
               
        data = forecaster.load_data(args.station, args.parameter, args.days)
        if data.empty:
            result = {"error": "No data available"}
        else:
            if args.model_type == 'lstm':
                train_result = forecaster.train_lstm_model(data, args.parameter)
                if "error" not in train_result:
                    result = forecaster.predict_future(args.station, args.parameter, args.horizon, "lstm")
                else:
                    result = train_result
            elif args.model_type == 'cnn_lstm':
                train_result = forecaster.train_cnn_lstm_model(data, args.parameter)
                if "error" not in train_result:
                    result = forecaster.predict_future(args.station, args.parameter, args.horizon, "cnn_lstm")
                else:
                    result = train_result
            elif args.model_type == 'transformer':
                train_result = forecaster.train_transformer_model(data, args.parameter)
                if "error" not in train_result:
                    result = forecaster.predict_future(args.station, args.parameter, args.horizon, "transformer")
                else:
                    result = train_result
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
