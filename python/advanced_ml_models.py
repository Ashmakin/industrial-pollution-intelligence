#!/usr/bin/env python3
"""
高级机器学习预测系统
- SARIMAX模型
- PyTorch深度学习模型
- TensorFlow模型
- 集成学习
- 时间序列特征工程
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import sys
warnings.filterwarnings('ignore')

# 重定向警告到stderr，确保stdout只有JSON输出
import logging
logging.getLogger().setLevel(logging.ERROR)

# 传统机器学习
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 时间序列分析
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """PyTorch时间序列数据集"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class AdvancedLSTM(nn.Module):
    """高级LSTM模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(AdvancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out.transpose(0, 1), 
                                   lstm_out.transpose(0, 1), 
                                   lstm_out.transpose(0, 1))
        
        # 取最后一个时间步的输出
        output = attn_out[-1]
        
        # 全连接层
        output = self.fc(output)
        return output

class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x[-1]  # 取最后一个时间步
        x = self.output_projection(x)
        return x

class AdvancedMLPredictor:
    """高级机器学习预测器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.scalers = {}
        self.models = {}
        self.parameter_ranges = {
            'ph': (0, 14),
            'dissolved_oxygen': (0, 20),
            'ammonia_nitrogen': (0, 10),
            'total_phosphorus': (0, 5),
            'cod': (0, 100),
            'bod5': (0, 50),
            'temperature': (-10, 40)
        }
    
    def load_data(self, station_name: str, parameter: str, days: int = 365) -> pd.DataFrame:
        """加载时间序列数据"""
        try:
            conn = psycopg2.connect(self.db_url)
            # 构建动态查询
            query = f"""
                SELECT monitoring_time, {parameter} as value
                FROM water_quality_data 
                WHERE station_name = %s 
                AND monitoring_time >= %s
                AND {parameter} IS NOT NULL
                ORDER BY monitoring_time
            """
            
            start_date = datetime.now() - timedelta(days=days)
            df = pd.read_sql_query(query, conn, params=[station_name, start_date])
            conn.close()
            
            if df.empty:
                logger.warning(f"No data found for {station_name} - {parameter}")
                return pd.DataFrame()
            
            df['monitoring_time'] = pd.to_datetime(df['monitoring_time'])
            df = df.set_index('monitoring_time')
            df = df.resample('H').mean().interpolate()  # 小时级插值
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间序列特征"""
        df_features = df.copy()
        
        # 时间特征
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_year'] = df_features.index.dayofyear
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        
        # 周期性特征
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        # 滞后特征
        for lag in [1, 2, 3, 6, 12, 24]:
            df_features[f'value_lag_{lag}'] = df_features['value'].shift(lag)
        
        # 滚动统计特征
        for window in [6, 12, 24]:
            df_features[f'value_mean_{window}'] = df_features['value'].rolling(window).mean()
            df_features[f'value_std_{window}'] = df_features['value'].rolling(window).std()
            df_features[f'value_min_{window}'] = df_features['value'].rolling(window).min()
            df_features[f'value_max_{window}'] = df_features['value'].rolling(window).max()
        
        # 差分特征
        df_features['value_diff_1'] = df_features['value'].diff(1)
        df_features['value_diff_24'] = df_features['value'].diff(24)
        
        # 趋势特征
        df_features['trend'] = np.arange(len(df_features))
        
        return df_features.dropna()
    
    def sarimax_forecast(self, df: pd.DataFrame, parameter: str, forecast_hours: int = 24) -> Dict:
        """SARIMAX模型预测"""
        try:
            # 数据预处理
            ts_data = df['value'].dropna()
            
            # 季节性分解
            decomposition = seasonal_decompose(ts_data, model='additive', period=24)
            
            # 自动选择SARIMAX参数
            best_aic = float('inf')
            best_params = None
            
            # 网格搜索最佳参数
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)
            P_values = range(0, 2)
            D_values = range(0, 2)
            Q_values = range(0, 2)
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        for P in P_values:
                            for D in D_values:
                                for Q in Q_values:
                                    try:
                                        model = SARIMAX(ts_data,
                                                       order=(p, d, q),
                                                       seasonal_order=(P, D, Q, 24),
                                                       enforce_stationarity=False,
                                                       enforce_invertibility=False)
                                        fitted_model = model.fit(disp=False)
                                        
                                        if fitted_model.aic < best_aic:
                                            best_aic = fitted_model.aic
                                            best_params = (p, d, q, P, D, Q)
                                    except:
                                        continue
            
            if best_params is None:
                best_params = (1, 1, 1, 1, 1, 1)  # 默认参数
            
            # 训练最终模型
            final_model = SARIMAX(ts_data,
                                 order=best_params[:3],
                                 seasonal_order=(*best_params[3:], 24),
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
            fitted_model = final_model.fit(disp=False)
            
            # 预测
            forecast = fitted_model.get_forecast(steps=forecast_hours)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            
            # 生成预测结果
            predictions = []
            current_time = datetime.now()
            
            for i in range(forecast_hours):
                pred_time = current_time + timedelta(hours=i)
                pred_value = forecast_mean.iloc[i]
                lower_bound = forecast_ci.iloc[i, 0]
                upper_bound = forecast_ci.iloc[i, 1]
                
                # 确保预测值在合理范围内
                if parameter in self.parameter_ranges:
                    min_val, max_val = self.parameter_ranges[parameter]
                    pred_value = np.clip(pred_value, min_val, max_val)
                    lower_bound = np.clip(lower_bound, min_val, max_val)
                    upper_bound = np.clip(upper_bound, min_val, max_val)
                
                predictions.append({
                    'timestamp': pred_time.isoformat(),
                    'predicted_value': float(pred_value),
                    'confidence_lower': float(lower_bound),
                    'confidence_upper': float(upper_bound)
                })
            
            # 转换为与Rust后端兼容的格式
            return {
                'station_name': 'Unknown Station',  # 将在调用时设置
                'parameter': parameter,
                'predictions': predictions,
                'model_metrics': {
                    'rmse': float(np.sqrt(fitted_model.mse_resid)) if hasattr(fitted_model, 'mse_resid') else 0.0,
                    'mae': 0.0,  # SARIMAX不直接提供MAE
                    'mape': 0.0   # SARIMAX不直接提供MAPE
                },
                'data_points_used': len(ts_data),
                'forecast_model': 'SARIMAX',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SARIMAX forecast error: {e}")
            return {'error': str(e)}
    
    def pytorch_forecast(self, df: pd.DataFrame, parameter: str, forecast_hours: int = 24) -> Dict:
        """PyTorch深度学习预测"""
        if not PYTORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        try:
            # 准备数据
            df_features = self.create_features(df)
            
            # 选择特征
            feature_columns = [col for col in df_features.columns if col != 'value']
            X = df_features[feature_columns].values
            y = df_features['value'].values
            
            # 标准化
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # 创建序列数据
            sequence_length = 24
            X_seq, y_seq = [], []
            
            for i in range(sequence_length, len(X_scaled)):
                X_seq.append(X_scaled[i-sequence_length:i])
                y_seq.append(y_scaled[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # 分割数据
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # 创建数据集
            train_dataset = TimeSeriesDataset(X_train, y_train)
            test_dataset = TimeSeriesDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # 创建模型
            model = AdvancedLSTM(input_size=X_seq.shape[2])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 训练模型
            model.train()
            for epoch in range(50):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 10 == 0:
                    logger.info(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
            
            # 预测
            model.eval()
            predictions = []
            current_sequence = X_seq[-1:]  # 使用最后一个序列
            
            with torch.no_grad():
                for i in range(forecast_hours):
                    pred = model(torch.FloatTensor(current_sequence))
                    pred_value = scaler_y.inverse_transform(pred.numpy())[0, 0]
                    
                    # 确保预测值在合理范围内
                    if parameter in self.parameter_ranges:
                        min_val, max_val = self.parameter_ranges[parameter]
                        pred_value = np.clip(pred_value, min_val, max_val)
                    
                    pred_time = datetime.now() + timedelta(hours=i)
                    predictions.append({
                        'timestamp': pred_time.isoformat(),
                        'predicted_value': float(pred_value),
                        'confidence_lower': float(pred_value * 0.9),
                        'confidence_upper': float(pred_value * 1.1)
                    })
                    
                    # 更新序列（简化处理）
                    new_point = np.zeros((1, sequence_length, X_seq.shape[2]))
                    new_point[0, :-1] = current_sequence[0, 1:]
                    new_point[0, -1] = X_scaled[-1]
                    current_sequence = new_point
            
            return {
                'model_type': 'PyTorch_LSTM',
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"PyTorch forecast error: {e}")
            return {'error': str(e)}
    
    def tensorflow_forecast(self, df: pd.DataFrame, parameter: str, forecast_hours: int = 24) -> Dict:
        """TensorFlow深度学习预测"""
        if not TENSORFLOW_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        try:
            # 准备数据
            df_features = self.create_features(df)
            
            # 选择特征
            feature_columns = [col for col in df_features.columns if col != 'value']
            X = df_features[feature_columns].values
            y = df_features['value'].values
            
            # 标准化
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
            
            # 创建序列数据
            sequence_length = 24
            X_seq, y_seq = [], []
            
            for i in range(sequence_length, len(X_scaled)):
                X_seq.append(X_scaled[i-sequence_length:i])
                y_seq.append(y_scaled[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # 分割数据
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # 创建模型
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, X_seq.shape[2])),
                MaxPooling1D(pool_size=2),
                LSTM(128, return_sequences=True, dropout=0.2),
                LSTM(64, return_sequences=False, dropout=0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # 回调函数
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # 预测
            predictions = []
            current_sequence = X_seq[-1:]  # 使用最后一个序列
            
            for i in range(forecast_hours):
                pred = model.predict(current_sequence, verbose=0)
                pred_value = scaler_y.inverse_transform(pred)[0, 0]
                
                # 确保预测值在合理范围内
                if parameter in self.parameter_ranges:
                    min_val, max_val = self.parameter_ranges[parameter]
                    pred_value = np.clip(pred_value, min_val, max_val)
                
                pred_time = datetime.now() + timedelta(hours=i)
                predictions.append({
                    'timestamp': pred_time.isoformat(),
                    'predicted_value': float(pred_value),
                    'confidence_lower': float(pred_value * 0.9),
                    'confidence_upper': float(pred_value * 1.1)
                })
                
                # 更新序列
                new_point = np.zeros((1, sequence_length, X_seq.shape[2]))
                new_point[0, :-1] = current_sequence[0, 1:]
                new_point[0, -1] = X_scaled[-1]
                current_sequence = new_point
            
            return {
                'model_type': 'TensorFlow_LSTM_CNN',
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"TensorFlow forecast error: {e}")
            return {'error': str(e)}
    
    def ensemble_forecast(self, df: pd.DataFrame, parameter: str, forecast_hours: int = 24) -> Dict:
        """集成预测"""
        try:
            # 获取不同模型的预测
            sarimax_result = self.sarimax_forecast(df, parameter, forecast_hours)
            pytorch_result = self.pytorch_forecast(df, parameter, forecast_hours)
            tensorflow_result = self.tensorflow_forecast(df, parameter, forecast_hours)
            
            # 收集有效预测
            valid_predictions = []
            if 'predictions' in sarimax_result:
                valid_predictions.append(sarimax_result['predictions'])
            if 'predictions' in pytorch_result:
                valid_predictions.append(pytorch_result['predictions'])
            if 'predictions' in tensorflow_result:
                valid_predictions.append(tensorflow_result['predictions'])
            
            if not valid_predictions:
                return {'error': 'No valid predictions available'}
            
            # 集成预测结果
            ensemble_predictions = []
            for i in range(forecast_hours):
                pred_values = []
                lower_bounds = []
                upper_bounds = []
                
                for pred_list in valid_predictions:
                    if i < len(pred_list):
                        pred_values.append(pred_list[i]['predicted_value'])
                        lower_bounds.append(pred_list[i]['confidence_lower'])
                        upper_bounds.append(pred_list[i]['confidence_upper'])
                
                if pred_values:
                    ensemble_value = np.mean(pred_values)
                    ensemble_lower = np.mean(lower_bounds)
                    ensemble_upper = np.mean(upper_bounds)
                    
                    ensemble_predictions.append({
                        'timestamp': datetime.now().isoformat(),
                        'predicted_value': float(ensemble_value),
                        'confidence_lower': float(ensemble_lower),
                        'confidence_upper': float(ensemble_upper)
                    })
            
            return {
                'model_type': 'Ensemble',
                'ensemble_models': len(valid_predictions),
                'predictions': ensemble_predictions
            }
            
        except Exception as e:
            logger.error(f"Ensemble forecast error: {e}")
            return {'error': str(e)}
    
    def predict(self, station_name: str, parameter: str, model_type: str = 'ensemble', forecast_hours: int = 24) -> Dict:
        """主预测函数"""
        try:
            # 加载数据
            df = self.load_data(station_name, parameter)
            if df.empty:
                return {'error': 'No data available'}
            
            # 选择模型
            result = None
            if model_type == 'sarimax':
                result = self.sarimax_forecast(df, parameter, forecast_hours)
            elif model_type == 'pytorch':
                result = self.pytorch_forecast(df, parameter, forecast_hours)
            elif model_type == 'tensorflow':
                result = self.tensorflow_forecast(df, parameter, forecast_hours)
            else:  # ensemble
                result = self.ensemble_forecast(df, parameter, forecast_hours)
            
            # 设置站点名称
            if isinstance(result, dict) and 'station_name' in result:
                result['station_name'] = station_name
                
            return result
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}

def main():
    """主函数"""
    import sys
    import json
    import argparse
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    parser = argparse.ArgumentParser(description='Advanced ML Models for Pollution Prediction')
    parser.add_argument('--station', required=True, help='Station name')
    parser.add_argument('--parameter', required=True, help='Parameter name')
    parser.add_argument('--model', required=True, help='Model type (sarimax, lstm, etc.)')
    parser.add_argument('--horizon', type=int, default=24, help='Forecast horizon in hours')
    parser.add_argument('--database-url', default=db_url, help='Database URL')
    
    args = parser.parse_args()
    
    predictor = AdvancedMLPredictor(args.database_url)
    result = predictor.predict(args.station, args.parameter, args.model, args.horizon)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
