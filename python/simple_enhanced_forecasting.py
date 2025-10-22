#!/usr/bin/env python3
"""
简化的增强机器学习预测系统
使用PyTorch LSTM进行水质预测
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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 传统机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# PyTorch深度学习
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class SimpleLSTMNet(nn.Module):
    """简化的LSTM神经网络模型"""
    
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        return output

class SimpleEnhancedForecaster:
    """简化的增强预测器"""
    
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
            
            # 参数映射
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
            
            # 时间范围
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
            
            # 重采样到4小时间隔
            df = df.resample('4h').mean()
            df = df.dropna()
            
            logger.info(f"Loaded {len(df)} records for {station} - {parameter}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def create_sequences(self, data: np.ndarray, seq_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列数据"""
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length])
        
        return np.array(sequences), np.array(targets)
    
    def train_lstm_model(self, data: pd.DataFrame, parameter: str, seq_length: int = 12) -> Dict[str, Any]:
        """训练LSTM模型"""
        try:
            values = data['value'].values.reshape(-1, 1)
            
            if len(values) < seq_length + 5:
                logger.warning(f"Insufficient data for LSTM training: {len(values)} records")
                return {"error": "Insufficient data"}
            
            # 数据标准化
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values)
            self.scalers[parameter] = scaler
            
            # 创建序列
            X, y = self.create_sequences(scaled_values.flatten(), seq_length)
            
            if len(X) < 5:
                logger.warning("Insufficient sequences for training")
                return {"error": "Insufficient sequences"}
            
            # 分割数据
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 转换为PyTorch张量
            X_train = torch.FloatTensor(X_train).unsqueeze(-1)
            X_test = torch.FloatTensor(X_test).unsqueeze(-1)
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=min(16, len(X_train)), shuffle=True)
            
            # 创建模型
            model = SimpleLSTMNet(input_size=1, hidden_size=32, num_layers=1, output_size=1)
            model = model.to(self.device)
            
            # 优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # 训练模型
            model.train()
            train_losses = []
            
            for epoch in range(50):  # 减少训练轮数
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
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.6f}")
            
            # 评估模型
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(self.device)
                y_pred = model(X_test).squeeze().cpu().numpy()
                y_test_np = y_test.numpy()
                
                mse = mean_squared_error(y_test_np, y_pred)
                mae = mean_absolute_error(y_test_np, y_pred)
                r2 = r2_score(y_test_np, y_pred)
            
            # 保存模型
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
                "training_losses": train_losses[-5:],  # 最后5个epoch的损失
                "data_points": len(values),
                "sequences": len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {"error": str(e)}
    
    def predict_future(self, station: str, parameter: str, horizon: int = 12, model_type: str = "lstm") -> Dict[str, Any]:
        """预测未来值"""
        try:
            # 加载数据
            data = self.load_data(station, parameter, days=60)
            if data.empty:
                return {"error": "No data available"}
            
            # 获取模型
            model_key = f"{model_type}_{parameter}"
            if model_key not in self.models:
                return {"error": f"Model {model_type} not trained for {parameter}"}
            
            model_info = self.models[model_key]
            model = model_info['model']
            scaler = model_info['scaler']
            seq_length = model_info['seq_length']
            
            # 准备数据
            values = data['value'].values.reshape(-1, 1)
            scaled_values = scaler.transform(values)
            
            # 获取最后的序列
            last_sequence = scaled_values[-seq_length:].flatten()
            last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1)
            last_sequence = last_sequence.to(self.device)
            
            # 预测
            model.eval()
            predictions = []
            
            with torch.no_grad():
                current_sequence = last_sequence
                
                for _ in range(horizon):
                    pred = model(current_sequence)
                    pred_value = pred.cpu().numpy()[0, 0]
                    predictions.append(pred_value)
                    
                    # 更新序列（滑动窗口）
                    # 移除第一个元素，添加新的预测值
                    new_seq = torch.cat([current_sequence[:, 1:, :], pred.unsqueeze(1).unsqueeze(-1)], dim=1)
                    current_sequence = new_seq
            
            # 反标准化
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions).flatten()
            
            # 生成时间戳
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
    
    def run_enhanced_forecast(self, station: str, parameter: str, horizon: int = 12) -> Dict[str, Any]:
        """运行增强预测"""
        try:
            # 加载数据
            data = self.load_data(station, parameter, days=60)
            if data.empty:
                return {"error": "No data available"}
            
            # 训练模型
            train_result = self.train_lstm_model(data, parameter)
            if "error" in train_result:
                return train_result
            
            # 进行预测
            prediction_result = self.predict_future(station, parameter, horizon, "lstm")
            if "error" in prediction_result:
                return prediction_result
            
            # 合并结果
            result = {
                "training": train_result,
                "prediction": prediction_result,
                "summary": {
                    "station": station,
                    "parameter": parameter,
                    "horizon": horizon,
                    "model_performance": {
                        "mse": train_result.get("mse", 0),
                        "mae": train_result.get("mae", 0),
                        "r2": train_result.get("r2", 0)
                    }
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced forecast: {e}")
            return {"error": str(e)}

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Enhanced ML Forecasting')
    parser.add_argument('--station', required=True, help='Station name')
    parser.add_argument('--parameter', required=True, help='Parameter name')
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon')
    parser.add_argument('--days', type=int, default=60, help='Days of historical data')
    
    args = parser.parse_args()
    
    # 数据库连接
    db_url = "postgres://pollution_user:pollution_pass@localhost:5432/pollution_db"
    
    # 创建预测器
    forecaster = SimpleEnhancedForecaster(db_url)
    
    # 运行预测
    result = forecaster.run_enhanced_forecast(args.station, args.parameter, args.horizon)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
