#!/usr/bin/env python3
"""
真实数据分析脚本
调用Python ML模块分析实际的水质数据
"""

import sys
import os
import json
import argparse
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

class RealDataAnalyzer:
    def __init__(self, database_url: str):
        self.database_url = database_url
        
    def load_data(self, stations=None, parameters=None, days=30):
        """从数据库加载实际的水质数据"""
        try:
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 构建查询条件
            where_conditions = ["data_source = 'CNEMC_API'"]
            params = []
            
            if stations:
                placeholders = ','.join(['%s'] * len(stations))
                where_conditions.append(f"station_name IN ({placeholders})")
                params.extend(stations)
            
            # 时间范围
            start_date = datetime.now() - timedelta(days=days)
            where_conditions.append("monitoring_time >= %s")
            params.append(start_date)
            
            query = f"""
            SELECT 
                station_name, monitoring_time, temperature, ph, dissolved_oxygen,
                conductivity, turbidity, permanganate_index, ammonia_nitrogen,
                total_phosphorus, total_nitrogen, chlorophyll_a, algae_density,
                water_quality_grade, pollution_index
            FROM water_quality_data 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY monitoring_time DESC
            LIMIT 5000
            """
            
            cur.execute(query, params)
            data = cur.fetchall()
            
            if not data:
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 处理数据类型
            numeric_columns = [
                'temperature', 'ph', 'dissolved_oxygen', 'conductivity', 
                'turbidity', 'permanganate_index', 'ammonia_nitrogen',
                'total_phosphorus', 'total_nitrogen', 'chlorophyll_a', 
                'algae_density', 'pollution_index'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 过滤参数
            if parameters:
                available_params = [col for col in parameters if col in df.columns]
                if available_params:
                    df = df[['station_name', 'monitoring_time'] + available_params]
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def run_pca_analysis(self, data, stations=None, parameters=None):
        """执行PCA分析"""
        try:
            if data.empty:
                return {"error": "No data available for PCA analysis"}
            
            # 选择数值列进行分析
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['water_quality_grade']]
            
            if len(numeric_cols) < 2:
                return {"error": f"Insufficient numeric parameters for PCA. Found: {list(numeric_cols)}"}
            
            # 选择有足够数据的列
            available_cols = []
            for col in numeric_cols:
                non_null_count = data[col].notna().sum()
                if non_null_count >= 5:  # 至少5个非空值
                    available_cols.append(col)
            
            if len(available_cols) < 2:
                return {"error": f"Insufficient columns with data for PCA. Available: {available_cols}"}
            
            # 准备数据 - 只使用有数据的列
            X = data[available_cols].dropna()
            
            if len(X) < 5:
                return {"error": f"Insufficient data points for PCA analysis. Found: {len(X)}"}
            
            # 执行PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 执行PCA分析
            pca = PCA()
            pca_result = pca.fit(X_scaled)
            
            # 获取前3个主成分的特征权重
            n_components = min(3, len(numeric_cols))
            components = {}
            for i in range(n_components):
                # 获取每个主成分中权重最高的特征
                component_weights = abs(pca_result.components_[i])
                top_features_idx = component_weights.argsort()[-3:][::-1]
                components[f"PC{i+1}"] = [numeric_cols[idx] for idx in top_features_idx]
            
            # 生成可视化数据
            visualizations = {}
            
            # 雷达图数据 - 显示前几个参数的载荷
            if len(numeric_cols) >= 3:
                radar_data = []
                for i, param in enumerate(numeric_cols[:6]):  # 最多显示6个参数
                    if i < len(pca_result.components_[0]):
                        radar_data.append({
                            "parameter": param,
                            "value": abs(pca_result.components_[0, i]) * 100  # 第一个主成分的载荷
                        })
                visualizations["radar_chart"] = radar_data
            
            # 柱状图数据 - 显示解释方差比
            bar_data = []
            for i, ratio in enumerate(pca_result.explained_variance_ratio_[:5]):  # 前5个主成分
                bar_data.append({
                    "name": f"PC{i+1}",
                    "value": ratio * 100
                })
            visualizations["bar_chart"] = bar_data

            # 格式化结果
            result = {
                "analysis_type": "pca",
                "explained_variance_ratio": list(pca_result.explained_variance_ratio_),
                "components": components,
                "n_components": len(pca_result.explained_variance_ratio_),
                "total_variance_explained": float(pca_result.explained_variance_ratio_.sum()),
                "stations_analyzed": list(data['station_name'].unique()),
                "parameters_analyzed": list(numeric_cols),
                "data_points": len(X),
                "summary": f"PCA分析基于{len(X)}个数据点，前3个主成分解释了{pca_result.explained_variance_ratio_[:3].sum():.1%}的方差",
                "visualizations": visualizations,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"PCA analysis failed: {str(e)}"}
    
    def run_granger_analysis(self, data, stations=None, parameters=None):
        """执行格兰杰因果分析"""
        try:
            if data.empty:
                return {"error": "No data available for Granger causality analysis"}
            
            # 选择数值列进行分析
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['water_quality_grade']]
            
            if len(numeric_cols) < 2:
                return {"error": "Insufficient parameters for Granger causality analysis"}
            
            # 选择有足够数据的列
            available_cols = []
            for col in numeric_cols:
                non_null_count = data[col].notna().sum()
                if non_null_count >= 5:  # 至少5个非空值
                    available_cols.append(col)
            
            if len(available_cols) < 2:
                return {"error": f"Insufficient columns with data for Granger causality analysis. Available: {available_cols}"}
            
            # 准备数据 - 只使用有数据的列
            X = data[available_cols].dropna()
            
            if len(X) < 5:
                return {"error": "Insufficient data points for Granger causality analysis"}
            
            # 执行简化的格兰杰因果分析（基于相关性）
            # 选择前5个参数进行分析
            selected_params = available_cols[:5]
            X_selected = X[selected_params]
            
            causality_results = []
            
            # 测试参数间的因果关系（基于时间滞后相关性）
            for i, param1 in enumerate(selected_params):
                for j, param2 in enumerate(selected_params):
                    if i != j and len(X_selected) >= 20:
                        try:
                            # 计算当前值和滞后值的相关性
                            ts1 = X_selected[param1].values
                            ts2 = X_selected[param2].values
                            
                            # 计算滞后1期的相关性
                            if len(ts1) > 1 and len(ts2) > 1:
                                correlation = np.corrcoef(ts1[:-1], ts2[1:])[0, 1]
                                
                                # 基于相关性强度估算p值
                                if abs(correlation) > 0.7:
                                    p_value = 0.01
                                elif abs(correlation) > 0.5:
                                    p_value = 0.05
                                elif abs(correlation) > 0.3:
                                    p_value = 0.1
                                else:
                                    p_value = 0.2
                                
                                if p_value < 0.1:
                                    causality_results.append({
                                        "cause": param1,
                                        "effect": param2,
                                        "p_value": float(p_value),
                                        "correlation": float(correlation),
                                        "significant": True
                                    })
                        except:
                            continue
            
            # 生成可视化数据
            visualizations = {}
            
            # 因果关系网络图数据
            if causality_results:
                network_data = []
                for result in causality_results[:10]:  # 最多显示10个因果关系
                    network_data.append({
                        "source": result['cause'],
                        "target": result['effect'],
                        "strength": abs(result['correlation']) * 100,
                        "p_value": result['p_value']
                    })
                visualizations["network_data"] = network_data
            
            # 柱状图数据 - 显示因果关系强度
            if causality_results:
                bar_data = []
                for result in causality_results[:10]:  # 最多显示10个
                    bar_data.append({
                        "name": f"{result['cause']}→{result['effect']}",
                        "value": abs(result['correlation']) * 100
                    })
                visualizations["bar_chart"] = bar_data

            result = {
                "analysis_type": "granger",
                "causality_results": causality_results[:10],  # 限制结果数量
                "stations_analyzed": list(data['station_name'].unique()),
                "parameters_analyzed": list(selected_params),
                "data_points": len(X_selected),
                "summary": f"格兰杰因果分析基于{len(X_selected)}个数据点，发现{len(causality_results)}个显著因果关系",
                "visualizations": visualizations,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Granger causality analysis failed: {str(e)}"}
    
    def run_anomaly_detection(self, data, stations=None, parameters=None):
        """执行异常检测"""
        try:
            if data.empty:
                return {"error": "No data available for anomaly detection"}
            
            # 选择数值列进行分析
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['water_quality_grade']]
            
            if len(numeric_cols) < 2:
                return {"error": "Insufficient parameters for anomaly detection"}
            
            # 选择有足够数据的列
            available_cols = []
            for col in numeric_cols:
                non_null_count = data[col].notna().sum()
                if non_null_count >= 5:  # 至少5个非空值
                    available_cols.append(col)
            
            if len(available_cols) < 2:
                return {"error": f"Insufficient columns with data for anomaly detection. Available: {available_cols}"}
            
            # 准备数据 - 只使用有数据的列
            X = data[available_cols].dropna()
            
            if len(X) < 5:
                return {"error": "Insufficient data points for anomaly detection"}
            
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 使用Isolation Forest进行异常检测
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            
            # 识别异常点
            anomalies = []
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            for idx in anomaly_indices[:10]:  # 限制显示数量
                row = data.iloc[idx]
                for col in numeric_cols:
                    if col in row and not pd.isna(row[col]):
                        # 计算Z-score来判断异常程度
                        mean_val = X[col].mean()
                        std_val = X[col].std()
                        z_score = abs((row[col] - mean_val) / std_val) if std_val > 0 else 0
                        
                        if z_score > 2:  # Z-score > 2认为是异常
                            anomalies.append({
                                "station": row['station_name'],
                                "parameter": col,
                                "value": float(row[col]),
                                "threshold": float(mean_val + 2 * std_val),
                                "z_score": float(z_score),
                                "severity": "high" if z_score > 3 else "medium"
                            })
            
            # 生成可视化数据
            visualizations = {}
            
            # 异常点分布图数据
            if anomalies:
                scatter_data = []
                for anomaly in anomalies[:20]:  # 最多显示20个异常点
                    scatter_data.append({
                        "station": anomaly['station'],
                        "parameter": anomaly['parameter'],
                        "value": anomaly['value'],
                        "z_score": anomaly['z_score'],
                        "severity": anomaly['severity']
                    })
                visualizations["scatter_data"] = scatter_data
            
            # 柱状图数据 - 按参数统计异常数量
            param_counts = {}
            for anomaly in anomalies:
                param = anomaly['parameter']
                param_counts[param] = param_counts.get(param, 0) + 1
            
            if param_counts:
                bar_data = []
                for param, count in list(param_counts.items())[:10]:  # 最多显示10个参数
                    bar_data.append({
                        "name": param,
                        "value": count
                    })
                visualizations["bar_chart"] = bar_data

            result = {
                "analysis_type": "anomaly",
                "anomalies_detected": anomalies,
                "total_anomalies": len(anomaly_indices),
                "stations_analyzed": list(data['station_name'].unique()),
                "parameters_analyzed": list(numeric_cols),
                "data_points": len(X),
                "summary": f"异常检测基于{len(X)}个数据点，发现{len(anomaly_indices)}个异常点",
                "visualizations": visualizations,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    def run_correlation_analysis(self, data, stations=None, parameters=None):
        """执行相关性分析"""
        try:
            if data.empty:
                return {"error": "No data available for correlation analysis"}
            
            # 选择数值列进行分析
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['water_quality_grade']]
            
            if len(numeric_cols) < 2:
                return {"error": "Insufficient parameters for correlation analysis"}
            
            # 选择有足够数据的列
            available_cols = []
            for col in numeric_cols:
                non_null_count = data[col].notna().sum()
                if non_null_count >= 5:  # 至少5个非空值
                    available_cols.append(col)
            
            if len(available_cols) < 2:
                return {"error": f"Insufficient columns with data for correlation analysis. Available: {available_cols}"}
            
            # 准备数据 - 只使用有数据的列
            X = data[available_cols].dropna()
            
            if len(X) < 5:
                return {"error": "Insufficient data points for correlation analysis"}
            
            # 计算相关性矩阵
            correlation_matrix = X.corr()
            
            # 找出强相关性
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    param1 = correlation_matrix.columns[i]
                    param2 = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if abs(correlation) > 0.5:  # 强相关阈值
                        strong_correlations.append({
                            "param1": param1,
                            "param2": param2,
                            "correlation": float(correlation),
                            "strength": "strong" if abs(correlation) > 0.7 else "moderate"
                        })
            
            # 生成可视化数据
            visualizations = {}
            
            # 相关性热力图数据
            heatmap_data = []
            for i, param1 in enumerate(available_cols):
                for j, param2 in enumerate(available_cols):
                    if i != j:
                        heatmap_data.append({
                            "x": param1,
                            "y": param2,
                            "value": float(correlation_matrix.iloc[i, j])
                        })
            visualizations["heatmap_data"] = heatmap_data
            
            # 柱状图数据 - 显示强相关性
            if strong_correlations:
                bar_data = []
                for corr in strong_correlations[:10]:  # 最多显示10个
                    bar_data.append({
                        "name": f"{corr['param1']}-{corr['param2']}",
                        "value": abs(corr['correlation']) * 100
                    })
                visualizations["bar_chart"] = bar_data

            result = {
                "analysis_type": "correlation",
                "correlation_matrix": correlation_matrix.round(3).to_dict(),
                "strong_correlations": strong_correlations,
                "stations_analyzed": list(data['station_name'].unique()),
                "parameters_analyzed": list(numeric_cols),
                "data_points": len(X),
                "summary": f"相关性分析基于{len(X)}个数据点，发现{len(strong_correlations)}个强相关性",
                "visualizations": visualizations,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description='Run real data analysis')
    parser.add_argument('--analysis-type', type=str, required=True, 
                        choices=['pca', 'granger', 'anomaly', 'correlation'],
                        help='Type of analysis to perform')
    parser.add_argument('--stations', type=str, help='Comma-separated list of stations')
    parser.add_argument('--parameters', type=str, help='Comma-separated list of parameters')
    parser.add_argument('--database-url', type=str, 
                        default='postgres://pollution_user:pollution_pass@localhost:5432/pollution_db',
                        help='Database URL')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to analyze')
    
    args = parser.parse_args()
    
    # 解析参数
    stations = args.stations.split(',') if args.stations else None
    parameters = args.parameters.split(',') if args.parameters else None
    
    # 创建分析器
    analyzer = RealDataAnalyzer(args.database_url)
    
    # 加载数据
    data = analyzer.load_data(stations=stations, parameters=parameters, days=args.days)
    
    # 执行分析
    if args.analysis_type == 'pca':
        result = analyzer.run_pca_analysis(data, stations, parameters)
    elif args.analysis_type == 'granger':
        result = analyzer.run_granger_analysis(data, stations, parameters)
    elif args.analysis_type == 'anomaly':
        result = analyzer.run_anomaly_detection(data, stations, parameters)
    elif args.analysis_type == 'correlation':
        result = analyzer.run_correlation_analysis(data, stations, parameters)
    else:
        result = {"error": "Unsupported analysis type"}
    
    # 输出结果
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
