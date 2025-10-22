#!/usr/bin/env python3
"""
模拟真实水质数据生成器
- 基于真实的水质参数范围
- 考虑季节性和地理因素
- 包含异常值和趋势
"""

import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import hashlib
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticDataSimulator:
    """真实数据模拟器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
        # 中国各省市信息
        self.provinces = {
            '北京': {'lat': 39.9042, 'lon': 116.4074, 'code': '110000', 'river': '永定河', 'basin': '海河流域'},
            '天津': {'lat': 39.3434, 'lon': 117.3616, 'code': '120000', 'river': '海河', 'basin': '海河流域'},
            '河北': {'lat': 38.0428, 'lon': 114.5149, 'code': '130000', 'river': '滦河', 'basin': '海河流域'},
            '山西': {'lat': 37.8735, 'lon': 112.5624, 'code': '140000', 'river': '汾河', 'basin': '黄河流域'},
            '内蒙古': {'lat': 40.8175, 'lon': 111.7652, 'code': '150000', 'river': '黄河', 'basin': '黄河流域'},
            '辽宁': {'lat': 41.8057, 'lon': 123.4315, 'code': '210000', 'river': '辽河', 'basin': '辽河流域'},
            '吉林': {'lat': 43.8868, 'lon': 125.3245, 'code': '220000', 'river': '松花江', 'basin': '松花江流域'},
            '黑龙江': {'lat': 45.7736, 'lon': 126.6617, 'code': '230000', 'river': '黑龙江', 'basin': '黑龙江流域'},
            '上海': {'lat': 31.2304, 'lon': 121.4737, 'code': '310000', 'river': '黄浦江', 'basin': '长江流域'},
            '江苏': {'lat': 32.0603, 'lon': 118.7969, 'code': '320000', 'river': '长江', 'basin': '长江流域'},
            '浙江': {'lat': 30.2741, 'lon': 120.1551, 'code': '330000', 'river': '钱塘江', 'basin': '长江流域'},
            '安徽': {'lat': 31.8612, 'lon': 117.2849, 'code': '340000', 'river': '淮河', 'basin': '淮河流域'},
            '福建': {'lat': 26.0745, 'lon': 119.2965, 'code': '350000', 'river': '闽江', 'basin': '东南诸河流域'},
            '江西': {'lat': 28.6765, 'lon': 115.8922, 'code': '360000', 'river': '赣江', 'basin': '长江流域'},
            '山东': {'lat': 36.6512, 'lon': 117.1201, 'code': '370000', 'river': '黄河', 'basin': '黄河流域'},
            '河南': {'lat': 34.7578, 'lon': 113.6254, 'code': '410000', 'river': '黄河', 'basin': '黄河流域'},
            '湖北': {'lat': 30.5928, 'lon': 114.3055, 'code': '420000', 'river': '长江', 'basin': '长江流域'},
            '湖南': {'lat': 28.2278, 'lon': 112.9388, 'code': '430000', 'river': '湘江', 'basin': '长江流域'},
            '广东': {'lat': 23.3417, 'lon': 113.4244, 'code': '440000', 'river': '珠江', 'basin': '珠江流域'},
            '广西': {'lat': 22.8170, 'lon': 108.3661, 'code': '450000', 'river': '西江', 'basin': '珠江流域'},
            '海南': {'lat': 20.0311, 'lon': 110.3312, 'code': '460000', 'river': '南渡江', 'basin': '海南岛流域'},
            '重庆': {'lat': 29.5647, 'lon': 106.5507, 'code': '500000', 'river': '长江', 'basin': '长江流域'},
            '四川': {'lat': 30.6512, 'lon': 104.0665, 'code': '510000', 'river': '长江', 'basin': '长江流域'},
            '贵州': {'lat': 26.5783, 'lon': 106.7074, 'code': '520000', 'river': '乌江', 'basin': '长江流域'},
            '云南': {'lat': 25.0389, 'lon': 102.7183, 'code': '530000', 'river': '金沙江', 'basin': '长江流域'},
            '西藏': {'lat': 29.6465, 'lon': 91.1172, 'code': '540000', 'river': '雅鲁藏布江', 'basin': '雅鲁藏布江流域'},
            '陕西': {'lat': 34.2658, 'lon': 108.9540, 'code': '610000', 'river': '渭河', 'basin': '黄河流域'},
            '甘肃': {'lat': 36.0611, 'lon': 103.8343, 'code': '620000', 'river': '黄河', 'basin': '黄河流域'},
            '青海': {'lat': 36.6232, 'lon': 101.7782, 'code': '630000', 'river': '黄河', 'basin': '黄河流域'},
            '宁夏': {'lat': 38.4872, 'lon': 106.2309, 'code': '640000', 'river': '黄河', 'basin': '黄河流域'},
            '新疆': {'lat': 43.7928, 'lon': 87.6177, 'code': '650000', 'river': '塔里木河', 'basin': '内陆河流域'}
        }
        
        # 水质参数基础范围（基于真实监测数据）
        self.parameter_ranges = {
            'ph': {
                'base': 7.2, 'std': 0.8, 'min': 6.0, 'max': 9.0,
                'seasonal_amplitude': 0.3, 'trend': 0.001
            },
            'dissolved_oxygen': {
                'base': 8.5, 'std': 1.5, 'min': 4.0, 'max': 12.0,
                'seasonal_amplitude': 1.0, 'trend': -0.002
            },
            'ammonia_nitrogen': {
                'base': 0.8, 'std': 0.6, 'min': 0.1, 'max': 5.0,
                'seasonal_amplitude': 0.3, 'trend': 0.001
            },
            'total_phosphorus': {
                'base': 0.15, 'std': 0.1, 'min': 0.02, 'max': 1.0,
                'seasonal_amplitude': 0.05, 'trend': 0.0005
            },
            'cod': {
                'base': 15.0, 'std': 8.0, 'min': 5.0, 'max': 40.0,
                'seasonal_amplitude': 3.0, 'trend': 0.01
            },
            'bod5': {
                'base': 3.5, 'std': 2.0, 'min': 1.0, 'max': 10.0,
                'seasonal_amplitude': 1.0, 'trend': 0.005
            },
            'total_nitrogen': {
                'base': 1.2, 'std': 0.8, 'min': 0.3, 'max': 4.0,
                'seasonal_amplitude': 0.4, 'trend': 0.002
            },
            'temperature': {
                'base': 15.0, 'std': 8.0, 'min': -5.0, 'max': 35.0,
                'seasonal_amplitude': 12.0, 'trend': 0.005
            }
        }
        
        # 流域污染系数（基于真实环境状况）
        self.basin_pollution_factors = {
            '海河流域': {'ph': 1.0, 'dissolved_oxygen': 0.9, 'ammonia_nitrogen': 1.2, 'total_phosphorus': 1.1},
            '黄河流域': {'ph': 0.95, 'dissolved_oxygen': 0.85, 'ammonia_nitrogen': 1.3, 'total_phosphorus': 1.2},
            '长江流域': {'ph': 1.05, 'dissolved_oxygen': 1.1, 'ammonia_nitrogen': 0.9, 'total_phosphorus': 0.95},
            '珠江流域': {'ph': 1.02, 'dissolved_oxygen': 1.05, 'ammonia_nitrogen': 0.95, 'total_phosphorus': 0.9},
            '辽河流域': {'ph': 0.9, 'dissolved_oxygen': 0.8, 'ammonia_nitrogen': 1.4, 'total_phosphorus': 1.3},
            '松花江流域': {'ph': 0.85, 'dissolved_oxygen': 0.75, 'ammonia_nitrogen': 1.5, 'total_phosphorus': 1.4},
            '黑龙江流域': {'ph': 0.88, 'dissolved_oxygen': 0.78, 'ammonia_nitrogen': 1.4, 'total_phosphorus': 1.3},
            '淮河流域': {'ph': 0.92, 'dissolved_oxygen': 0.82, 'ammonia_nitrogen': 1.3, 'total_phosphorus': 1.2},
            '东南诸河流域': {'ph': 1.08, 'dissolved_oxygen': 1.15, 'ammonia_nitrogen': 0.85, 'total_phosphorus': 0.8},
            '海南岛流域': {'ph': 1.1, 'dissolved_oxygen': 1.2, 'ammonia_nitrogen': 0.8, 'total_phosphorus': 0.75},
            '雅鲁藏布江流域': {'ph': 1.15, 'dissolved_oxygen': 1.25, 'ammonia_nitrogen': 0.7, 'total_phosphorus': 0.7},
            '内陆河流域': {'ph': 0.8, 'dissolved_oxygen': 0.7, 'ammonia_nitrogen': 1.6, 'total_phosphorus': 1.5}
        }
    
    def generate_data_hash(self, station_name: str, parameter: str, value: float, timestamp: datetime) -> str:
        """生成数据哈希值"""
        data_string = f"{station_name}_{parameter}_{value:.3f}_{timestamp.isoformat()}"
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def generate_parameter_value(self, parameter: str, province: str, timestamp: datetime, 
                                base_noise: float = 0.1) -> float:
        """生成单个参数值"""
        if parameter not in self.parameter_ranges:
            return 0.0
        
        # 基础参数
        param_config = self.parameter_ranges[parameter]
        base_value = param_config['base']
        std_value = param_config['std']
        min_value = param_config['min']
        max_value = param_config['max']
        
        # 季节性变化
        day_of_year = timestamp.timetuple().tm_yday
        seasonal_component = param_config['seasonal_amplitude'] * np.sin(2 * np.pi * day_of_year / 365)
        
        # 日变化（小时级）
        hour_component = 0.1 * np.sin(2 * np.pi * timestamp.hour / 24)
        
        # 长期趋势
        days_since_start = (timestamp - datetime(2024, 1, 1)).days
        trend_component = param_config['trend'] * days_since_start
        
        # 流域污染因子
        basin_factor = 1.0
        if province in self.provinces:
            basin = self.provinces[province]['basin']
            if basin in self.basin_pollution_factors:
                basin_factor = self.basin_pollution_factors[basin].get(parameter, 1.0)
        
        # 随机噪声
        noise = np.random.normal(0, std_value * base_noise)
        
        # 异常值（5%概率）
        if np.random.random() < 0.05:
            noise *= 3  # 异常值
        
        # 计算最终值
        final_value = (base_value + seasonal_component + hour_component + 
                      trend_component + noise) * basin_factor
        
        # 确保在合理范围内
        final_value = np.clip(final_value, min_value, max_value)
        
        return round(final_value, 3)
    
    def generate_station_data(self, province: str, days: int = 30) -> List[Dict]:
        """生成单个省份的监测数据"""
        if province not in self.provinces:
            return []
        
        province_info = self.provinces[province]
        station_name = f"{province}监测站({province_info['river']}-{province_info['basin']})"
        
        data_points = []
        start_time = datetime.now() - timedelta(days=days)
        
        # 生成每小时的数据
        for i in range(days * 24):
            timestamp = start_time + timedelta(hours=i)
            
            # 为每个参数生成数据
            for parameter in self.parameter_ranges.keys():
                value = self.generate_parameter_value(parameter, province, timestamp)
                data_hash = self.generate_data_hash(station_name, parameter, value, timestamp)
                
                data_points.append({
                    'station_name': station_name,
                    'area_id': province_info['code'],
                    'parameter': parameter,
                    'value': value,
                    'unit': self.get_parameter_unit(parameter),
                    'measurement_time': timestamp,
                    'data_hash': data_hash,
                    'created_at': datetime.now()
                })
        
        return data_points
    
    def get_parameter_unit(self, parameter: str) -> str:
        """获取参数单位"""
        units = {
            'ph': '',
            'dissolved_oxygen': 'mg/L',
            'ammonia_nitrogen': 'mg/L',
            'total_phosphorus': 'mg/L',
            'cod': 'mg/L',
            'bod5': 'mg/L',
            'total_nitrogen': 'mg/L',
            'temperature': '°C'
        }
        return units.get(parameter, 'mg/L')
    
    def extract_province_from_station(self, station_name: str) -> str:
        """从站点名称提取省份"""
        for province in self.provinces.keys():
            if province in station_name:
                return province
        return 'Unknown'
    
    def extract_watershed_from_station(self, station_name: str) -> str:
        """从站点名称提取流域"""
        for province in self.provinces.keys():
            if province in station_name:
                return self.provinces[province]['basin']
        return 'Unknown'
    
    def calculate_water_quality_grade(self, ph: float, do: float, nh3: float, tp: float) -> int:
        """计算水质等级"""
        if ph is None or do is None or nh3 is None or tp is None:
            return 3  # 默认III类
        
        # 基于地表水环境质量标准
        if ph >= 6.0 and ph <= 9.0 and do >= 5.0 and nh3 <= 1.0 and tp <= 0.2:
            return 1  # I类
        elif ph >= 6.0 and ph <= 9.0 and do >= 3.0 and nh3 <= 1.5 and tp <= 0.3:
            return 2  # II类
        elif ph >= 6.0 and ph <= 9.0 and do >= 2.0 and nh3 <= 2.0 and tp <= 0.4:
            return 3  # III类
        elif ph >= 6.0 and ph <= 9.0 and do >= 2.0 and nh3 <= 2.0 and tp <= 0.4:
            return 4  # IV类
        else:
            return 5  # V类
    
    def calculate_pollution_index(self, ph: float, do: float, nh3: float, tp: float, temp: float) -> float:
        """计算污染指数"""
        if ph is None or do is None or nh3 is None or tp is None:
            return 1.0
        
        # 简化的污染指数计算
        ph_factor = 1.0 if 6.5 <= ph <= 8.5 else 1.5
        do_factor = 1.0 if do >= 5.0 else 2.0
        nh3_factor = 1.0 + nh3 * 0.5
        tp_factor = 1.0 + tp * 2.0
        
        return (ph_factor + do_factor + nh3_factor + tp_factor) / 4.0
    
    def store_data_to_db(self, data_points: List[Dict]) -> int:
        """存储数据到数据库"""
        if not data_points:
            return 0
        
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # 批量插入数据
            insert_query = """
                INSERT INTO water_quality_data 
                (station_name, station_code, province, watershed, monitoring_time, 
                 temperature, ph, dissolved_oxygen, conductivity, turbidity, 
                 permanganate_index, ammonia_nitrogen, total_phosphorus, total_nitrogen,
                 chlorophyll_a, algae_density, water_quality_grade, pollution_index, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # 按时间戳分组数据
            grouped_data = {}
            for point in data_points:
                key = (point['station_name'], point['measurement_time'])
                if key not in grouped_data:
                    grouped_data[key] = {
                        'station_name': point['station_name'],
                        'station_code': point['area_id'],
                        'province': self.extract_province_from_station(point['station_name']),
                        'watershed': self.extract_watershed_from_station(point['station_name']),
                        'monitoring_time': point['measurement_time'],
                        'parameters': {}
                    }
                grouped_data[key]['parameters'][point['parameter']] = point['value']
            
            data_tuples = []
            for key, data in grouped_data.items():
                # 提取参数值，如果不存在则设为NULL
                temp = data['parameters'].get('temperature')
                ph = data['parameters'].get('ph')
                do = data['parameters'].get('dissolved_oxygen')
                cond = data['parameters'].get('conductivity')
                turb = data['parameters'].get('turbidity')
                perm = data['parameters'].get('permanganate_index')
                nh3 = data['parameters'].get('ammonia_nitrogen')
                tp = data['parameters'].get('total_phosphorus')
                tn = data['parameters'].get('total_nitrogen')
                chla = data['parameters'].get('chlorophyll_a')
                algae = data['parameters'].get('algae_density')
                
                # 计算水质等级和污染指数
                wq_grade = self.calculate_water_quality_grade(ph, do, nh3, tp)
                pollution_idx = self.calculate_pollution_index(ph, do, nh3, tp, temp)
                
                data_tuples.append((
                    data['station_name'],
                    data['station_code'],
                    data['province'],
                    data['watershed'],
                    data['monitoring_time'],
                    float(temp) if temp is not None else None,
                    float(ph) if ph is not None else None,
                    float(do) if do is not None else None,
                    float(cond) if cond is not None else None,
                    float(turb) if turb is not None else None,
                    float(perm) if perm is not None else None,
                    float(nh3) if nh3 is not None else None,
                    float(tp) if tp is not None else None,
                    float(tn) if tn is not None else None,
                    float(chla) if chla is not None else None,
                    float(algae) if algae is not None else None,
                    int(wq_grade),
                    float(pollution_idx),
                    'SIMULATED'
                ))
            
            cursor.executemany(insert_query, data_tuples)
            conn.commit()
            
            inserted_count = cursor.rowcount
            logger.info(f"Inserted {inserted_count} new records")
            
            cursor.close()
            conn.close()
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return 0
    
    def generate_all_data(self, days: int = 30) -> Dict[str, int]:
        """生成所有省份的数据"""
        logger.info(f"Generating realistic data for {days} days...")
        
        results = {}
        total_inserted = 0
        
        for province in self.provinces.keys():
            logger.info(f"Generating data for {province}...")
            
            # 生成数据
            data_points = self.generate_station_data(province, days)
            
            # 存储数据
            inserted_count = self.store_data_to_db(data_points)
            
            results[province] = inserted_count
            total_inserted += inserted_count
            
            logger.info(f"{province}: {inserted_count} records")
        
        logger.info(f"Total records inserted: {total_inserted}")
        
        return results
    
    def update_data_continuously(self, update_interval_hours: int = 4):
        """持续更新数据"""
        import time
        
        logger.info(f"Starting continuous data update every {update_interval_hours} hours...")
        
        while True:
            try:
                # 生成最新4小时的数据
                logger.info("Updating data...")
                results = self.generate_all_data(days=1)  # 生成1天的数据
                
                # 记录更新结果
                summary = {
                    'timestamp': datetime.now().isoformat(),
                    'update_interval_hours': update_interval_hours,
                    'results': results,
                    'total_records': sum(results.values())
                }
                
                with open('data_update_summary.json', 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Data update completed. Next update in {update_interval_hours} hours.")
                
                # 等待下次更新
                time.sleep(update_interval_hours * 3600)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous data update...")
                break
            except Exception as e:
                logger.error(f"Error in continuous update: {e}")
                time.sleep(300)  # 错误后等待5分钟

def main():
    """主函数"""
    import sys
    import os
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    simulator = RealisticDataSimulator(db_url)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'generate':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            results = simulator.generate_all_data(days)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            
        elif command == 'continuous':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            simulator.update_data_continuously(interval)
            
        else:
            print("Usage: python simulate_realistic_data.py [generate|continuous] [days|interval]")
    else:
        # 默认生成30天数据
        results = simulator.generate_all_data(30)
        print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
