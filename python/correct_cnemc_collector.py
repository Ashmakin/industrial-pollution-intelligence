#!/usr/bin/env python3
"""
正确的CNEMC数据收集器
使用用户提供的真实API链接和参数
"""

import json
import pandas as pd
import numpy as np
import psycopg2
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import time
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectCNEMCCollector:
    """正确的CNEMC数据收集器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.base_url = "https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx"
        
        # 中国各省市代码
        self.area_codes = {
            '北京': '110000',
            '天津': '120000', 
            '河北': '130000',
            '山西': '140000',
            '内蒙古': '150000',
            '辽宁': '210000',
            '吉林': '220000',
            '黑龙江': '230000',
            '上海': '310000',
            '江苏': '320000',
            '浙江': '330000',
            '安徽': '340000',
            '福建': '350000',
            '江西': '360000',
            '山东': '370000',
            '河南': '410000',
            '湖北': '420000',
            '湖南': '430000',
            '广东': '440000',
            '广西': '450000',
            '海南': '460000',
            '重庆': '500000',
            '四川': '510000',
            '贵州': '520000',
            '云南': '530000',
            '西藏': '540000',
            '陕西': '610000',
            '甘肃': '620000',
            '青海': '630000',
            '宁夏': '640000',
            '新疆': '650000'
        }
        
        # 参数映射
        self.parameter_mapping = {
            'pH': 'ph',
            '溶解氧': 'dissolved_oxygen',
            '氨氮': 'ammonia_nitrogen',
            '总磷': 'total_phosphorus',
            '温度': 'temperature',
            '电导率': 'conductivity',
            '浊度': 'turbidity'
        }
    
    def fetch_cnemc_data(self, area_id: str) -> Dict:
        """获取CNEMC数据"""
        try:
            params = {
                'AreaID': area_id,
                'RiverID': '',
                'MNName': '',
                'PageIndex': -1,
                'PageSize': 60,
                'action': 'getRealDatas'
            }
            
            logger.info(f"Fetching data for area {area_id}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # 尝试解析JSON响应
            try:
                data = response.json()
                logger.info(f"Successfully fetched JSON data for area {area_id}")
                return data
            except json.JSONDecodeError:
                # 如果不是JSON，尝试解析为文本
                text_data = response.text.strip()
                logger.info(f"Received text response for area {area_id}: {text_data}")
                
                if text_data == "-1":
                    logger.warning(f"No data available for area {area_id}")
                    return {"data": [], "total": 0}
                else:
                    # 尝试解析其他格式
                    return {"raw_data": text_data, "total": 0}
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for area {area_id}: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error for area {area_id}: {e}")
            return {"error": str(e)}
    
    def parse_cnemc_response(self, data: Dict, area_id: str) -> List[Dict]:
        """解析CNEMC响应数据"""
        parsed_data = []
        
        try:
            if "tbody" in data and isinstance(data["tbody"], list):
                # 处理表格数据格式
                logger.info(f"Processing table data for area {area_id}")
                
                for row in data["tbody"]:
                    if isinstance(row, list) and len(row) > 0:
                        parsed_item = self.parse_table_row(row, area_id)
                        if parsed_item:
                            parsed_data.append(parsed_item)
            elif "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        parsed_item = self.parse_data_item(item, area_id)
                        if parsed_item:
                            parsed_data.append(parsed_item)
            elif "raw_data" in data:
                # 处理原始文本数据
                logger.info(f"Processing raw data for area {area_id}")
                # 这里可以添加更多解析逻辑
                
        except Exception as e:
            logger.error(f"Error parsing data for area {area_id}: {e}")
        
        return parsed_data
    
    def parse_table_row(self, row: List, area_id: str) -> Optional[Dict]:
        """解析表格行数据"""
        try:
            # 表格列的顺序：省份、流域、断面名称、监测时间、水质类别、水温、pH、溶解氧、电导率、浊度、高锰酸盐指数、氨氮、总磷、总氮、叶绿素α、藻密度、站点情况
            if len(row) < 17:
                return None
            
            # 提取基本信息
            province = row[0] if row[0] else "未知"
            watershed = row[1] if row[1] else "未知"
            
            # 提取断面名称（去除HTML标签）
            station_name_raw = row[2]
            station_name = self.extract_station_name(station_name_raw)
            
            monitoring_time_raw = row[3]
            monitoring_time = self.parse_monitoring_time(monitoring_time_raw)
            
            # 提取参数数据
            parameter_data = {}
            
            # 水温 (索引5)
            if row[5] and row[5] != "*":
                temp_value = self.extract_numeric_value(row[5])
                if temp_value is not None:
                    parameter_data['temperature'] = temp_value
            
            # pH (索引6)
            if row[6] and row[6] != "*":
                ph_value = self.extract_numeric_value(row[6])
                if ph_value is not None:
                    parameter_data['ph'] = ph_value
            
            # 溶解氧 (索引7)
            if row[7] and row[7] != "*":
                do_value = self.extract_numeric_value(row[7])
                if do_value is not None:
                    parameter_data['dissolved_oxygen'] = do_value
            
            # 电导率 (索引8)
            if row[8] and row[8] != "*":
                ec_value = self.extract_numeric_value(row[8])
                if ec_value is not None:
                    parameter_data['conductivity'] = ec_value
            
            # 浊度 (索引9)
            if row[9] and row[9] != "*":
                turb_value = self.extract_numeric_value(row[9])
                if turb_value is not None:
                    parameter_data['turbidity'] = turb_value
            
            # 氨氮 (索引11)
            if row[11] and row[11] != "*":
                nh3n_value = self.extract_numeric_value(row[11])
                if nh3n_value is not None:
                    parameter_data['ammonia_nitrogen'] = nh3n_value
            
            # 总磷 (索引12)
            if row[12] and row[12] != "*":
                tp_value = self.extract_numeric_value(row[12])
                if tp_value is not None:
                    parameter_data['total_phosphorus'] = tp_value
            
            if not parameter_data:
                return None
            
            # 计算水质等级和污染指数
            water_quality_grade = self.calculate_water_quality_grade(parameter_data)
            pollution_index = self.calculate_pollution_index(parameter_data)
            
            result = {
                'station_name': station_name,
                'station_code': area_id,
                'monitoring_time': monitoring_time,
                'province': province,
                'watershed': watershed,
                'water_quality_grade': water_quality_grade,
                'pollution_index': pollution_index,
                **parameter_data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing table row: {e}")
            return None
    
    def extract_station_name(self, station_name_raw: str) -> str:
        """从HTML中提取站点名称"""
        try:
            import re
            # 移除HTML标签
            clean_name = re.sub(r'<[^>]+>', '', station_name_raw)
            # 移除多余的空白字符
            clean_name = re.sub(r'\s+', ' ', clean_name).strip()
            return clean_name if clean_name else "未知站点"
        except Exception:
            return "未知站点"
    
    def extract_numeric_value(self, value_str: str) -> Optional[float]:
        """从HTML中提取数值"""
        try:
            import re
            if not value_str or value_str == "*":
                return None
            
            # 移除HTML标签
            clean_value = re.sub(r'<[^>]+>', '', value_str)
            
            # 提取数值（支持小数）
            match = re.search(r'(\d+\.?\d*)', clean_value)
            if match:
                return float(match.group(1))
            
            return None
        except Exception:
            return None
    
    def parse_monitoring_time(self, time_str: str) -> str:
        """解析监测时间"""
        try:
            if not time_str or time_str == "*":
                return datetime.now().isoformat()
            
            # 格式：10-22 12:00
            if "-" in time_str and ":" in time_str:
                current_year = datetime.now().year
                time_str = f"{current_year}-{time_str}"
                parsed_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                return parsed_time.isoformat()
            
            return datetime.now().isoformat()
        except Exception:
            return datetime.now().isoformat()
    
    def parse_data_item(self, item: Dict, area_id: str) -> Optional[Dict]:
        """解析单个数据项"""
        try:
            # 提取基本信息
            station_name = item.get('MNName', f'监测站_{area_id}')
            station_code = item.get('MNCode', area_id)
            monitoring_time = item.get('MonitorTime', datetime.now().isoformat())
            
            # 提取参数数据
            parameter_data = {}
            
            # 常见的水质参数
            parameters = {
                'pH': item.get('pH'),
                '溶解氧': item.get('DO'),  # Dissolved Oxygen
                '氨氮': item.get('NH3N'),  # Ammonia Nitrogen
                '总磷': item.get('TP'),    # Total Phosphorus
                '温度': item.get('WT'),    # Water Temperature
                '电导率': item.get('EC'),   # Electrical Conductivity
                '浊度': item.get('TU')     # Turbidity
            }
            
            # 处理参数值
            for param_name, value in parameters.items():
                if value is not None and value != '' and value != '-':
                    try:
                        numeric_value = float(value)
                        if not np.isnan(numeric_value):
                            parameter_data[self.parameter_mapping.get(param_name, param_name.lower())] = numeric_value
                    except (ValueError, TypeError):
                        continue
            
            if not parameter_data:
                logger.warning(f"No valid parameter data found for station {station_name}")
                return None
            
            # 计算水质等级和污染指数
            water_quality_grade = self.calculate_water_quality_grade(parameter_data)
            pollution_index = self.calculate_pollution_index(parameter_data)
            
            # 提取省份和流域信息
            province = self.get_province_from_area_id(area_id)
            watershed = self.get_watershed_from_station_name(station_name)
            
            result = {
                'station_name': station_name,
                'station_code': station_code,
                'monitoring_time': monitoring_time,
                'province': province,
                'watershed': watershed,
                'water_quality_grade': water_quality_grade,
                'pollution_index': pollution_index,
                **parameter_data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing data item: {e}")
            return None
    
    def calculate_water_quality_grade(self, parameters: Dict) -> int:
        """计算水质等级"""
        try:
            # 基于主要参数计算水质等级
            grade = 1  # 默认优秀
            
            # pH值评分
            if 'ph' in parameters:
                ph = parameters['ph']
                if ph < 6.0 or ph > 9.0:
                    grade = max(grade, 5)  # 重度污染
                elif ph < 6.5 or ph > 8.5:
                    grade = max(grade, 3)  # 轻度污染
            
            # 溶解氧评分
            if 'dissolved_oxygen' in parameters:
                do = parameters['dissolved_oxygen']
                if do < 2.0:
                    grade = max(grade, 6)  # 严重污染
                elif do < 3.0:
                    grade = max(grade, 5)  # 重度污染
                elif do < 5.0:
                    grade = max(grade, 3)  # 轻度污染
            
            # 氨氮评分
            if 'ammonia_nitrogen' in parameters:
                nh3n = parameters['ammonia_nitrogen']
                if nh3n > 5.0:
                    grade = max(grade, 6)  # 严重污染
                elif nh3n > 2.0:
                    grade = max(grade, 5)  # 重度污染
                elif nh3n > 1.0:
                    grade = max(grade, 3)  # 轻度污染
            
            return grade
            
        except Exception as e:
            logger.error(f"Error calculating water quality grade: {e}")
            return 1
    
    def calculate_pollution_index(self, parameters: Dict) -> float:
        """计算污染指数"""
        try:
            index = 0.0
            count = 0
            
            # 基于主要参数计算污染指数
            if 'ph' in parameters:
                ph = parameters['ph']
                # pH偏离7.0的程度
                index += abs(ph - 7.0) * 2
                count += 1
            
            if 'dissolved_oxygen' in parameters:
                do = parameters['dissolved_oxygen']
                # 溶解氧越低，污染指数越高
                if do < 7.5:
                    index += (7.5 - do) * 0.5
                count += 1
            
            if 'ammonia_nitrogen' in parameters:
                nh3n = parameters['ammonia_nitrogen']
                # 氨氮浓度越高，污染指数越高
                index += nh3n * 2
                count += 1
            
            if 'total_phosphorus' in parameters:
                tp = parameters['total_phosphorus']
                # 总磷浓度越高，污染指数越高
                index += tp * 5
                count += 1
            
            return index / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating pollution index: {e}")
            return 0.0
    
    def get_province_from_area_id(self, area_id: str) -> str:
        """从区域ID获取省份名称"""
        for province, code in self.area_codes.items():
            if code == area_id:
                return province
        return "未知"
    
    def get_watershed_from_station_name(self, station_name: str) -> str:
        """从站点名称获取流域信息"""
        # 常见的流域关键词
        watersheds = {
            '长江': ['长江', '扬子江', '江'],
            '黄河': ['黄河', '河'],
            '珠江': ['珠江', '珠'],
            '松花江': ['松花江', '松花'],
            '辽河': ['辽河', '辽'],
            '海河': ['海河', '海'],
            '淮河': ['淮河', '淮'],
            '闽江': ['闽江', '闽'],
            '钱塘江': ['钱塘江', '钱塘'],
            '瓯江': ['瓯江', '瓯']
        }
        
        for watershed, keywords in watersheds.items():
            for keyword in keywords:
                if keyword in station_name:
                    return watershed
        
        return "其他"
    
    def store_data_to_db(self, data_points: List[Dict]) -> int:
        """存储数据到数据库"""
        if not data_points:
            return 0
        
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # 准备插入语句
            insert_query = """
                INSERT INTO water_quality_data 
                (station_name, station_code, ph, temperature, dissolved_oxygen, ammonia_nitrogen, total_phosphorus, conductivity, turbidity, monitoring_time, province, watershed, water_quality_grade, pollution_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            inserted_count = 0
            
            for point in data_points:
                try:
                    cursor.execute(insert_query, (
                        point.get('station_name'),
                        point.get('station_code'),
                        point.get('ph'),
                        point.get('temperature'),
                        point.get('dissolved_oxygen'),
                        point.get('ammonia_nitrogen'),
                        point.get('total_phosphorus'),
                        point.get('conductivity'),
                        point.get('turbidity'),
                        point.get('monitoring_time'),
                        point.get('province'),
                        point.get('watershed'),
                        point.get('water_quality_grade'),
                        point.get('pollution_index')
                    ))
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting data point: {e}")
                    continue
            
            conn.commit()
            logger.info(f"Successfully inserted {inserted_count} records")
            
            cursor.close()
            conn.close()
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return 0
    
    def collect_all_data(self) -> Dict:
        """收集所有地区的数据"""
        total_collected = 0
        results = {}
        
        for province, area_id in self.area_codes.items():
            try:
                logger.info(f"Collecting data for {province} ({area_id})")
                
                # 获取数据
                raw_data = self.fetch_cnemc_data(area_id)
                
                # 解析数据
                parsed_data = self.parse_cnemc_response(raw_data, area_id)
                
                # 存储数据
                if parsed_data:
                    stored_count = self.store_data_to_db(parsed_data)
                    total_collected += stored_count
                    results[province] = {
                        'area_id': area_id,
                        'raw_records': len(parsed_data),
                        'stored_records': stored_count,
                        'status': 'success'
                    }
                else:
                    results[province] = {
                        'area_id': area_id,
                        'raw_records': 0,
                        'stored_records': 0,
                        'status': 'no_data'
                    }
                
                # 避免请求过于频繁
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting data for {province}: {e}")
                results[province] = {
                    'area_id': area_id,
                    'error': str(e),
                    'status': 'error'
                }
        
        return {
            'total_collected': total_collected,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """主函数"""
    import sys
    import json
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    collector = CorrectCNEMCCollector(db_url)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 测试模式 - 只收集一个地区
        test_area_id = '110000'  # 北京
        logger.info(f"Testing with area {test_area_id}")
        
        raw_data = collector.fetch_cnemc_data(test_area_id)
        parsed_data = collector.parse_cnemc_response(raw_data, test_area_id)
        
        result = {
            'test_area_id': test_area_id,
            'raw_data': raw_data,
            'parsed_data': parsed_data,
            'timestamp': datetime.now().isoformat()
        }
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 正常模式 - 收集所有地区
        result = collector.collect_all_data()
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
