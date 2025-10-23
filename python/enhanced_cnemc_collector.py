#!/usr/bin/env python3
"""
增强的CNEMC数据收集器
支持完整的省市、流域、监测站选择
实现自动数据采集和智能去重
"""

import json
import pandas as pd
import numpy as np
import psycopg2
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import requests
import time
import os
import hashlib
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataRecord:
    """数据记录类"""
    province: str
    basin: str
    station_name: str
    monitoring_time: datetime
    water_quality_class: str
    temperature: Optional[float]
    ph: Optional[float]
    dissolved_oxygen: Optional[float]
    conductivity: Optional[float]
    turbidity: Optional[float]
    permanganate_index: Optional[float]
    ammonia_nitrogen: Optional[float]
    total_phosphorus: Optional[float]
    total_nitrogen: Optional[float]
    chlorophyll_a: Optional[float]
    algae_density: Optional[float]
    station_status: str
    
    def to_hash(self) -> str:
        """生成数据记录的哈希值用于去重"""
        key_data = f"{self.province}_{self.basin}_{self.station_name}_{self.monitoring_time.isoformat()}"
        return hashlib.md5(key_data.encode()).hexdigest()

class EnhancedCNEMCCollector:
    """增强的CNEMC数据收集器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.base_url = "https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx"
        
        # 完整的中国各省市代码
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
            '水温': 'temperature',
            '电导率': 'conductivity',
            '浊度': 'turbidity',
            '高锰酸盐指数': 'permanganate_index',
            '总氮': 'total_nitrogen',
            '叶绿素α': 'chlorophyll_a',
            '藻密度': 'algae_density'
        }
        
        # 流域映射
        self.basin_mapping = {
            '海河流域': 'haihe',
            '黄河流域': 'yellow_river',
            '长江流域': 'yangtze',
            '珠江流域': 'pearl_river',
            '松花江流域': 'songhua',
            '辽河流域': 'liaohe',
            '淮河流域': 'huaihe',
            '太湖流域': 'taihu',
            '巢湖流域': 'chaohu',
            '滇池流域': 'dianchi',
            '其他': 'other'
        }
    
    def fetch_cnemc_data(self, area_id: str, river_id: str = '', station_name: str = '') -> Dict:
        """获取CNEMC数据"""
        try:
            params = {
                'AreaID': area_id,
                'RiverID': river_id,
                'MNName': station_name,
                'PageIndex': -1,
                'PageSize': 100,  # 增加页面大小
                'action': 'getRealDatas'
            }
            
            logger.info(f"Fetching data for area {area_id}, river {river_id}, station {station_name}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched data: {data.get('records', 0)} records")
            
            # 检查响应格式
            if isinstance(data, dict) and 'result' in data:
                return data
            else:
                logger.error(f"Unexpected response format: {type(data)}")
                return {"result": 0, "records": 0, "tbody": []}
            
        except Exception as e:
            logger.error(f"Error fetching data for area {area_id}: {e}")
            return {"result": 0, "records": 0, "tbody": []}
    
    def parse_data_record(self, row: List, headers: List[str]) -> Optional[DataRecord]:
        """解析单条数据记录"""
        try:
            if len(row) < len(headers):
                return None
            
            # 解析监测时间
            time_str = row[3]  # 监测时间
            if time_str and time_str != "null" and time_str != "--":
                try:
                    # 处理格式如 "10-22 16:00"
                    if "-" in time_str and ":" in time_str:
                        # 添加当前年份
                        current_year = datetime.now().year
                        full_time_str = f"{current_year}-{time_str}"
                        monitoring_time = datetime.strptime(full_time_str, "%Y-%m-%d %H:%M")
                    else:
                        monitoring_time = datetime.now()
                except Exception as e:
                    logger.warning(f"Failed to parse time '{time_str}': {e}")
                    monitoring_time = datetime.now()
            else:
                monitoring_time = datetime.now()
            
            # 解析数值数据
            def parse_value(value_str):
                if not value_str or value_str == '*' or value_str.strip() == '':
                    return None
                try:
                    # 提取原始值
                    if 'title=' in value_str:
                        import re
                        match = re.search(r'原始值：([0-9.-]+)', value_str)
                        if match:
                            return float(match.group(1))
                    return float(value_str.strip())
                except:
                    return None
            
            # 解析站点名称，去除HTML标签
            def clean_station_name(name_str):
                import re
                if '<span' in name_str:
                    # 提取span标签内的文本
                    match = re.search(r'>([^<]+)<', name_str)
                    if match:
                        return match.group(1).strip()
                return name_str.strip()
            
            return DataRecord(
                province=row[0].strip(),
                basin=row[1].strip(),
                station_name=clean_station_name(row[2]),
                monitoring_time=monitoring_time,
                water_quality_class=row[4].strip(),
                temperature=parse_value(row[5]),
                ph=parse_value(row[6]),
                dissolved_oxygen=parse_value(row[7]),
                conductivity=parse_value(row[8]),
                turbidity=parse_value(row[9]),
                permanganate_index=parse_value(row[10]),
                ammonia_nitrogen=parse_value(row[11]),
                total_phosphorus=parse_value(row[12]),
                total_nitrogen=parse_value(row[13]),
                chlorophyll_a=parse_value(row[14]),
                algae_density=parse_value(row[15]),
                station_status=row[16].strip() if len(row) > 16 else '正常'
            )
            
        except Exception as e:
            logger.error(f"Error parsing data record: {e}")
            return None
    
    def get_existing_data_hashes(self) -> Set[str]:
        """获取数据库中已存在的数据哈希值"""
        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # 查询已存在的数据哈希
            cursor.execute("""
                SELECT data_hash FROM water_quality_data 
                WHERE data_hash IS NOT NULL
            """)
            
            hashes = {row[0] for row in cursor.fetchall()}
            logger.info(f"Found {len(hashes)} existing data hashes")
            return hashes
            
        except Exception as e:
            logger.error(f"Error getting existing data hashes: {e}")
            return set()
        finally:
            if conn:
                conn.close()
    
    def insert_data_batch(self, records: List[DataRecord], existing_hashes: Set[str]) -> Tuple[int, int]:
        """批量插入数据，返回(插入数量, 跳过数量)"""
        if not records:
            return 0, 0
        
        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            inserted_count = 0
            skipped_count = 0
            
            for record in records:
                # 检查是否已存在
                record_hash = record.to_hash()
                if record_hash in existing_hashes:
                    skipped_count += 1
                    continue
                
                # 插入新数据
                cursor.execute("""
                    INSERT INTO water_quality_data (
                        province, basin, station_name, monitoring_time, water_quality_class,
                        temperature, ph, dissolved_oxygen, conductivity, turbidity,
                        permanganate_index, ammonia_nitrogen, total_phosphorus, total_nitrogen,
                        chlorophyll_a, algae_density, station_status, data_hash, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    record.province, record.basin, record.station_name, record.monitoring_time,
                    record.water_quality_class, record.temperature, record.ph, record.dissolved_oxygen,
                    record.conductivity, record.turbidity, record.permanganate_index,
                    record.ammonia_nitrogen, record.total_phosphorus, record.total_nitrogen,
                    record.chlorophyll_a, record.algae_density, record.station_status,
                    record_hash, datetime.now()
                ))
                
                existing_hashes.add(record_hash)
                inserted_count += 1
            
            conn.commit()
            logger.info(f"Inserted {inserted_count} new records, skipped {skipped_count} duplicates")
            return inserted_count, skipped_count
            
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            if conn:
                conn.rollback()
            return 0, 0
        finally:
            if conn:
                conn.close()
    
    def collect_all_data(self, selected_areas: List[str] = None) -> Dict:
        """收集所有地区的数据"""
        if selected_areas is None:
            selected_areas = list(self.area_codes.keys())
        
        logger.info(f"Starting data collection for {len(selected_areas)} areas")
        
        # 获取已存在的数据哈希
        existing_hashes = self.get_existing_data_hashes()
        
        total_inserted = 0
        total_skipped = 0
        collection_results = {}
        
        for area_name in selected_areas:
            area_id = self.area_codes.get(area_name)
            if not area_id:
                logger.warning(f"Unknown area: {area_name}")
                continue
            
            logger.info(f"Collecting data for {area_name} ({area_id})")
            
            try:
                # 获取数据
                data = self.fetch_cnemc_data(area_id)
                
                if data.get('result') != 1 or not data.get('tbody'):
                    logger.warning(f"No data found for {area_name}")
                    collection_results[area_name] = {'inserted': 0, 'skipped': 0, 'error': 'No data'}
                    continue
                
                # 解析数据
                headers = data.get('thead', [])
                records = []
                
                for row in data['tbody']:
                    record = self.parse_data_record(row, headers)
                    if record:
                        records.append(record)
                
                # 批量插入数据
                inserted, skipped = self.insert_data_batch(records, existing_hashes)
                
                total_inserted += inserted
                total_skipped += skipped
                
                collection_results[area_name] = {
                    'inserted': inserted,
                    'skipped': skipped,
                    'total_records': len(records)
                }
                
                logger.info(f"{area_name}: inserted {inserted}, skipped {skipped}")
                
                # 避免请求过于频繁
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting data for {area_name}: {e}")
                collection_results[area_name] = {'inserted': 0, 'skipped': 0, 'error': str(e)}
        
        return {
            'total_inserted': total_inserted,
            'total_skipped': total_skipped,
            'collection_results': collection_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_available_areas(self) -> Dict[str, str]:
        """获取可用的地区列表"""
        return self.area_codes.copy()
    
    def get_available_basins(self) -> List[str]:
        """获取可用的流域列表"""
        return list(self.basin_mapping.keys())
    
    def get_available_stations(self, area_id: str) -> List[Dict]:
        """获取指定地区的监测站列表"""
        try:
            data = self.fetch_cnemc_data(area_id)
            if data.get('result') != 1 or not data.get('tbody'):
                return []
            
            stations = []
            seen_stations = set()
            
            for row in data['tbody']:
                if len(row) > 2:
                    station_name = row[2].strip()
                    basin = row[1].strip()
                    
                    if station_name not in seen_stations:
                        stations.append({
                            'name': station_name,
                            'basin': basin,
                            'area_id': area_id
                        })
                        seen_stations.add(station_name)
            
            return stations
            
        except Exception as e:
            logger.error(f"Error getting stations for area {area_id}: {e}")
            return []

def main():
    """主函数"""
    import sys
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    collector = EnhancedCNEMCCollector(db_url)
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_cnemc_collector.py <command> [args...]")
        print("Commands:")
        print("  collect [area1,area2,...] - Collect data for specified areas")
        print("  collect-all - Collect data for all areas")
        print("  areas - List available areas")
        print("  basins - List available basins")
        print("  stations <area_id> - List stations for area")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'collect':
        areas = sys.argv[2].split(',') if len(sys.argv) > 2 else None
        result = collector.collect_all_data(areas)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == 'collect-all':
        result = collector.collect_all_data()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == 'areas':
        areas = collector.get_available_areas()
        print(json.dumps(areas, indent=2, ensure_ascii=False))
    
    elif command == 'basins':
        basins = collector.get_available_basins()
        print(json.dumps(basins, indent=2, ensure_ascii=False))
    
    elif command == 'stations':
        if len(sys.argv) < 3:
            print("Error: area_id required for stations command")
            sys.exit(1)
        area_id = sys.argv[2]
        stations = collector.get_available_stations(area_id)
        print(json.dumps(stations, indent=2, ensure_ascii=False))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
