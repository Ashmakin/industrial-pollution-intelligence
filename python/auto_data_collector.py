                      
"""
自动数据采集系统
- 每4小时自动采集全部地区数据
- 查重检验机制
- 支持监测断面选择
- 数据质量验证
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from bs4 import BeautifulSoup
import time
import schedule
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

      
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """数据点结构"""
    station_name: str
    area_id: str
    parameter: str
    value: float
    unit: str
    timestamp: datetime
    data_hash: str        

class AutoDataCollector:
    """自动数据采集器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
                    
        self.monitoring_areas = {
            '110000': {'name': '北京', 'river': '永定河', 'basin': '海河流域'},
            '120000': {'name': '天津', 'river': '海河', 'basin': '海河流域'},
            '130000': {'name': '河北', 'river': '滦河', 'basin': '海河流域'},
            '140000': {'name': '山西', 'river': '汾河', 'basin': '黄河流域'},
            '150000': {'name': '内蒙古', 'river': '黄河', 'basin': '黄河流域'},
            '210000': {'name': '辽宁', 'river': '辽河', 'basin': '辽河流域'},
            '220000': {'name': '吉林', 'river': '松花江', 'basin': '松花江流域'},
            '230000': {'name': '黑龙江', 'river': '黑龙江', 'basin': '黑龙江流域'},
            '310000': {'name': '上海', 'river': '黄浦江', 'basin': '长江流域'},
            '320000': {'name': '江苏', 'river': '长江', 'basin': '长江流域'},
            '330000': {'name': '浙江', 'river': '钱塘江', 'basin': '长江流域'},
            '340000': {'name': '安徽', 'river': '淮河', 'basin': '淮河流域'},
            '350000': {'name': '福建', 'river': '闽江', 'basin': '东南诸河流域'},
            '360000': {'name': '江西', 'river': '赣江', 'basin': '长江流域'},
            '370000': {'name': '山东', 'river': '黄河', 'basin': '黄河流域'},
            '410000': {'name': '河南', 'river': '黄河', 'basin': '黄河流域'},
            '420000': {'name': '湖北', 'river': '长江', 'basin': '长江流域'},
            '430000': {'name': '湖南', 'river': '湘江', 'basin': '长江流域'},
            '440000': {'name': '广东', 'river': '珠江', 'basin': '珠江流域'},
            '450000': {'name': '广西', 'river': '西江', 'basin': '珠江流域'},
            '460000': {'name': '海南', 'river': '南渡江', 'basin': '海南岛流域'},
            '500000': {'name': '重庆', 'river': '长江', 'basin': '长江流域'},
            '510000': {'name': '四川', 'river': '长江', 'basin': '长江流域'},
            '520000': {'name': '贵州', 'river': '乌江', 'basin': '长江流域'},
            '530000': {'name': '云南', 'river': '金沙江', 'basin': '长江流域'},
            '540000': {'name': '西藏', 'river': '雅鲁藏布江', 'basin': '雅鲁藏布江流域'},
            '610000': {'name': '陕西', 'river': '渭河', 'basin': '黄河流域'},
            '620000': {'name': '甘肃', 'river': '黄河', 'basin': '黄河流域'},
            '630000': {'name': '青海', 'river': '黄河', 'basin': '黄河流域'},
            '640000': {'name': '宁夏', 'river': '黄河', 'basin': '黄河流域'},
            '650000': {'name': '新疆', 'river': '塔里木河', 'basin': '内陆河流域'}
        }
        
              
        self.parameter_mapping = {
            'pH': 'ph',
            '溶解氧': 'dissolved_oxygen',
            '高锰酸盐指数': 'permanganate_index',
            '化学需氧量': 'cod',
            '五日生化需氧量': 'bod5',
            '氨氮': 'ammonia_nitrogen',
            '总磷': 'total_phosphorus',
            '总氮': 'total_nitrogen',
            '铜': 'copper',
            '锌': 'zinc',
            '氟化物': 'fluoride',
            '硒': 'selenium',
            '砷': 'arsenic',
            '汞': 'mercury',
            '镉': 'cadmium',
            '六价铬': 'chromium_vi',
            '铅': 'lead',
            '氰化物': 'cyanide',
            '挥发酚': 'volatile_phenol',
            '石油类': 'petroleum',
            '阴离子表面活性剂': 'surfactant',
            '硫化物': 'sulfide',
            '粪大肠菌群': 'fecal_coliform',
            '硫酸盐': 'sulfate',
            '氯化物': 'chloride',
            '硝酸盐氮': 'nitrate_nitrogen',
            '铁': 'iron',
            '锰': 'manganese'
        }
    
    def generate_data_hash(self, station_name: str, parameter: str, value: float, timestamp: datetime) -> str:
        """生成数据哈希值用于查重"""
        data_string = f"{station_name}_{parameter}_{value}_{timestamp.isoformat()}"
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def extract_province_from_station(self, station_name: str) -> Optional[str]:
        """从站点名称提取省份"""
        for province in self.monitoring_areas.values():
            if province['name'] in station_name:
                return province['name']
        return None
    
    def extract_watershed_from_station(self, station_name: str) -> Optional[str]:
        """从站点名称提取流域"""
        for area_info in self.monitoring_areas.values():
            if area_info['name'] in station_name:
                return area_info['basin']
        return None
    
    def calculate_water_quality_grade(self, parameter: str, value: float) -> int:
        """计算水质等级 (1-5级)"""
        if parameter == 'ph':
            if 6.5 <= value <= 8.5:
                return 1      
            elif 6.0 <= value <= 9.0:
                return 2      
            elif 5.5 <= value <= 9.5:
                return 3        
            elif 5.0 <= value <= 10.0:
                return 4        
            else:
                return 5        
        elif parameter == 'dissolved_oxygen':
            if value >= 7.5:
                return 1
            elif value >= 6.0:
                return 2
            elif value >= 5.0:
                return 3
            elif value >= 3.0:
                return 4
            else:
                return 5
        elif parameter == 'ammonia_nitrogen':
            if value <= 0.15:
                return 1
            elif value <= 0.5:
                return 2
            elif value <= 1.0:
                return 3
            elif value <= 2.0:
                return 4
            else:
                return 5
        elif parameter == 'total_phosphorus':
            if value <= 0.02:
                return 1
            elif value <= 0.1:
                return 2
            elif value <= 0.2:
                return 3
            elif value <= 0.3:
                return 4
            else:
                return 5
        else:
            return 3        
    
    def calculate_pollution_index(self, parameter: str, value: float) -> float:
        """计算污染指数"""
        grade = self.calculate_water_quality_grade(parameter, value)
        return float(grade * 20)               
    
    def fetch_cnemc_data(self, area_id: str) -> List[Dict]:
        """获取CNEMC数据"""
        try:
            url = f"https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx?AreaID=110000&RiverID=&MNName=&PageIndex=-1&PageSize=60&action=getRealDatas"
            params = {'AreaID': area_id}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success') and data.get('data'):
                return data['data']
            else:
                logger.warning(f"Area {area_id}: No data returned")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching data for area {area_id}: {e}")
            return []
    
    def parse_cnemc_response(self, data: List[Dict], area_info: Dict) -> List[DataPoint]:
        """解析CNEMC响应数据"""
        data_points = []
        current_time = datetime.now()
        
        for item in data:
            try:
                station_name = item.get('StationName', 'Unknown Station')
                                
                full_station_name = f"{station_name}({area_info['river']}-{area_info['basin']})"
                
                       
                measurements = item.get('Measurements', [])
                for measurement in measurements:
                    param_name = measurement.get('ParamName')
                    if param_name in self.parameter_mapping:
                              
                        value_text = measurement.get('Value', '')
                        try:
                                    
                            if '>' in value_text:
                                value = float(value_text.replace('>', '')) * 1.1         
                            elif '<' in value_text:
                                value = float(value_text.replace('<', '')) * 0.9         
                            else:
                                value = float(value_text)
                            
                                   
                            data_point = DataPoint(
                                station_name=full_station_name,
                                area_id=item.get('AreaID', ''),
                                parameter=self.parameter_mapping[param_name],
                                value=value,
                                unit=measurement.get('Unit', ''),
                                timestamp=current_time,
                                data_hash=''        
                            )
                            
                                   
                            data_point.data_hash = self.generate_data_hash(
                                data_point.station_name,
                                data_point.parameter,
                                data_point.value,
                                data_point.timestamp
                            )
                            
                            data_points.append(data_point)
                            
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid value format: {value_text} for {param_name}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error parsing item: {e}")
                continue
        
        return data_points
    
    def check_duplicates(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """检查并过滤重复数据"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
                                     
            cursor.execute("""
                SELECT DISTINCT station_name, monitoring_time 
                FROM water_quality_data 
                WHERE monitoring_time >= %s
            """, (datetime.now() - timedelta(hours=4),))
            
            existing_records = {(row[0], row[1]) for row in cursor.fetchall()}
            
                    
            new_data_points = []
            duplicate_count = 0
            
            for point in data_points:
                                  
                record_key = (point.station_name, point.timestamp)
                if record_key not in existing_records:
                    new_data_points.append(point)
                else:
                    duplicate_count += 1
            
            logger.info(f"Filtered {duplicate_count} duplicate records, {len(new_data_points)} new records")
            
            cursor.close()
            conn.close()
            
            return new_data_points
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
            return data_points               
    
    def store_data_to_db(self, data_points: List[DataPoint]) -> int:
        """存储数据到数据库"""
        if not data_points:
            return 0
        
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
                                 
            insert_query = """
                INSERT INTO water_quality_data 
                (station_name, station_code, ph, temperature, dissolved_oxygen, ammonia_nitrogen, total_phosphorus, conductivity, turbidity, monitoring_time, province, watershed, water_quality_grade, pollution_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
                    
            data_tuples = []
            for point in data_points:
                           
                province = self.extract_province_from_station(point.station_name)
                watershed = self.extract_watershed_from_station(point.station_name)
                
                             
                wq_grade = self.calculate_water_quality_grade(point.parameter, point.value)
                pollution_index = self.calculate_pollution_index(point.parameter, point.value)
                
                              
                ph = point.value if point.parameter == 'ph' else None
                temperature = point.value if point.parameter == 'temperature' else None
                dissolved_oxygen = point.value if point.parameter == 'dissolved_oxygen' else None
                ammonia_nitrogen = point.value if point.parameter == 'ammonia_nitrogen' else None
                total_phosphorus = point.value if point.parameter == 'total_phosphorus' else None
                conductivity = point.value if point.parameter == 'conductivity' else None
                turbidity = point.value if point.parameter == 'turbidity' else None
                
                data_tuples.append((
                    point.station_name,
                    point.area_id,                  
                    ph,
                    temperature,
                    dissolved_oxygen,
                    ammonia_nitrogen,
                    total_phosphorus,
                    conductivity,
                    turbidity,
                    point.timestamp,
                    province,
                    watershed,
                    wq_grade,
                    pollution_index
                ))
            
            cursor.executemany(insert_query, data_tuples)
            conn.commit()
            
            inserted_count = len(data_tuples)
            logger.info(f"Successfully inserted {inserted_count} new records")
            
            cursor.close()
            conn.close()
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return 0
    
    def collect_area_data(self, area_id: str, area_info: Dict) -> int:
        """采集单个区域数据"""
        logger.info(f"Collecting data for {area_info['name']} ({area_id})")
        
              
        raw_data = self.fetch_cnemc_data(area_id)
        if not raw_data:
            logger.warning(f"No data received for {area_info['name']}")
            return 0
        
              
        data_points = self.parse_cnemc_response(raw_data, area_info)
        if not data_points:
            logger.warning(f"No valid data points for {area_info['name']}")
            return 0
        
            
        new_data_points = self.check_duplicates(data_points)
        
              
        inserted_count = self.store_data_to_db(new_data_points)
        
        return inserted_count
    
    def collect_all_data(self) -> Dict[str, int]:
        """采集所有区域数据"""
        logger.info("Starting full data collection...")
        start_time = time.time()
        
        results = {}
        total_inserted = 0
        
                   
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for area_id, area_info in self.monitoring_areas.items():
                future = executor.submit(self.collect_area_data, area_id, area_info)
                futures[future] = (area_id, area_info['name'])
            
                  
            for future in futures:
                area_id, area_name = futures[future]
                try:
                    count = future.result(timeout=60)         
                    results[area_name] = count
                    total_inserted += count
                    logger.info(f"{area_name}: {count} records inserted")
                except Exception as e:
                    logger.error(f"Error collecting data for {area_name}: {e}")
                    results[area_name] = 0
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Data collection completed in {duration:.2f} seconds")
        logger.info(f"Total records inserted: {total_inserted}")
        
        return results
    
    def run_scheduled_collection(self):
        """运行定时采集任务"""
        logger.info("Running scheduled data collection...")
        try:
            results = self.collect_all_data()
            
                    
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_areas': len(results),
                'total_records': sum(results.values()),
                'area_results': results
            }
            
            with open('collection_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error in scheduled collection: {e}")
    
    def start_scheduler(self):
        """启动定时调度器"""
        logger.info("Starting auto data collector scheduler...")
        
                  
        schedule.every(4).hours.do(self.run_scheduled_collection)
        
                
        self.run_scheduled_collection()
        
             
        while True:
            schedule.run_pending()
            time.sleep(60)           

def main():
    """主函数"""
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    collector = AutoDataCollector(db_url)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
                
        logger.info("Running one-time data collection...")
        results = collector.collect_all_data()
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
                
        collector.start_scheduler()

if __name__ == "__main__":
    main()
