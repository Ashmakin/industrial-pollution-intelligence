"""
CNEMC Surface Water Quality Data Collector

Collects real-time surface water quality data from China National Environmental
Monitoring Center (CNEMC) API and stores in PostgreSQL database.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from sqlalchemy import create_engine, text
import json
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WaterQualityRecord:
    """Data structure for water quality measurements"""
    province: str
    watershed: str
    station_name: str
    monitoring_time: datetime
    water_quality_grade: str
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
    area_id: str

class CNEMCCollector:
    """CNEMC API data collector with multi-province support"""
    
    BASE_URL = "https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx"
    
    # Provincial Area IDs (first 2 digits + 0000)
    PROVINCE_CODES = {
        "110000": "北京市",
        "120000": "天津市", 
        "130000": "河北省",
        "140000": "山西省",
        "150000": "内蒙古自治区",
        "210000": "辽宁省",
        "220000": "吉林省",
        "230000": "黑龙江省",
        "310000": "上海市",
        "320000": "江苏省",
        "330000": "浙江省",
        "340000": "安徽省",
        "350000": "福建省",
        "360000": "江西省",
        "370000": "山东省",
        "410000": "河南省",
        "420000": "湖北省",
        "430000": "湖南省",
        "440000": "广东省",
        "450000": "广西壮族自治区",
        "460000": "海南省",
        "500000": "重庆市",
        "TC0000": "四川省",
        "520000": "贵州省",
        "530000": "云南省",
        "540000": "西藏自治区",
        "610000": "陕西省",
        "620000": "甘肃省",
        "630000": "青海省",
        "640000": "宁夏回族自治区",
        "650000": "新疆维吾尔自治区"
    }
    
    def __init__(self, db_url: str, batch_size: int = 100):
        self.db_url = db_url
        self.batch_size = batch_size
        self.engine = create_engine(db_url)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _parse_measurement_value(self, value_str: str) -> Optional[float]:
        """Parse measurement value from HTML span elements"""
        if not value_str or value_str in ['*', '-', '']:
            return None
            
        try:
            # Extract numeric value from HTML spans
            if '<span' in value_str:
                # Extract tooltip value if available
                if '原始值：' in value_str:
                    start = value_str.find('原始值：') + 4
                    end = value_str.find('"', start)
                    if end == -1:
                        end = value_str.find('>', start)
                    raw_value = value_str[start:end]
                    return float(raw_value)
                else:
                    # Extract displayed value
                    start = value_str.rfind('>') + 1
                    end = value_str.rfind('<')
                    if start < end:
                        display_value = value_str[start:end]
                        return float(display_value)
            else:
                return float(value_str)
        except (ValueError, IndexError):
            return None
    
    async def fetch_province_data(self, area_id: str, page_size: int = 60) -> List[WaterQualityRecord]:
        """Fetch water quality data for a specific province"""
        payload = {
            "AreaID": area_id,
            "RiverID": "",
            "MNName": "",
            "PageIndex": "-1",
            "PageSize": str(page_size),
            "action": "getRealDatas"
        }
        
        try:
            async with self.session.post(
                self.BASE_URL,
                data=payload,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_response(data, area_id)
                else:
                    logger.error(f"HTTP {response.status} for area {area_id}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching data for area {area_id}: {e}")
            return []
    
    def _parse_response(self, data: dict, area_id: str) -> List[WaterQualityRecord]:
        """Parse CNEMC API response into WaterQualityRecord objects"""
        records = []
        
        if data.get("result") != 1 or not data.get("tbody"):
            return records
            
        headers = data.get("thead", [])
        rows = data.get("tbody", [])
        
        # Map column indices
        col_map = {}
        for i, header in enumerate(headers):
            if "省份" in header:
                col_map["province"] = i
            elif "流域" in header:
                col_map["watershed"] = i
            elif "断面名称" in header:
                col_map["station_name"] = i
            elif "监测时间" in header:
                col_map["monitoring_time"] = i
            elif "水质类别" in header:
                col_map["water_quality_grade"] = i
            elif "水温" in header:
                col_map["temperature"] = i
            elif "pH" in header:
                col_map["ph"] = i
            elif "溶解氧" in header:
                col_map["dissolved_oxygen"] = i
            elif "电导率" in header:
                col_map["conductivity"] = i
            elif "浊度" in header:
                col_map["turbidity"] = i
            elif "高锰酸盐指数" in header:
                col_map["permanganate_index"] = i
            elif "氨氮" in header:
                col_map["ammonia_nitrogen"] = i
            elif "总磷" in header:
                col_map["total_phosphorus"] = i
            elif "总氮" in header:
                col_map["total_nitrogen"] = i
            elif "叶绿素α" in header:
                col_map["chlorophyll_a"] = i
            elif "藻密度" in header:
                col_map["algae_density"] = i
            elif "站点情况" in header:
                col_map["station_status"] = i
        
        for row in rows:
            try:
                # Extract station name from HTML
                station_name = row[col_map["station_name"]]
                if '<span' in station_name:
                    start = station_name.rfind('>') + 1
                    end = station_name.rfind('<')
                    if start < end:
                        station_name = station_name[start:end]
                
                # Parse monitoring time
                time_str = row[col_map["monitoring_time"]]
                try:
                    monitoring_time = datetime.strptime(f"2024-{time_str}", "%Y-%m-%d %H:%M")
                except ValueError:
                    monitoring_time = datetime.now()
                
                record = WaterQualityRecord(
                    province=row[col_map["province"]],
                    watershed=row[col_map["watershed"]],
                    station_name=station_name,
                    monitoring_time=monitoring_time,
                    water_quality_grade=row[col_map["water_quality_grade"]],
                    temperature=self._parse_measurement_value(row[col_map["temperature"]]),
                    ph=self._parse_measurement_value(row[col_map["ph"]]),
                    dissolved_oxygen=self._parse_measurement_value(row[col_map["dissolved_oxygen"]]),
                    conductivity=self._parse_measurement_value(row[col_map["conductivity"]]),
                    turbidity=self._parse_measurement_value(row[col_map["turbidity"]]),
                    permanganate_index=self._parse_measurement_value(row[col_map["permanganate_index"]]),
                    ammonia_nitrogen=self._parse_measurement_value(row[col_map["ammonia_nitrogen"]]),
                    total_phosphorus=self._parse_measurement_value(row[col_map["total_phosphorus"]]),
                    total_nitrogen=self._parse_measurement_value(row[col_map["total_nitrogen"]]),
                    chlorophyll_a=self._parse_measurement_value(row[col_map["chlorophyll_a"]]),
                    algae_density=self._parse_measurement_value(row[col_map["algae_density"]]),
                    station_status=row[col_map["station_status"]],
                    area_id=area_id
                )
                records.append(record)
                
            except (IndexError, KeyError) as e:
                logger.warning(f"Error parsing row: {e}")
                continue
                
        return records
    
    async def collect_all_provinces(self) -> List[WaterQualityRecord]:
        """Collect data from all provinces concurrently"""
        tasks = []
        for area_id in self.PROVINCE_CODES.keys():
            task = self.fetch_province_data(area_id)
            tasks.append(task)
        
        # Add delay to avoid overwhelming the API
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            province_data = await task
            results.extend(province_data)
            
            # Rate limiting
            if i % 5 == 0:
                await asyncio.sleep(1)
                
        return results
    
    def create_database_schema(self):
        """Create database tables for water quality data"""
        schema_sql = """
        CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
        
        CREATE TABLE IF NOT EXISTS water_quality_stations (
            id SERIAL PRIMARY KEY,
            station_name VARCHAR(255) NOT NULL,
            province VARCHAR(100) NOT NULL,
            watershed VARCHAR(100) NOT NULL,
            area_id VARCHAR(10) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(station_name, province)
        );
        
        CREATE TABLE IF NOT EXISTS water_quality_measurements (
            id SERIAL PRIMARY KEY,
            station_id INTEGER REFERENCES water_quality_stations(id),
            monitoring_time TIMESTAMP NOT NULL,
            water_quality_grade VARCHAR(10),
            temperature DECIMAL(8,3),
            ph DECIMAL(8,3),
            dissolved_oxygen DECIMAL(8,3),
            conductivity DECIMAL(8,3),
            turbidity DECIMAL(8,3),
            permanganate_index DECIMAL(8,3),
            ammonia_nitrogen DECIMAL(8,3),
            total_phosphorus DECIMAL(8,3),
            total_nitrogen DECIMAL(8,3),
            chlorophyll_a DECIMAL(8,3),
            algae_density DECIMAL(12,3),
            station_status VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(station_id, monitoring_time)
        );
        
        -- Convert to hypertable for time-series optimization
        SELECT create_hypertable('water_quality_measurements', 'monitoring_time', 
                                if_not_exists => TRUE);
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_measurements_time ON water_quality_measurements(monitoring_time);
        CREATE INDEX IF NOT EXISTS idx_measurements_station ON water_quality_measurements(station_id);
        CREATE INDEX IF NOT EXISTS idx_stations_area ON water_quality_stations(area_id);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
    
    def store_records(self, records: List[WaterQualityRecord]):
        """Store water quality records in database"""
        if not records:
            return
            
        stations_data = []
        measurements_data = []
        
        # Prepare station data
        station_map = {}
        for record in records:
            station_key = (record.station_name, record.province)
            if station_key not in station_map:
                stations_data.append({
                    'station_name': record.station_name,
                    'province': record.province,
                    'watershed': record.watershed,
                    'area_id': record.area_id
                })
                station_map[station_key] = len(stations_data)
        
        # Store stations
        if stations_data:
            stations_df = pd.DataFrame(stations_data)
            stations_df.to_sql('water_quality_stations', self.engine, 
                             if_exists='append', index=False, method='multi')
        
        # Get station IDs
        station_query = """
        SELECT id, station_name, province 
        FROM water_quality_stations 
        WHERE (station_name, province) = ANY(%s)
        """
        station_keys = list(station_map.keys())
        station_df = pd.read_sql(station_query, self.engine, params=[station_keys])
        station_id_map = dict(zip(
            zip(station_df['station_name'], station_df['province']),
            station_df['id']
        ))
        
        # Prepare measurement data
        for record in records:
            station_key = (record.station_name, record.province)
            station_id = station_id_map.get(station_key)
            if station_id:
                measurements_data.append({
                    'station_id': station_id,
                    'monitoring_time': record.monitoring_time,
                    'water_quality_grade': record.water_quality_grade,
                    'temperature': record.temperature,
                    'ph': record.ph,
                    'dissolved_oxygen': record.dissolved_oxygen,
                    'conductivity': record.conductivity,
                    'turbidity': record.turbidity,
                    'permanganate_index': record.permanganate_index,
                    'ammonia_nitrogen': record.ammonia_nitrogen,
                    'total_phosphorus': record.total_phosphorus,
                    'total_nitrogen': record.total_nitrogen,
                    'chlorophyll_a': record.chlorophyll_a,
                    'algae_density': record.algae_density,
                    'station_status': record.station_status
                })
        
        # Store measurements
        if measurements_data:
            measurements_df = pd.DataFrame(measurements_data)
            measurements_df.to_sql('water_quality_measurements', self.engine,
                                 if_exists='append', index=False, method='multi')

async def main():
    """Main collection function"""
    # Database connection (adjust as needed)
    DB_URL = "postgresql://user:password@localhost:5432/pollution_db"
    
    collector = CNEMCCollector(DB_URL)
    collector.create_database_schema()
    
    async with collector as c:
        logger.info("Starting data collection from CNEMC API...")
        records = await c.collect_all_provinces()
        logger.info(f"Collected {len(records)} water quality records")
        
        # Store in database
        c.store_records(records)
        logger.info("Data stored successfully")

if __name__ == "__main__":
    asyncio.run(main())

