#!/usr/bin/env python3
"""
数据采集脚本 - 从CNEMC API采集实时水质数据
"""

import sys
import os
import asyncio
import json
import argparse
from datetime import datetime
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.cnemc_collector import CNEMCCollector
from processing.etl_pipeline import WaterQualityETL
import psycopg2
from psycopg2.extras import RealDictCursor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollectionRunner:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.collector = CNEMCCollector(self.database_url)
        self.etl = WaterQualityETL()
        
    async def collect_data(self, area_ids: list, max_records: int = 1000):
        """执行数据采集"""
        logger.info(f"开始数据采集 - 区域: {area_ids}, 最大记录数: {max_records}")
        
        total_collected = 0
        errors = []
        
        for area_id in area_ids:
            try:
                logger.info(f"采集区域 {area_id} 的数据...")
                
                # 采集数据
                raw_data = await self.collector.collect_real_time_data(
                    area_id=area_id,
                    max_records=max_records // len(area_ids)
                )
                
                if raw_data:
                    logger.info(f"区域 {area_id} 采集到 {len(raw_data)} 条数据")
                    
                    # 处理数据
                    processed_data = self.etl.process_data(raw_data)
                    
                    # 存储数据
                    stored_count = await self.store_data(processed_data)
                    total_collected += stored_count
                    
                    logger.info(f"区域 {area_id} 成功存储 {stored_count} 条数据")
                else:
                    logger.warning(f"区域 {area_id} 未采集到数据")
                    
            except Exception as e:
                error_msg = f"区域 {area_id} 采集失败: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return {
            "total_collected": total_collected,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
    
    async def store_data(self, data: list):
        """存储数据到数据库"""
        if not data:
            return 0
            
        try:
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            
            stored_count = 0
            for record in data:
                try:
                    # 插入水质数据
                    insert_query = """
                    INSERT INTO water_quality_measurements (
                        station_name, station_code, province, watershed,
                        monitoring_time, temperature, ph, dissolved_oxygen,
                        conductivity, turbidity, permanganate_index,
                        ammonia_nitrogen, total_phosphorus, total_nitrogen,
                        chlorophyll_a, algae_density, water_quality_grade,
                        pollution_index, data_source, created_at, updated_at
                    ) VALUES (
                        %(station_name)s, %(station_code)s, %(province)s, %(watershed)s,
                        %(monitoring_time)s, %(temperature)s, %(ph)s, %(dissolved_oxygen)s,
                        %(conductivity)s, %(turbidity)s, %(permanganate_index)s,
                        %(ammonia_nitrogen)s, %(total_phosphorus)s, %(total_nitrogen)s,
                        %(chlorophyll_a)s, %(algae_density)s, %(water_quality_grade)s,
                        %(pollution_index)s, %(data_source)s, NOW(), NOW()
                    ) ON CONFLICT (station_name, monitoring_time) DO UPDATE SET
                        temperature = EXCLUDED.temperature,
                        ph = EXCLUDED.ph,
                        dissolved_oxygen = EXCLUDED.dissolved_oxygen,
                        conductivity = EXCLUDED.conductivity,
                        turbidity = EXCLUDED.turbidity,
                        permanganate_index = EXCLUDED.permanganate_index,
                        ammonia_nitrogen = EXCLUDED.ammonia_nitrogen,
                        total_phosphorus = EXCLUDED.total_phosphorus,
                        total_nitrogen = EXCLUDED.total_nitrogen,
                        chlorophyll_a = EXCLUDED.chlorophyll_a,
                        algae_density = EXCLUDED.algae_density,
                        water_quality_grade = EXCLUDED.water_quality_grade,
                        pollution_index = EXCLUDED.pollution_index,
                        data_source = EXCLUDED.data_source,
                        updated_at = NOW()
                    """
                    
                    cur.execute(insert_query, record)
                    stored_count += 1
                    
                except Exception as e:
                    logger.warning(f"存储记录失败: {e}")
                    continue
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"成功存储 {stored_count} 条数据")
            return stored_count
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return 0

async def main():
    parser = argparse.ArgumentParser(description='数据采集脚本')
    parser.add_argument('--area-ids', nargs='+', type=int, default=[110000],
                        help='要采集的区域ID列表')
    parser.add_argument('--max-records', type=int, default=100,
                        help='最大采集记录数')
    parser.add_argument('--database-url', 
                        default='postgres://pollution_user:pollution_pass@localhost:5432/pollution_db',
                        help='数据库连接URL')
    
    args = parser.parse_args()
    
    runner = DataCollectionRunner(args.database_url)
    
    try:
        result = await runner.collect_data(args.area_ids, args.max_records)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result["errors"]:
            logger.error(f"采集完成，但有 {len(result['errors'])} 个错误")
            sys.exit(1)
        else:
            logger.info(f"采集完成，共采集 {result['total_collected']} 条数据")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"采集失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
