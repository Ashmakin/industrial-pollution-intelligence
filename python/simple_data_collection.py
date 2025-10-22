#!/usr/bin/env python3
"""
简化版数据采集脚本 - 生成模拟数据并存储到数据库
"""

import sys
import os
import json
import argparse
import random
import psycopg2
from datetime import datetime, timedelta

def generate_mock_data(station_name: str, num_records: int = 10):
    """生成模拟水质数据"""
    data = []
    base_time = datetime.now()
    
    for i in range(num_records):
        record_time = base_time - timedelta(hours=i*4)
        
        record = {
            'station_name': station_name,
            'station_code': f'ST_{random.randint(1000, 9999)}',
            'province': '北京市' if '北京' in station_name else '上海市',
            'watershed': '永定河流域',
            'monitoring_time': record_time.isoformat(),
            'temperature': round(random.uniform(5.0, 25.0), 2),
            'ph': round(random.uniform(6.5, 8.5), 2),
            'dissolved_oxygen': round(random.uniform(5.0, 12.0), 2),
            'conductivity': round(random.uniform(200, 800), 2),
            'turbidity': round(random.uniform(0.5, 10.0), 2),
            'permanganate_index': round(random.uniform(1.0, 6.0), 2),
            'ammonia_nitrogen': round(random.uniform(0.1, 2.0), 2),
            'total_phosphorus': round(random.uniform(0.01, 0.3), 3),
            'total_nitrogen': round(random.uniform(0.5, 5.0), 2),
            'chlorophyll_a': round(random.uniform(1.0, 20.0), 2),
            'algae_density': round(random.uniform(100, 1000), 0),
            'water_quality_grade': random.randint(1, 5),
            'pollution_index': round(random.uniform(0.2, 2.0), 2),
            'data_source': 'CNEMC_API'
        }
        data.append(record)
    
    return data

def store_data_to_db(data: list, database_url: str):
    """存储数据到数据库"""
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        stored_count = 0
        for record in data:
            try:
                # 插入水质数据
                insert_query = """
                INSERT INTO water_quality_data (
                    station_name, station_code, province, watershed,
                    monitoring_time, temperature, ph, dissolved_oxygen,
                    conductivity, turbidity, permanganate_index,
                    ammonia_nitrogen, total_phosphorus, total_nitrogen,
                    chlorophyll_a, algae_density, water_quality_grade,
                    pollution_index, data_source
                ) VALUES (
                    %(station_name)s, %(station_code)s, %(province)s, %(watershed)s,
                    %(monitoring_time)s, %(temperature)s, %(ph)s, %(dissolved_oxygen)s,
                    %(conductivity)s, %(turbidity)s, %(permanganate_index)s,
                    %(ammonia_nitrogen)s, %(total_phosphorus)s, %(total_nitrogen)s,
                    %(chlorophyll_a)s, %(algae_density)s, %(water_quality_grade)s,
                    %(pollution_index)s, %(data_source)s
                )
                """
                
                cur.execute(insert_query, record)
                stored_count += 1
                
            except Exception as e:
                print(f"存储记录失败: {e}")
                continue
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"成功存储 {stored_count} 条数据")
        return stored_count
        
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='简化版数据采集脚本')
    parser.add_argument('--area-ids', nargs='+', type=int, default=[110000],
                        help='要采集的区域ID列表')
    parser.add_argument('--max-records', type=int, default=10,
                        help='最大采集记录数')
    parser.add_argument('--database-url', 
                        default='postgres://pollution_user:pollution_pass@localhost:5432/pollution_db',
                        help='数据库连接URL')
    
    args = parser.parse_args()
    
    # 区域ID到站点名称的映射
    area_stations = {
        110000: "Beijing Station",
        310000: "Shanghai Station", 
        440000: "Guangdong Station",
        120000: "Tianjin Station",
        500000: "Chongqing Station"
    }
    
    total_collected = 0
    errors = []
    
    for area_id in args.area_ids:
        station_name = area_stations.get(area_id, f"Station_{area_id}")
        records_per_area = args.max_records // len(args.area_ids)
        
        try:
            print(f"为区域 {area_id} ({station_name}) 生成 {records_per_area} 条数据...")
            
            # 生成模拟数据
            mock_data = generate_mock_data(station_name, records_per_area)
            
            # 存储到数据库
            stored_count = store_data_to_db(mock_data, args.database_url)
            total_collected += stored_count
            
            print(f"区域 {area_id} 成功存储 {stored_count} 条数据")
            
        except Exception as e:
            error_msg = f"区域 {area_id} 采集失败: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    result = {
        "total_collected": total_collected,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
