                      
"""
真实的CNEMC API数据采集脚本
使用您提供的API接口采集真实的水质监测数据
"""

import sys
import os
import json
import argparse
import requests
from datetime import datetime
import psycopg2
from bs4 import BeautifulSoup
import time

def fetch_cnemc_data(area_id: str, max_records: int = 60):
    """从CNEMC API获取真实水质数据"""
    print(f"正在从CNEMC API获取区域 {area_id} 的数据...")
    
                 
    url = "https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx"
    
          
    payload = {
        "AreaID": area_id,
        "RiverID": "",
        "MNName": "",
        "PageIndex": "-1",
        "PageSize": str(max_records),
        "action": "getRealDatas"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://szzdjc.cnemc.cn:8070/GJZ/Business/Publish/RealDatas.html",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://szzdjc.cnemc.cn:8070",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0",
        "Connection": "keep-alive"
    }
    
    try:
                  
        response = requests.post(url, data=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"成功获取API响应，状态码: {response.status_code}")
            return parse_cnemc_response(data, area_id)
        else:
            print(f"API请求失败，状态码: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"请求CNEMC API时发生错误: {e}")
        return []

def parse_cnemc_response(data: dict, area_id: str):
    """解析CNEMC API响应数据"""
    records = []
    
            
    area_names = {
        "110000": "北京市",
        "310000": "上海市", 
        "440000": "广东省",
        "120000": "天津市",
        "500000": "重庆市"
    }
    
    area_name = area_names.get(area_id, f"区域{area_id}")
    
    try:
                                 
        if 'tbody' in data and isinstance(data['tbody'], list):
            print(f"找到 {len(data['tbody'])} 条数据记录")
            
            for row in data['tbody']:
                try:
                    if len(row) >= 17:           
                                        
                        station_html = row[2] if len(row) > 2 else ""
                        station_name = extract_text_from_html(station_html)
                        
                                
                        monitor_time_str = row[3] if len(row) > 3 else ""
                        monitor_time = parse_monitoring_time(monitor_time_str)
                        
                              
                        water_quality_grade_str = row[4] if len(row) > 4 else ""
                        water_quality_grade = parse_water_quality_grade(water_quality_grade_str)
                        
                              
                        record = {
                            'station_name': station_name,
                            'station_code': f'CNEMC_{hash(station_name) % 10000:04d}',
                            'province': row[0] if len(row) > 0 else area_name,
                            'watershed': row[1] if len(row) > 1 else '',
                            'monitoring_time': monitor_time,
                            'temperature': parse_numeric_from_html(row[5] if len(row) > 5 else ""),
                            'ph': parse_numeric_from_html(row[6] if len(row) > 6 else ""),
                            'dissolved_oxygen': parse_numeric_from_html(row[7] if len(row) > 7 else ""),
                            'conductivity': parse_numeric_from_html(row[8] if len(row) > 8 else ""),
                            'turbidity': parse_numeric_from_html(row[9] if len(row) > 9 else ""),
                            'permanganate_index': parse_numeric_from_html(row[10] if len(row) > 10 else ""),
                            'ammonia_nitrogen': parse_numeric_from_html(row[11] if len(row) > 11 else ""),
                            'total_phosphorus': parse_numeric_from_html(row[12] if len(row) > 12 else ""),
                            'total_nitrogen': parse_numeric_from_html(row[13] if len(row) > 13 else ""),
                            'chlorophyll_a': parse_numeric_from_html(row[14] if len(row) > 14 else ""),
                            'algae_density': parse_numeric_from_html(row[15] if len(row) > 15 else ""),
                            'water_quality_grade': water_quality_grade,
                            'pollution_index': None,                 
                            'data_source': 'CNEMC_API'
                        }
                        records.append(record)
                        
                except Exception as e:
                    print(f"解析单条记录时出错: {e}")
                    continue
        
        print(f"成功解析 {len(records)} 条记录")
        return records
        
    except Exception as e:
        print(f"解析API响应时出错: {e}")
        return []

def parse_numeric_value(value):
    """解析数值，处理空值和字符串"""
    if value is None or value == '' or value == '-':
        return None
    
    try:
                       
        if isinstance(value, str):
                           
            value = BeautifulSoup(value, 'html.parser').get_text().strip()
            if value == '' or value == '-':
                return None
        
        return float(value)
    except (ValueError, TypeError):
        return None

def parse_numeric_from_html(html_value):
    """从HTML标签中解析数值"""
    if not html_value or html_value == '*' or html_value == '-':
        return None
    
    try:
                               
        soup = BeautifulSoup(html_value, 'html.parser')
        text = soup.get_text().strip()
        
        if text == '' or text == '*' or text == '-':
            return None
            
        return float(text)
    except (ValueError, TypeError):
        return None

def extract_text_from_html(html_value):
    """从HTML中提取文本内容"""
    if not html_value:
        return "Unknown Station"
    
    try:
        soup = BeautifulSoup(html_value, 'html.parser')
        text = soup.get_text().strip()
        return text if text else "Unknown Station"
    except:
        return "Unknown Station"

def parse_monitoring_time(time_str):
    """解析监测时间字符串"""
    if not time_str:
        return datetime.now().isoformat()
    
    try:
                           
        current_year = datetime.now().year
        full_time_str = f"{current_year}-{time_str}"
        parsed_time = datetime.strptime(full_time_str, "%Y-%m-%d %H:%M")
        return parsed_time.isoformat()
    except:
        return datetime.now().isoformat()

def parse_water_quality_grade(grade_str):
    """解析水质等级"""
    if not grade_str:
        return None
    
            
    grade_map = {
        "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5,
        "劣Ⅴ": 6
    }
    
    return grade_map.get(grade_str.strip(), None)

def store_data_to_db(data: list, database_url: str):
    """存储数据到数据库"""
    if not data:
        print("没有数据需要存储")
        return 0
        
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        stored_count = 0
        for record in data:
            try:
                        
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
        
        print(f"成功存储 {stored_count} 条数据到数据库")
        return stored_count
        
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='真实CNEMC API数据采集脚本')
    parser.add_argument('--area-ids', nargs='+', type=str, default=['110000'],
                        help='要采集的区域ID列表')
    parser.add_argument('--max-records', type=int, default=60,
                        help='最大采集记录数')
    parser.add_argument('--database-url', 
                        default='postgres://pollution_user:pollution_pass@localhost:5432/pollution_db',
                        help='数据库连接URL')
    
    args = parser.parse_args()
    
    total_collected = 0
    errors = []
    
    for area_id in args.area_ids:
        try:
            print(f"\n开始采集区域 {area_id} 的数据...")
            
                            
            api_data = fetch_cnemc_data(area_id, args.max_records)
            
            if api_data:
                        
                stored_count = store_data_to_db(api_data, args.database_url)
                total_collected += stored_count
                print(f"区域 {area_id} 成功存储 {stored_count} 条数据")
            else:
                print(f"区域 {area_id} 未获取到数据")
                
                          
            time.sleep(2)
            
        except Exception as e:
            error_msg = f"区域 {area_id} 采集失败: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    result = {
        "total_collected": total_collected,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
        "source": "CNEMC_API"
    }
    
    print(f"\n采集完成:")
    print(f"总计采集: {total_collected} 条数据")
    print(f"错误数量: {len(errors)}")
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
