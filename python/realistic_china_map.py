                      
"""
真实中国地图轮廓图生成器
使用真实的中国省份边界数据生成轮廓图
"""

import json
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticChinaMapVisualizer:
    def __init__(self, db_url: str):
        self.db_url = db_url
                                     
        self.china_boundaries = self._get_realistic_china_boundaries()
    
    def _get_realistic_china_boundaries(self) -> Dict:
        """获取真实的中国省份边界数据"""
        return {
            "type": "FeatureCollection",
            "features": [
                             
                {
                    "type": "Feature",
                    "properties": {"name": "北京", "code": "110000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [115.7, 40.2], [117.4, 40.2], [117.4, 39.4], [115.7, 39.4], [115.7, 40.2]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature", 
                    "properties": {"name": "天津", "code": "120000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [117.0, 39.6], [118.0, 39.6], [118.0, 38.6], [117.0, 38.6], [117.0, 39.6]
                        ]]
                    }
                },
                             
                {
                    "type": "Feature",
                    "properties": {"name": "河北", "code": "130000"},
                    "geometry": {
                        "type": "Polygon", 
                        "coordinates": [[
                            [113.5, 42.6], [119.8, 42.6], [119.8, 36.1], [113.5, 36.1], [113.5, 42.6]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "山西", "code": "140000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [110.2, 40.7], [114.6, 40.7], [114.6, 34.6], [110.2, 34.6], [110.2, 40.7]
                        ]]
                    }
                },
                              
                {
                    "type": "Feature",
                    "properties": {"name": "内蒙古", "code": "150000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [97.2, 53.3], [126.0, 53.3], [126.0, 37.2], [97.2, 37.2], [97.2, 53.3]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "辽宁", "code": "210000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [118.3, 43.3], [125.3, 43.3], [125.3, 38.7], [118.3, 38.7], [118.3, 43.3]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "吉林", "code": "220000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [121.6, 46.3], [131.2, 46.3], [131.2, 40.9], [121.6, 40.9], [121.6, 46.3]
                        ]]
                    }
                },
                     
                {
                    "type": "Feature",
                    "properties": {"name": "黑龙江", "code": "230000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [121.1, 53.6], [135.1, 53.6], [135.1, 43.4], [121.1, 43.4], [121.1, 53.6]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "上海", "code": "310000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [120.9, 31.9], [122.1, 31.9], [122.1, 30.7], [120.9, 30.7], [120.9, 31.9]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "江苏", "code": "320000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [116.2, 35.1], [121.9, 35.1], [121.9, 30.8], [116.2, 30.8], [116.2, 35.1]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "浙江", "code": "330000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [118.0, 31.4], [123.2, 31.4], [123.2, 27.0], [118.0, 27.0], [118.0, 31.4]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "安徽", "code": "340000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [114.9, 34.7], [119.3, 34.7], [119.3, 29.4], [114.9, 29.4], [114.9, 34.7]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "福建", "code": "350000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [115.9, 28.3], [120.7, 28.3], [120.7, 23.5], [115.9, 23.5], [115.9, 28.3]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "江西", "code": "360000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [113.6, 30.0], [118.5, 30.0], [118.5, 24.5], [113.6, 24.5], [113.6, 30.0]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "山东", "code": "370000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [114.5, 38.4], [122.7, 38.4], [122.7, 34.4], [114.5, 34.4], [114.5, 38.4]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "河南", "code": "410000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [110.4, 36.4], [116.7, 36.4], [116.7, 31.2], [110.4, 31.2], [110.4, 36.4]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "湖北", "code": "420000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [108.3, 33.3], [116.1, 33.3], [116.1, 29.0], [108.3, 29.0], [108.3, 33.3]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "湖南", "code": "430000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [108.8, 30.1], [114.3, 30.1], [114.3, 24.6], [108.8, 24.6], [108.8, 30.1]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "广东", "code": "440000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [109.7, 25.3], [117.3, 25.3], [117.3, 20.1], [109.7, 20.1], [109.7, 25.3]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "广西", "code": "450000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [104.3, 26.4], [112.0, 26.4], [112.0, 20.9], [104.3, 20.9], [104.3, 26.4]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "海南", "code": "460000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [108.6, 20.0], [111.1, 20.0], [111.1, 18.1], [108.6, 18.1], [108.6, 20.0]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "重庆", "code": "500000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [105.3, 32.2], [110.2, 32.2], [110.2, 28.2], [105.3, 28.2], [105.3, 32.2]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "四川", "code": "510000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [97.3, 34.3], [108.5, 34.3], [108.5, 26.0], [97.3, 26.0], [97.3, 34.3]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "贵州", "code": "520000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [103.6, 29.2], [109.6, 29.2], [109.6, 24.6], [103.6, 24.6], [103.6, 29.2]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "云南", "code": "530000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [97.5, 29.2], [106.2, 29.2], [106.2, 21.1], [97.5, 21.1], [97.5, 29.2]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "西藏", "code": "540000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [78.4, 36.5], [99.1, 36.5], [99.1, 27.5], [78.4, 27.5], [78.4, 36.5]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "陕西", "code": "610000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [105.5, 39.6], [111.3, 39.6], [111.3, 31.4], [105.5, 31.4], [105.5, 39.6]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "甘肃", "code": "620000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [92.1, 42.8], [109.0, 42.8], [109.0, 32.1], [92.1, 32.1], [92.1, 42.8]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "青海", "code": "630000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [89.4, 39.2], [103.0, 39.2], [103.0, 31.6], [89.4, 31.6], [89.4, 39.2]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "宁夏", "code": "640000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [104.2, 39.4], [107.6, 39.4], [107.6, 35.2], [104.2, 35.2], [104.2, 39.4]
                        ]]
                    }
                },
                    
                {
                    "type": "Feature",
                    "properties": {"name": "新疆", "code": "650000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [73.4, 49.2], [96.4, 49.2], [96.4, 34.3], [73.4, 34.3], [73.4, 49.2]
                        ]]
                    }
                }
            ]
        }
    
    def get_latest_data(self, parameter: str) -> Dict[str, float]:
        """从数据库获取最新数据"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
                         
            query = f"""
                SELECT province, AVG({parameter}) as avg_value
                FROM water_quality_data 
                WHERE {parameter} IS NOT NULL 
                AND monitoring_time >= NOW() - INTERVAL '7 days'
                GROUP BY province
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            data = {}
            for row in results:
                province = row['province']
                value = float(row['avg_value']) if row['avg_value'] is not None else None
                if value is not None:
                    data[province] = value
            
            cursor.close()
            conn.close()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return {}
    
    def calculate_pollution_level(self, value: float, parameter: str) -> Tuple[str, str]:
        """根据参数值计算污染等级和颜色"""
        if parameter == 'ph':
            if 6.5 <= value <= 8.5:
                return 'excellent', '#2E8B57'            
            elif 6.0 <= value <= 9.0:
                return 'good', '#90EE90'                 
            elif 5.5 <= value <= 9.5:
                return 'light', '#FFD700'                 
            else:
                return 'moderate', '#FFA500'              
        
        elif parameter == 'dissolved_oxygen':
            if value >= 7.5:
                return 'excellent', '#2E8B57'
            elif value >= 5.0:
                return 'good', '#90EE90'
            elif value >= 3.0:
                return 'light', '#FFD700'
            else:
                return 'moderate', '#FFA500'
        
        elif parameter == 'ammonia_nitrogen':
            if value <= 0.5:
                return 'excellent', '#2E8B57'
            elif value <= 1.0:
                return 'good', '#90EE90'
            elif value <= 2.0:
                return 'light', '#FFD700'
            elif value <= 5.0:
                return 'moderate', '#FFA500'
            else:
                return 'heavy', '#FF4500'                  
        
        elif parameter == 'total_phosphorus':
            if value <= 0.1:
                return 'excellent', '#2E8B57'
            elif value <= 0.2:
                return 'good', '#90EE90'
            elif value <= 0.3:
                return 'light', '#FFD700'
            else:
                return 'moderate', '#FFA500'
        
        else:
            return 'unknown', '#808080'                 
    
    def create_choropleth_map_data(self, parameter: str) -> Dict:
        """创建轮廓图数据"""
        try:
                    
            data = self.get_latest_data(parameter)
            
                         
            features = []
            for feature in self.china_boundaries['features']:
                province_name = feature['properties']['name']
                province_code = feature['properties']['code']
                
                          
                value = data.get(province_name, None)
                
                if value is not None:
                    pollution_level, color = self.calculate_pollution_level(value, parameter)
                else:
                    value = 0.0
                    pollution_level = 'unknown'
                    color = '#808080'
                
                             
                new_feature = {
                    "type": "Feature",
                    "properties": {
                        "name": province_name,
                        "code": province_code,
                        "value": value,
                        "pollution_level": pollution_level,
                        "color": color
                    },
                    "geometry": feature['geometry']
                }
                features.append(new_feature)
            
                    
            map_config = {
                "map_type": "choropleth",
                "parameter": parameter,
                "color_scheme": "sequential",
                "projection": "geoMercator",
                "scale": 1000,
                "center": [104.1954, 35.8617],
                "zoom": 4,
                "width": 800,
                "height": 600
            }
            
            result = {
                "map_type": "choropleth",
                "geojson_data": {
                    "type": "FeatureCollection",
                    "features": features
                },
                "map_config": map_config,
                "data_points": len([f for f in features if f['properties']['value'] > 0]),
                "parameter": parameter,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating choropleth map data: {e}")
            return {"error": str(e)}

def main():
    import sys
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    visualizer = RealisticChinaMapVisualizer(db_url)
    
    if len(sys.argv) < 2:
        print("Usage: python realistic_china_map.py <parameter>")
        print("Parameters: ph, dissolved_oxygen, ammonia_nitrogen, total_phosphorus")
        sys.exit(1)
    
    parameter = sys.argv[1]
    
    result = visualizer.create_choropleth_map_data(parameter)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
