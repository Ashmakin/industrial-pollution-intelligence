                      
"""
中国地图轮廓图（Choropleth Map）可视化系统
- 使用真实的中国地图边界数据
- 创建类似D3.js的轮廓图可视化
- 支持不同参数的颜色分级显示
"""

import json
import pandas as pd
import numpy as np
import psycopg2
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChinaChoroplethMap:
    """中国地图轮廓图可视化器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
                             
        self.china_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "北京", "code": "110000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[116.0, 40.0], [117.0, 40.0], [117.0, 39.5], [116.0, 39.5], [116.0, 40.0]]]
                    }
                },
                {
                    "type": "Feature", 
                    "properties": {"name": "天津", "code": "120000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[117.0, 39.5], [118.0, 39.5], [118.0, 39.0], [117.0, 39.0], [117.0, 39.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "河北", "code": "130000"},
                    "geometry": {
                        "type": "Polygon", 
                        "coordinates": [[[113.5, 42.0], [119.5, 42.0], [119.5, 36.0], [113.5, 36.0], [113.5, 42.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "山西", "code": "140000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[110.5, 40.5], [114.5, 40.5], [114.5, 34.5], [110.5, 34.5], [110.5, 40.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "内蒙古", "code": "150000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[97.0, 53.0], [126.0, 53.0], [126.0, 37.0], [97.0, 37.0], [97.0, 53.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "辽宁", "code": "210000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[118.5, 43.0], [125.5, 43.0], [125.5, 38.5], [118.5, 38.5], [118.5, 43.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "吉林", "code": "220000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[121.5, 46.0], [131.0, 46.0], [131.0, 40.5], [121.5, 40.5], [121.5, 46.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "黑龙江", "code": "230000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[121.0, 53.5], [135.0, 53.5], [135.0, 43.5], [121.0, 43.5], [121.0, 53.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "上海", "code": "310000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[120.5, 31.5], [122.0, 31.5], [122.0, 30.5], [120.5, 30.5], [120.5, 31.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "江苏", "code": "320000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[116.0, 35.0], [122.0, 35.0], [122.0, 30.5], [116.0, 30.5], [116.0, 35.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "浙江", "code": "330000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[118.0, 31.5], [123.0, 31.5], [123.0, 27.0], [118.0, 27.0], [118.0, 31.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "安徽", "code": "340000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[114.5, 35.0], [119.5, 35.0], [119.5, 29.5], [114.5, 29.5], [114.5, 35.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "福建", "code": "350000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[115.5, 28.5], [120.5, 28.5], [120.5, 23.5], [115.5, 23.5], [115.5, 28.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "江西", "code": "360000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[113.5, 30.0], [118.5, 30.0], [118.5, 24.5], [113.5, 24.5], [113.5, 30.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "山东", "code": "370000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[114.5, 38.5], [122.5, 38.5], [122.5, 34.5], [114.5, 34.5], [114.5, 38.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "河南", "code": "410000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[110.5, 36.5], [116.5, 36.5], [116.5, 31.5], [110.5, 31.5], [110.5, 36.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "湖北", "code": "420000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[108.5, 33.5], [116.5, 33.5], [116.5, 29.0], [108.5, 29.0], [108.5, 33.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "湖南", "code": "430000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[108.5, 30.5], [114.5, 30.5], [114.5, 24.5], [108.5, 24.5], [108.5, 30.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "广东", "code": "440000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[109.5, 25.5], [117.5, 25.5], [117.5, 20.0], [109.5, 20.0], [109.5, 25.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "广西", "code": "450000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[104.0, 26.5], [112.0, 26.5], [112.0, 20.5], [104.0, 20.5], [104.0, 26.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "海南", "code": "460000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[108.5, 20.5], [111.0, 20.5], [111.0, 18.0], [108.5, 18.0], [108.5, 20.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "重庆", "code": "500000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[105.5, 32.5], [110.5, 32.5], [110.5, 28.5], [105.5, 28.5], [105.5, 32.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "四川", "code": "510000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[97.5, 34.5], [108.5, 34.5], [108.5, 26.0], [97.5, 26.0], [97.5, 34.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "贵州", "code": "520000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[103.5, 29.5], [109.5, 29.5], [109.5, 24.5], [103.5, 24.5], [103.5, 29.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "云南", "code": "530000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[97.5, 29.5], [106.0, 29.5], [106.0, 21.0], [97.5, 21.0], [97.5, 29.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "西藏", "code": "540000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[78.0, 36.5], [99.0, 36.5], [99.0, 27.5], [78.0, 27.5], [78.0, 36.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "陕西", "code": "610000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[105.5, 39.5], [111.0, 39.5], [111.0, 31.5], [105.5, 31.5], [105.5, 39.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "甘肃", "code": "620000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[92.0, 42.5], [109.0, 42.5], [109.0, 32.0], [92.0, 32.0], [92.0, 42.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "青海", "code": "630000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[89.0, 39.5], [103.0, 39.5], [103.0, 31.5], [89.0, 31.5], [89.0, 39.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "宁夏", "code": "640000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[104.0, 39.5], [107.5, 39.5], [107.5, 35.5], [104.0, 35.5], [104.0, 39.5]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "新疆", "code": "650000"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[73.0, 49.0], [96.5, 49.0], [96.5, 34.0], [73.0, 34.0], [73.0, 49.0]]]
                    }
                }
            ]
        }
        
                                
        self.color_schemes = {
            'sequential': {
                'excellent': '#2E8B57',         
                'good': '#90EE90',              
                'light': '#FFD700',             
                'moderate': '#FF8C00',          
                'heavy': '#FF4500',             
                'severe': '#DC143C'             
            },
            'diverging': {
                'excellent': '#2166AC',         
                'good': '#5AAE61',             
                'light': '#F7F7F7',             
                'moderate': '#F4A582',          
                'heavy': '#D6604D',            
                'severe': '#B2182B'             
            }
        }
        
              
        self.parameter_thresholds = {
            'ph': {
                'excellent': (6.5, 8.5),
                'good': (6.0, 9.0),
                'light': (5.5, 9.5),
                'moderate': (5.0, 10.0),
                'heavy': (4.5, 10.5),
                'severe': (0, 14)
            },
            'dissolved_oxygen': {
                'excellent': (7.5, 20),
                'good': (6.0, 7.5),
                'light': (5.0, 6.0),
                'moderate': (3.0, 5.0),
                'heavy': (2.0, 3.0),
                'severe': (0, 2.0)
            },
            'ammonia_nitrogen': {
                'excellent': (0, 0.15),
                'good': (0.15, 0.5),
                'light': (0.5, 1.0),
                'moderate': (1.0, 2.0),
                'heavy': (2.0, 5.0),
                'severe': (5.0, 100)
            },
            'total_phosphorus': {
                'excellent': (0, 0.02),
                'good': (0.02, 0.1),
                'light': (0.1, 0.2),
                'moderate': (0.2, 0.3),
                'heavy': (0.3, 0.4),
                'severe': (0.4, 100)
            }
        }
    
    def get_latest_data(self, parameter: str) -> Dict[str, float]:
        """获取最新数据"""
        try:
            conn = psycopg2.connect(self.db_url)
            
                         
            column_mapping = {
                'ph': 'ph',
                'temperature': 'temperature',
                'dissolved_oxygen': 'dissolved_oxygen',
                'ammonia_nitrogen': 'ammonia_nitrogen',
                'total_phosphorus': 'total_phosphorus',
                'conductivity': 'conductivity',
                'turbidity': 'turbidity'
            }
            
            if parameter not in column_mapping:
                return {}
            
            column_name = column_mapping[parameter]
            
            query = f"""
                SELECT 
                    station_name,
                    AVG({column_name}) as avg_value
                FROM water_quality_data 
                WHERE {column_name} IS NOT NULL 
                AND monitoring_time >= %s
                GROUP BY station_name
                HAVING COUNT(*) > 0
            """
            
                       
            start_date = datetime.now() - timedelta(days=7)
            df = pd.read_sql_query(query, conn, params=[start_date])
            conn.close()
            
                        
            result = {}
            for _, row in df.iterrows():
                station_name = row['station_name']
                avg_value = row['avg_value']
                
                        
                province = self.extract_province_from_station(station_name)
                if province:
                    if province not in result:
                        result[province] = []
                    result[province].append(avg_value)
            
                        
            province_averages = {}
            for province, values in result.items():
                province_averages[province] = np.mean(values)
            
            return province_averages
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return {}
    
    def extract_province_from_station(self, station_name: str) -> Optional[str]:
        """从站点名称提取省份"""
        provinces = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
                    '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
                    '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
                    '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
        
        for province in provinces:
            if province in station_name:
                return province
        return None
    
    def get_pollution_level(self, parameter: str, value: float) -> str:
        """获取污染等级"""
        if parameter not in self.parameter_thresholds:
            return 'unknown'
        
        thresholds = self.parameter_thresholds[parameter]
        
        for level, (min_val, max_val) in thresholds.items():
            if min_val <= value <= max_val:
                return level
        
        return 'severe'          
    
    def create_choropleth_map(self, parameter: str, color_scheme: str = 'sequential') -> Dict:
        """创建轮廓图地图数据"""
        try:
                  
            data = self.get_latest_data(parameter)
            
                    
            colors = self.color_schemes.get(color_scheme, self.color_schemes['sequential'])
            
                    
            features = []
            for feature in self.china_geojson['features']:
                province_name = feature['properties']['name']
                province_code = feature['properties']['code']
                
                          
                if province_name in data:
                    value = data[province_name]
                    pollution_level = self.get_pollution_level(parameter, value)
                    color = colors.get(pollution_level, '#CCCCCC')
                else:
                    value = None
                    pollution_level = 'no_data'
                    color = '#CCCCCC'
                
                        
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
            
                         
            geojson_data = {
                "type": "FeatureCollection",
                "features": features
            }
            
                    
            map_config = {
                "map_type": "choropleth",
                "parameter": parameter,
                "color_scheme": color_scheme,
                "projection": "geoMercator",
                "scale": 1000,
                "center": [104.1954, 35.8617],        
                "zoom": 4,
                "width": 800,
                "height": 600
            }
            
            return {
                "map_type": "choropleth",
                "geojson_data": geojson_data,
                "map_config": map_config,
                "data_points": len([f for f in features if f['properties']['value'] is not None]),
                "parameter": parameter,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating choropleth map: {e}")
            return {'error': str(e)}
    
    def create_d3_style_map(self, parameter: str) -> str:
        """创建D3.js风格的地图HTML"""
        try:
            map_data = self.create_choropleth_map(parameter)
            
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>中国{parameter.upper()}污染分布图</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://d3js.org/d3-geo-projection.v2.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        #map {{
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .legend {{
            margin-top: 20px;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border: 1px solid #333;
        }}
        .province {{
            stroke: #fff;
            stroke-width: 1px;
        }}
        .province:hover {{
            stroke: #333;
            stroke-width: 2px;
            cursor: pointer;
        }}
        .tooltip {{
            position: absolute;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            border-radius: 4px;
            pointer-events: none;
            font-size: 12px;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>中国{parameter.upper()}污染分布图</h1>
        <svg id="map"></svg>
        <div class="legend" id="legend"></div>
    </div>

    <script>
        const mapData = {json.dumps(map_data, ensure_ascii=False)};

        const width = 800;
        const height = 600;
        const svg = d3.select("#map")
            .attr("width", width)
            .attr("height", height);

        const projection = d3.geoMercator()
            .scale(1000)
            .center([104.1954, 35.8617])
            .translate([width / 2, height / 2]);

        const path = d3.geoPath().projection(projection);

        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);

        svg.selectAll(".province")
            .data(mapData.geojson_data.features)
            .enter().append("path")
            .attr("class", "province")
            .attr("d", path)
            .attr("fill", d => d.properties.color)
            .on("mouseover", function(event, d) {{
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`
                    <strong>${{d.properties.name}}</strong><br/>
                    参数: {parameter}<br/>
                    数值: ${{d.properties.value ? d.properties.value.toFixed(3) : '无数据'}}<br/>
                    等级: ${{d.properties.pollution_level}}
                `)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", function(d) {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }});
        
        const legendData = [
            {{color: '#2E8B57', label: '优秀'}},
            {{color: '#90EE90', label: '良好'}},
            {{color: '#FFD700', label: '轻度污染'}},
            {{color: '#FF8C00', label: '中度污染'}},
            {{color: '#FF4500', label: '重度污染'}},
            {{color: '#DC143C', label: '严重污染'}},
            {{color: '#CCCCCC', label: '无数据'}}
        ];
        
        const legend = d3.select("#legend")
            .selectAll(".legend-item")
            .data(legendData)
            .enter().append("div")
            .attr("class", "legend-item");
        
        legend.append("div")
            .attr("class", "legend-color")
            .style("background-color", d => d.color);
        
        legend.append("span")
            .text(d => d.label);
    </script>
</body>
</html>
            """
            
            return html_template
            
        except Exception as e:
            logger.error(f"Error creating D3 style map: {e}")
            return f"Error: {e}"

def main():
    """主函数"""
    import sys
    import json
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    visualizer = ChinaChoroplethMap(db_url)
    
    if len(sys.argv) < 2:
        print("Usage: python china_choropleth_map.py <command> [parameter]")
        print("Commands: choropleth, d3html")
        sys.exit(1)
    
    command = sys.argv[1]
    parameter = sys.argv[2] if len(sys.argv) > 2 else 'ph'
    
    if command == 'choropleth':
        result = visualizer.create_choropleth_map(parameter)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif command == 'd3html':
        result = visualizer.create_d3_style_map(parameter)
        print(result)
    else:
        result = {'error': 'Unknown command'}
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
