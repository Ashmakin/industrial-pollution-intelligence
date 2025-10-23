                      
"""
中国地图污染可视化系统
- 使用Folium创建交互式地图
- 根据污染程度用颜色分级显示
- 实时更新污染数据
- 支持不同参数的可视化
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

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Folium not available, install with: pip install folium")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, install with: pip install plotly")

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChinaMapVisualizer:
    """中国地图污染可视化器"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
                 
        self.province_coordinates = {
            '北京': {'lat': 39.9042, 'lon': 116.4074, 'code': '110000'},
            '天津': {'lat': 39.3434, 'lon': 117.3616, 'code': '120000'},
            '河北': {'lat': 38.0428, 'lon': 114.5149, 'code': '130000'},
            '山西': {'lat': 37.8735, 'lon': 112.5624, 'code': '140000'},
            '内蒙古': {'lat': 40.8175, 'lon': 111.7652, 'code': '150000'},
            '辽宁': {'lat': 41.8057, 'lon': 123.4315, 'code': '210000'},
            '吉林': {'lat': 43.8868, 'lon': 125.3245, 'code': '220000'},
            '黑龙江': {'lat': 45.7736, 'lon': 126.6617, 'code': '230000'},
            '上海': {'lat': 31.2304, 'lon': 121.4737, 'code': '310000'},
            '江苏': {'lat': 32.0603, 'lon': 118.7969, 'code': '320000'},
            '浙江': {'lat': 30.2741, 'lon': 120.1551, 'code': '330000'},
            '安徽': {'lat': 31.8612, 'lon': 117.2849, 'code': '340000'},
            '福建': {'lat': 26.0745, 'lon': 119.2965, 'code': '350000'},
            '江西': {'lat': 28.6765, 'lon': 115.8922, 'code': '360000'},
            '山东': {'lat': 36.6512, 'lon': 117.1201, 'code': '370000'},
            '河南': {'lat': 34.7578, 'lon': 113.6254, 'code': '410000'},
            '湖北': {'lat': 30.5928, 'lon': 114.3055, 'code': '420000'},
            '湖南': {'lat': 28.2278, 'lon': 112.9388, 'code': '430000'},
            '广东': {'lat': 23.3417, 'lon': 113.4244, 'code': '440000'},
            '广西': {'lat': 22.8170, 'lon': 108.3661, 'code': '450000'},
            '海南': {'lat': 20.0311, 'lon': 110.3312, 'code': '460000'},
            '重庆': {'lat': 29.5647, 'lon': 106.5507, 'code': '500000'},
            '四川': {'lat': 30.6512, 'lon': 104.0665, 'code': '510000'},
            '贵州': {'lat': 26.5783, 'lon': 106.7074, 'code': '520000'},
            '云南': {'lat': 25.0389, 'lon': 102.7183, 'code': '530000'},
            '西藏': {'lat': 29.6465, 'lon': 91.1172, 'code': '540000'},
            '陕西': {'lat': 34.2658, 'lon': 108.9540, 'code': '610000'},
            '甘肃': {'lat': 36.0611, 'lon': 103.8343, 'code': '620000'},
            '青海': {'lat': 36.6232, 'lon': 101.7782, 'code': '630000'},
            '宁夏': {'lat': 38.4872, 'lon': 106.2309, 'code': '640000'},
            '新疆': {'lat': 43.7928, 'lon': 87.6177, 'code': '650000'}
        }
        
                
        self.pollution_colors = {
            'excellent': '#00E400',             
            'good': '#FFFF00',                  
            'light': '#FF7E00',                   
            'moderate': '#FF0000',                
            'heavy': '#8F3F97',                   
            'severe': '#7E0023'                    
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
    
    def get_latest_data(self, parameter: str) -> Dict[str, Dict]:
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
                    AVG({column_name}) as avg_value,
                    MAX(monitoring_time) as latest_time,
                    COUNT(*) as data_count
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
                if province and province in self.province_coordinates:
                    if province not in result:
                        result[province] = {
                            'stations': [],
                            'avg_value': 0,
                            'data_count': 0
                        }
                    
                    result[province]['stations'].append(station_name)
                    result[province]['avg_value'] = avg_value
                    result[province]['data_count'] += row['data_count']
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return {}
    
    def extract_province_from_station(self, station_name: str) -> Optional[str]:
        """从站点名称提取省份"""
        for province in self.province_coordinates.keys():
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
    
    def create_folium_map(self, parameter: str, output_file: str = 'china_pollution_map.html') -> str:
        """创建Folium交互式地图"""
        if not FOLIUM_AVAILABLE:
            return "Folium not available"
        
        try:
                  
            data = self.get_latest_data(parameter)
            
                  
            m = folium.Map(
                location=[35.8617, 104.1954],        
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
                  
            title_html = f'''
                <h3 align="center" style="font-size:20px">
                <b>中国{parameter.upper()}污染分布图</b>
                <br>
                <small>数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
                </h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
                   
            for province, info in data.items():
                coords = self.province_coordinates[province]
                pollution_level = self.get_pollution_level(parameter, info['avg_value'])
                color = self.pollution_colors.get(pollution_level, '#808080')
                
                        
                popup_text = f'''
                <b>{province}</b><br>
                参数: {parameter}<br>
                平均值: {info['avg_value']:.3f}<br>
                污染等级: {pollution_level}<br>
                监测站数量: {len(info['stations'])}<br>
                数据点数: {info['data_count']}
                '''
                
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=15,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='black',
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(m)
            
                  
            legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 200px; height: 150px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; padding: 10px">
                <p><b>污染等级</b></p>
                <p><i class="fa fa-circle" style="color:#00E400"></i> 优秀</p>
                <p><i class="fa fa-circle" style="color:#FFFF00"></i> 良好</p>
                <p><i class="fa fa-circle" style="color:#FF7E00"></i> 轻度污染</p>
                <p><i class="fa fa-circle" style="color:#FF0000"></i> 中度污染</p>
                <p><i class="fa fa-circle" style="color:#8F3F97"></i> 重度污染</p>
                <p><i class="fa fa-circle" style="color:#7E0023"></i> 严重污染</p>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
                  
            m.save(output_file)
            logger.info(f"Map saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error creating Folium map: {e}")
            return f"Error: {e}"
    
    def create_plotly_map(self, parameter: str) -> Dict:
        """创建Plotly地图"""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available'}
        
        try:
                  
            data = self.get_latest_data(parameter)
            
                  
            lats = []
            lons = []
            values = []
            texts = []
            colors = []
            
            for province, info in data.items():
                coords = self.province_coordinates[province]
                pollution_level = self.get_pollution_level(parameter, info['avg_value'])
                color = self.pollution_colors.get(pollution_level, '#808080')
                
                lats.append(coords['lat'])
                lons.append(coords['lon'])
                values.append(info['avg_value'])
                colors.append(color)
                
                text = f'{province}<br>'
                text += f'参数: {parameter}<br>'
                text += f'平均值: {info["avg_value"]:.3f}<br>'
                text += f'污染等级: {pollution_level}<br>'
                text += f'监测站: {len(info["stations"])}'
                texts.append(text)
            
                   
            fig = go.Figure()
            
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=20,
                    color=colors,
                    opacity=0.8
                ),
                text=texts,
                hovertemplate='%{text}<extra></extra>',
                name='监测点'
            ))
            
                    
            fig.update_layout(
                title=f'中国{parameter.upper()}污染分布图',
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=35.8617, lon=104.1954),
                    zoom=4
                ),
                height=600,
                showlegend=False
            )
            
                     
            map_json = fig.to_json()
            
            return {
                'map_type': 'plotly',
                'map_data': map_json,
                'data_points': len(data),
                'parameter': parameter,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating Plotly map: {e}")
            return {'error': str(e)}
    
    def create_pollution_summary(self) -> Dict:
        """创建污染汇总报告"""
        try:
            summary = {}
            
                        
            parameters = ['ph', 'dissolved_oxygen', 'ammonia_nitrogen', 'total_phosphorus']
            
            for param in parameters:
                data = self.get_latest_data(param)
                
                        
                pollution_stats = {
                    'excellent': 0,
                    'good': 0,
                    'light': 0,
                    'moderate': 0,
                    'heavy': 0,
                    'severe': 0
                }
                
                total_provinces = len(data)
                
                for province, info in data.items():
                    level = self.get_pollution_level(param, info['avg_value'])
                    pollution_stats[level] += 1
                
                summary[param] = {
                    'total_provinces': total_provinces,
                    'pollution_distribution': pollution_stats,
                    'data_quality': 'good' if total_provinces > 10 else 'limited'
                }
            
            return {
                'summary_type': 'pollution_overview',
                'timestamp': datetime.now().isoformat(),
                'parameters': summary,
                'overall_quality': self.calculate_overall_quality(summary)
            }
            
        except Exception as e:
            logger.error(f"Error creating pollution summary: {e}")
            return {'error': str(e)}
    
    def calculate_overall_quality(self, summary: Dict) -> str:
        """计算整体水质"""
        try:
            total_excellent = 0
            total_good = 0
            total_provinces = 0
            
            for param, data in summary.items():
                if isinstance(data, dict) and 'pollution_distribution' in data:
                    total_excellent += data['pollution_distribution'].get('excellent', 0)
                    total_good += data['pollution_distribution'].get('good', 0)
                    total_provinces += data.get('total_provinces', 0)
            
            if total_provinces == 0:
                return 'unknown'
            
            good_ratio = (total_excellent + total_good) / (total_provinces * len(summary))
            
            if good_ratio > 0.7:
                return 'excellent'
            elif good_ratio > 0.5:
                return 'good'
            elif good_ratio > 0.3:
                return 'moderate'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return 'unknown'
    
    def generate_dashboard_data(self) -> Dict:
        """生成仪表盘数据"""
        try:
                    
            map_data = {}
            parameters = ['ph', 'dissolved_oxygen', 'ammonia_nitrogen', 'total_phosphorus']
            
            for param in parameters:
                map_data[param] = self.create_plotly_map(param)
            
                    
            summary = self.create_pollution_summary()
            
                      
            charts_data = self.generate_charts_data()
            
            return {
                'dashboard_type': 'china_pollution_dashboard',
                'timestamp': datetime.now().isoformat(),
                'maps': map_data,
                'summary': summary,
                'charts': charts_data
            }
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {'error': str(e)}
    
    def generate_charts_data(self) -> Dict:
        """生成图表数据"""
        try:
            charts = {}
            
                      
            conn = psycopg2.connect(self.db_url)
            
                    
            parameters = ['ph', 'dissolved_oxygen', 'ammonia_nitrogen', 'total_phosphorus']
            
                  
            column_mapping = {
                'ph': 'ph',
                'dissolved_oxygen': 'dissolved_oxygen',
                'ammonia_nitrogen': 'ammonia_nitrogen',
                'total_phosphorus': 'total_phosphorus'
            }
            
            for param in parameters:
                if param not in column_mapping:
                    continue
                    
                column_name = column_mapping[param]
                query = f"""
                    SELECT 
                        DATE(monitoring_time) as date,
                        AVG({column_name}) as avg_value,
                        COUNT(*) as data_count
                    FROM water_quality_data 
                    WHERE {column_name} IS NOT NULL 
                    AND monitoring_time >= %s
                    GROUP BY DATE(monitoring_time)
                    ORDER BY date
                """
                
                start_date = datetime.now() - timedelta(days=30)
                df = pd.read_sql_query(query, conn, params=[start_date])
                
                if not df.empty:
                    charts[f'{param}_trend'] = {
                        'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                        'values': df['avg_value'].tolist(),
                        'data_counts': df['data_count'].tolist()
                    }
            
            conn.close()
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts data: {e}")
            return {}

def main():
    """主函数"""
    import sys
    import json
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    visualizer = ChinaMapVisualizer(db_url)
    
    if len(sys.argv) < 2:
        print("Usage: python china_map_visualization.py <command> [parameter]")
        print("Commands: map, dashboard, summary")
        sys.exit(1)
    
    command = sys.argv[1]
    parameter = sys.argv[2] if len(sys.argv) > 2 else 'ph'
    
    if command == 'map':
        result = visualizer.create_plotly_map(parameter)
    elif command == 'dashboard':
        result = visualizer.generate_dashboard_data()
    elif command == 'summary':
        result = visualizer.create_pollution_summary()
    else:
        result = {'error': 'Unknown command'}
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
