                      
import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

PROVINCE_GEOJSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '中国_省.geojson'))
                                                                                                    


def _get_db_connection(db_url: str):
    return psycopg2.connect(db_url)


def _load_geojson(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _calc_level(parameter: str, value: float) -> Tuple[str, str]:
    if value is None:
        return "no_data", "#CCCCCC"
    if parameter == 'ph':
        if 6.5 <= value <= 8.5: return 'excellent', '#2E8B57'
        if 6.0 <= value <= 9.0: return 'good', '#90EE90'
        if 5.5 <= value <= 9.5: return 'mild_pollution', '#FFD700'
        return 'moderate_pollution', '#FFA500'
    if parameter == 'ammonia_nitrogen':
        if value <= 0.15: return 'excellent', '#2E8B57'
        if value <= 0.5: return 'good', '#90EE90'
        if value <= 1.0: return 'mild_pollution', '#FFD700'
        if value <= 1.5: return 'moderate_pollution', '#FFA500'
        if value <= 2.0: return 'heavy_pollution', '#FF0000'
        return 'severe_pollution', '#8B0000'
    if parameter == 'dissolved_oxygen':
        if value >= 7.5: return 'excellent', '#2E8B57'
        if value >= 6.0: return 'good', '#90EE90'
        if value >= 5.0: return 'mild_pollution', '#FFD700'
        if value >= 3.0: return 'moderate_pollution', '#FFA500'
        if value >= 2.0: return 'heavy_pollution', '#FF0000'
        return 'severe_pollution', '#8B0000'
    if parameter == 'total_phosphorus':
        if value <= 0.02: return 'excellent', '#2E8B57'
        if value <= 0.1: return 'good', '#90EE90'
        if value <= 0.2: return 'mild_pollution', '#FFD700'
        if value <= 0.3: return 'moderate_pollution', '#FFA500'
        if value <= 0.4: return 'heavy_pollution', '#FF0000'
        return 'severe_pollution', '#8B0000'
    return 'no_data', '#CCCCCC'


def _fetch_latest_province_avgs(db_url: str, parameter: str) -> Dict[str, float]:
    conn = None
    try:
        conn = _get_db_connection(db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
                                
        query = f"""
            WITH latest_times AS (
                SELECT province, MAX(monitoring_time) AS latest_time
                FROM water_quality_data
                WHERE province IS NOT NULL AND {parameter} IS NOT NULL
                GROUP BY province
            )
            SELECT w.province, AVG(w.{parameter}) AS avg_value
            FROM water_quality_data w
            JOIN latest_times lt ON lt.province = w.province AND lt.latest_time = w.monitoring_time
            WHERE w.{parameter} IS NOT NULL
            GROUP BY w.province
        """
        cur.execute(query)
        rows = cur.fetchall()
        return {row['province']: float(row['avg_value']) for row in rows if row['avg_value'] is not None}
    except Exception as e:
        logger.error(f"DB error: {e}")
        return {}
    finally:
        if conn:
            conn.close()


def create_map(parameter: str) -> Dict:
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    province_geo = _load_geojson(PROVINCE_GEOJSON_PATH)
    province_avgs = _fetch_latest_province_avgs(db_url, parameter)

    features = []
    for feature in province_geo.get('features', []):
        props = feature.get('properties', {})
                       
        name = props.get('name') or props.get('NAME') or props.get('Name') or props.get('省') or props.get('省份')
        code = props.get('adcode') or props.get('code') or props.get('adCode')
        val = province_avgs.get(name)
        level, color = _calc_level(parameter, val)
        new_props = dict(props)
        new_props.update({
            'value': val,
            'pollution_level': level,
            'color': color
        })
        features.append({
            'type': 'Feature',
            'properties': new_props,
            'geometry': feature.get('geometry')
        })

    return {
        'map_type': 'choropleth',
        'geojson_data': {
            'type': 'FeatureCollection',
            'features': features
        },
        'map_config': {
            'map_type': 'choropleth',
            'parameter': parameter,
            'color_scheme': 'sequential',
            'projection': 'geoMercator',
            'scale': 1000,
            'center': [104.1954, 35.8617],
            'zoom': 4,
            'width': 800,
            'height': 600
        },
        'data_points': len([f for f in features if f['properties'].get('value') is not None]),
        'parameter': parameter,
        'timestamp': datetime.now().isoformat()
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python use_real_geojson_map.py <parameter>")
        sys.exit(1)
    parameter = sys.argv[1]
    result = create_map(parameter)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
