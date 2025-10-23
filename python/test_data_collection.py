                      
"""
Simple data collection test script for the Industrial Pollution Intelligence System
"""

import asyncio
import sys
import os
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import random

                                         
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def create_test_data():
    """Create test water quality data for demonstration"""
    
                         
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="pollution_db",
        user="pollution_user",
        password="pollution_pass"
    )
    
    cursor = conn.cursor()
    
                                                
    stations = [
        ("Beijing Station", "BJ001", "Beijing", "Haihe River", 39.9042, 116.4074),
        ("Shanghai Station", "SH001", "Shanghai", "Yangtze River", 31.2304, 121.4737),
        ("Guangdong Station", "GD001", "Guangdong", "Pearl River", 23.1291, 113.2644),
        ("Tianjin Station", "TJ001", "Tianjin", "Haihe River", 39.3434, 117.3616),
        ("Chongqing Station", "CQ001", "Chongqing", "Yangtze River", 29.4316, 106.9123),
    ]
    
    print("Creating test stations...")
    for station_name, station_code, province, watershed, lat, lon in stations:
        cursor.execute("""
            INSERT INTO monitoring_stations (station_name, station_code, province, watershed, latitude, longitude)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (station_name) DO NOTHING
        """, (station_name, station_code, province, watershed, lat, lon))
    
    conn.commit()
    
                                      
    print("Generating test water quality data...")
    
                                    
    base_values = {
        'temperature': 20.0,
        'ph': 7.5,
        'dissolved_oxygen': 8.0,
        'conductivity': 500.0,
        'turbidity': 2.0,
        'permanganate_index': 3.0,
        'ammonia_nitrogen': 0.5,
        'total_phosphorus': 0.1,
        'total_nitrogen': 2.0,
        'chlorophyll_a': 10.0,
        'algae_density': 100.0
    }
    
                                                          
    start_date = datetime.now() - timedelta(days=30)
    data_points = []
    
    for station_name, _, _, _, _, _ in stations:
        for i in range(180):                                    
            timestamp = start_date + timedelta(hours=i*4)
            
                                                          
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / (24 * 30))                 
            
            values = {}
            for param, base_val in base_values.items():
                                                                                               
                noise = random.gauss(0, base_val * 0.1)             
                seasonal = base_val * (seasonal_factor - 1) * 0.3                          
                
                                                                                             
                if param == 'ammonia_nitrogen' and random.random() < 0.05:                                
                    values[param] = base_val * random.uniform(2, 5) + noise + seasonal
                elif param == 'dissolved_oxygen' and random.random() < 0.05:                                
                    values[param] = max(0, base_val * random.uniform(0.3, 0.7) + noise + seasonal)
                else:
                    values[param] = max(0, base_val + noise + seasonal)
            
                                                        
            if values['ammonia_nitrogen'] > 2.0 or values['total_phosphorus'] > 0.4:
                water_quality_grade = 5        
            elif values['ammonia_nitrogen'] > 1.0 or values['total_phosphorus'] > 0.2:
                water_quality_grade = 4                 
            elif values['dissolved_oxygen'] < 5.0 or values['ph'] < 6.5 or values['ph'] > 8.5:
                water_quality_grade = 3           
            elif values['dissolved_oxygen'] > 6.0 and 6.5 <= values['ph'] <= 8.5:
                water_quality_grade = 2        
            else:
                water_quality_grade = 1             
            
                                                    
            pollution_index = (
                values['ammonia_nitrogen'] * 10 +
                values['total_phosphorus'] * 20 +
                values['total_nitrogen'] * 2 +
                (8.0 - values['dissolved_oxygen']) * 5 +
                abs(values['ph'] - 7.0) * 10
            )
            
            data_points.append({
                'station_name': station_name,
                'station_code': None,
                'province': next(p[2] for p in stations if p[0] == station_name),
                'watershed': next(p[3] for p in stations if p[0] == station_name),
                'monitoring_time': timestamp,
                'temperature': float(round(values['temperature'], 3)),
                'ph': float(round(values['ph'], 3)),
                'dissolved_oxygen': float(round(values['dissolved_oxygen'], 3)),
                'conductivity': float(round(values['conductivity'], 3)),
                'turbidity': float(round(values['turbidity'], 3)),
                'permanganate_index': float(round(values['permanganate_index'], 3)),
                'ammonia_nitrogen': float(round(values['ammonia_nitrogen'], 3)),
                'total_phosphorus': float(round(values['total_phosphorus'], 3)),
                'total_nitrogen': float(round(values['total_nitrogen'], 3)),
                'chlorophyll_a': float(round(values['chlorophyll_a'], 3)),
                'algae_density': float(round(values['algae_density'], 3)),
                'water_quality_grade': water_quality_grade,
                'pollution_index': float(round(pollution_index, 3)),
                'data_source': 'TEST_DATA'
            })
    
                            
    print(f"Inserting {len(data_points)} data points...")
    
    batch_size = 100
    for i in range(0, len(data_points), batch_size):
        batch = data_points[i:i + batch_size]
        
        insert_query = """
            INSERT INTO water_quality_data (
                station_name, station_code, province, watershed, monitoring_time,
                temperature, ph, dissolved_oxygen, conductivity, turbidity,
                permanganate_index, ammonia_nitrogen, total_phosphorus, total_nitrogen,
                chlorophyll_a, algae_density, water_quality_grade, pollution_index, data_source
            ) VALUES (
                %(station_name)s, %(station_code)s, %(province)s, %(watershed)s, %(monitoring_time)s,
                %(temperature)s, %(ph)s, %(dissolved_oxygen)s, %(conductivity)s, %(turbidity)s,
                %(permanganate_index)s, %(ammonia_nitrogen)s, %(total_phosphorus)s, %(total_nitrogen)s,
                %(chlorophyll_a)s, %(algae_density)s, %(water_quality_grade)s, %(pollution_index)s, %(data_source)s
            )
        """
        
        cursor.executemany(insert_query, batch)
        conn.commit()
        print(f"Inserted batch {i//batch_size + 1}/{(len(data_points) + batch_size - 1)//batch_size}")
    
                           
    cursor.execute("SELECT COUNT(*) FROM water_quality_data")
    count = cursor.fetchone()[0]
    print(f"Total data points in database: {count}")
    
    cursor.execute("SELECT COUNT(DISTINCT station_name) FROM water_quality_data")
    station_count = cursor.fetchone()[0]
    print(f"Total stations with data: {station_count}")
    
    cursor.close()
    conn.close()
    
    print("Test data creation completed successfully!")

if __name__ == "__main__":
    import numpy as np
    asyncio.run(create_test_data())
