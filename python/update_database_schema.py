                      
"""
更新数据库表结构
添加数据哈希字段用于去重
"""

import psycopg2
import logging
from typing import Optional

      
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_database_schema(db_url: str) -> bool:
    """更新数据库表结构"""
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
                 
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'water_quality_data'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.info("Creating water_quality_data table...")
            
                  
            cursor.execute("""
                CREATE TABLE water_quality_data (
                    id SERIAL PRIMARY KEY,
                    province VARCHAR(50),
                    basin VARCHAR(100),
                    station_name VARCHAR(200),
                    monitoring_time TIMESTAMP,
                    water_quality_class VARCHAR(10),
                    temperature DECIMAL(10,2),
                    ph DECIMAL(10,2),
                    dissolved_oxygen DECIMAL(10,2),
                    conductivity DECIMAL(10,2),
                    turbidity DECIMAL(10,2),
                    permanganate_index DECIMAL(10,2),
                    ammonia_nitrogen DECIMAL(10,2),
                    total_phosphorus DECIMAL(10,2),
                    total_nitrogen DECIMAL(10,2),
                    chlorophyll_a DECIMAL(10,2),
                    algae_density DECIMAL(10,2),
                    station_status VARCHAR(50),
                    data_hash VARCHAR(64) UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
                  
            cursor.execute("CREATE INDEX idx_water_quality_province ON water_quality_data(province);")
            cursor.execute("CREATE INDEX idx_water_quality_basin ON water_quality_data(basin);")
            cursor.execute("CREATE INDEX idx_water_quality_station ON water_quality_data(station_name);")
            cursor.execute("CREATE INDEX idx_water_quality_time ON water_quality_data(monitoring_time);")
            cursor.execute("CREATE INDEX idx_water_quality_hash ON water_quality_data(data_hash);")
            
            logger.info("Table created successfully")
            
        else:
            logger.info("Table exists, checking for missing columns...")
            
                       
            columns_to_add = [
                ('basin', 'VARCHAR(100)'),
                ('station_name', 'VARCHAR(200)'),
                ('water_quality_class', 'VARCHAR(10)'),
                ('temperature', 'DECIMAL(10,2)'),
                ('conductivity', 'DECIMAL(10,2)'),
                ('turbidity', 'DECIMAL(10,2)'),
                ('permanganate_index', 'DECIMAL(10,2)'),
                ('total_nitrogen', 'DECIMAL(10,2)'),
                ('chlorophyll_a', 'DECIMAL(10,2)'),
                ('algae_density', 'DECIMAL(10,2)'),
                ('station_status', 'VARCHAR(50)'),
                ('data_hash', 'VARCHAR(64)'),
                ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
                ('updated_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
            ]
            
            for column_name, column_type in columns_to_add:
                try:
                    cursor.execute(f"ALTER TABLE water_quality_data ADD COLUMN {column_name} {column_type};")
                    logger.info(f"Added column: {column_name}")
                except psycopg2.Error as e:
                    if "already exists" in str(e):
                        logger.info(f"Column {column_name} already exists")
                    else:
                        logger.warning(f"Error adding column {column_name}: {e}")
            
                              
            try:
                cursor.execute("ALTER TABLE water_quality_data ADD CONSTRAINT unique_data_hash UNIQUE (data_hash);")
                logger.info("Added unique constraint to data_hash")
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    logger.info("Unique constraint on data_hash already exists")
                else:
                    logger.warning(f"Error adding unique constraint: {e}")
            
                         
            indexes_to_create = [
                ("idx_water_quality_basin", "CREATE INDEX idx_water_quality_basin ON water_quality_data(basin);"),
                ("idx_water_quality_station", "CREATE INDEX idx_water_quality_station ON water_quality_data(station_name);"),
                ("idx_water_quality_time", "CREATE INDEX idx_water_quality_time ON water_quality_data(monitoring_time);"),
                ("idx_water_quality_hash", "CREATE INDEX idx_water_quality_hash ON water_quality_data(data_hash);")
            ]
            
            for index_name, index_sql in indexes_to_create:
                try:
                    cursor.execute(index_sql)
                    logger.info(f"Created index: {index_name}")
                except psycopg2.Error as e:
                    if "already exists" in str(e):
                        logger.info(f"Index {index_name} already exists")
                    else:
                        logger.warning(f"Error creating index {index_name}: {e}")
        
        conn.commit()
        logger.info("Database schema updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def main():
    """主函数"""
    import os
    
    db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
    
    success = update_database_schema(db_url)
    if success:
        print("Database schema updated successfully")
    else:
        print("Failed to update database schema")
        exit(1)

if __name__ == "__main__":
    main()
