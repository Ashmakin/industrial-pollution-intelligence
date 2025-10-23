

CREATE TABLE water_quality_data (
    id BIGSERIAL PRIMARY KEY,
    station_name VARCHAR(255) NOT NULL,
    station_code VARCHAR(100),
    province VARCHAR(100),
    watershed VARCHAR(100),
    monitoring_time TIMESTAMPTZ NOT NULL,
    
    
    temperature DECIMAL(8,3),
    ph DECIMAL(6,3),
    dissolved_oxygen DECIMAL(8,3),
    conductivity DECIMAL(10,3),
    turbidity DECIMAL(8,3),
    
    
    permanganate_index DECIMAL(8,3),
    ammonia_nitrogen DECIMAL(8,3),
    total_phosphorus DECIMAL(8,3),
    total_nitrogen DECIMAL(8,3),
    
    
    chlorophyll_a DECIMAL(8,3),
    algae_density DECIMAL(8,3),
    
    
    water_quality_grade INTEGER,
    pollution_index DECIMAL(8,3),
    
    
    data_source VARCHAR(100) DEFAULT 'CNEMC',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


CREATE INDEX idx_water_quality_time ON water_quality_data (monitoring_time);
CREATE INDEX idx_water_quality_station ON water_quality_data (station_name);
CREATE INDEX idx_water_quality_province ON water_quality_data (province);
CREATE INDEX idx_water_quality_watershed ON water_quality_data (watershed);


CREATE TABLE forecasting_results (
    id BIGSERIAL PRIMARY KEY,
    station_name VARCHAR(255) NOT NULL,
    parameter VARCHAR(100) NOT NULL,
    forecast_time TIMESTAMPTZ NOT NULL,
    prediction_value DECIMAL(12,6),
    confidence_lower DECIMAL(12,6),
    confidence_upper DECIMAL(12,6),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);


CREATE TABLE analysis_results (
    id BIGSERIAL PRIMARY KEY,
    analysis_type VARCHAR(100) NOT NULL,
    station_name VARCHAR(255),
    parameter VARCHAR(100),
    result_key VARCHAR(255),
    result_value DECIMAL(12,6),
    result_text TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


CREATE TABLE monitoring_stations (
    id BIGSERIAL PRIMARY KEY,
    station_name VARCHAR(255) UNIQUE NOT NULL,
    station_code VARCHAR(100),
    province VARCHAR(100),
    watershed VARCHAR(100),
    latitude DECIMAL(10,7),
    longitude DECIMAL(10,7),
    elevation DECIMAL(8,2),
    station_type VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


INSERT INTO monitoring_stations (station_name, station_code, province, watershed, latitude, longitude) VALUES
('Sample Station 1', 'SS001', 'Beijing', 'Haihe River', 39.9042, 116.4074),
('Sample Station 2', 'SS002', 'Shanghai', 'Yangtze River', 31.2304, 121.4737),
('Sample Station 3', 'SS003', 'Guangdong', 'Pearl River', 23.1291, 113.2644);


CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_water_quality_updated_at BEFORE UPDATE ON water_quality_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_stations_updated_at BEFORE UPDATE ON monitoring_stations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

