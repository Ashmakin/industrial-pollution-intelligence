-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create water quality monitoring table
CREATE TABLE water_quality_data (
    id BIGSERIAL PRIMARY KEY,
    station_name VARCHAR(255) NOT NULL,
    station_code VARCHAR(100),
    province VARCHAR(100),
    watershed VARCHAR(100),
    monitoring_time TIMESTAMPTZ NOT NULL,
    
    -- Physical parameters
    temperature DECIMAL(8,3),
    ph DECIMAL(6,3),
    dissolved_oxygen DECIMAL(8,3),
    conductivity DECIMAL(10,3),
    turbidity DECIMAL(8,3),
    
    -- Chemical parameters
    permanganate_index DECIMAL(8,3),
    ammonia_nitrogen DECIMAL(8,3),
    total_phosphorus DECIMAL(8,3),
    total_nitrogen DECIMAL(8,3),
    
    -- Biological parameters
    chlorophyll_a DECIMAL(8,3),
    algae_density DECIMAL(8,3),
    
    -- Derived metrics
    water_quality_grade INTEGER,
    pollution_index DECIMAL(8,3),
    
    -- Metadata
    data_source VARCHAR(100) DEFAULT 'CNEMC',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_water_quality_time ON water_quality_data (monitoring_time);
CREATE INDEX idx_water_quality_station ON water_quality_data (station_name);
CREATE INDEX idx_water_quality_province ON water_quality_data (province);
CREATE INDEX idx_water_quality_watershed ON water_quality_data (watershed);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('water_quality_data', 'monitoring_time', chunk_time_interval => INTERVAL '1 day');

-- Create forecasting results table
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

-- Create hypertable for forecasting results
SELECT create_hypertable('forecasting_results', 'forecast_time', chunk_time_interval => INTERVAL '1 day');

-- Create analysis results table
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

-- Create stations metadata table
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

-- Insert sample stations (will be populated by data collection)
INSERT INTO monitoring_stations (station_name, station_code, province, watershed, latitude, longitude) VALUES
('Sample Station 1', 'SS001', 'Beijing', 'Haihe River', 39.9042, 116.4074),
('Sample Station 2', 'SS002', 'Shanghai', 'Yangtze River', 31.2304, 121.4737),
('Sample Station 3', 'SS003', 'Guangdong', 'Pearl River', 23.1291, 113.2644);

-- Create trigger for updating updated_at timestamp
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

