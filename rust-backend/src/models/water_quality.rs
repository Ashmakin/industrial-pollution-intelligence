use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct WaterQualityStation {
    pub id: i32,
    pub station_name: String,
    pub province: String,
    pub watershed: String,
    pub area_id: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct WaterQualityMeasurement {
    pub id: i32,
    pub station_id: i32,
    pub monitoring_time: DateTime<Utc>,
    pub water_quality_grade: Option<String>,
    pub temperature: Option<f64>,
    pub ph: Option<f64>,
    pub dissolved_oxygen: Option<f64>,
    pub conductivity: Option<f64>,
    pub turbidity: Option<f64>,
    pub permanganate_index: Option<f64>,
    pub ammonia_nitrogen: Option<f64>,
    pub total_phosphorus: Option<f64>,
    pub total_nitrogen: Option<f64>,
    pub chlorophyll_a: Option<f64>,
    pub algae_density: Option<f64>,
    pub station_status: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StationWithMeasurements {
    #[serde(flatten)]
    pub station: WaterQualityStation,
    pub latest_measurement: Option<WaterQualityMeasurement>,
    pub measurement_count: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MeasurementStatistics {
    pub station_name: String,
    pub parameter: String,
    pub count: i64,
    pub mean: Option<f64>,
    pub std_dev: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub q25: Option<f64>,
    pub q50: Option<f64>,
    pub q75: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationResult {
    pub parameter1: String,
    pub parameter2: String,
    pub correlation: f64,
    pub p_value: Option<f64>,
    pub station_count: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PollutionTrend {
    pub station_name: String,
    pub parameter: String,
    pub trend_direction: String, // "increasing", "decreasing", "stable"
    pub trend_strength: f64,
    pub p_value: f64,
    pub confidence_interval_lower: f64,
    pub confidence_interval_upper: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub station_name: String,
    pub parameter: String,
    pub anomaly_score: f64,
    pub anomaly_type: String, // "statistical", "isolation_forest", "autoencoder"
    pub detected_at: DateTime<Utc>,
    pub value: f64,
    pub threshold: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WaterQualityGradeDistribution {
    pub grade: String,
    pub count: i64,
    pub percentage: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpatialCluster {
    pub cluster_id: i32,
    pub center_lat: f64,
    pub center_lon: f64,
    pub station_count: i32,
    pub avg_pollution_level: f64,
    pub dominant_pollutants: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub station_name: String,
    pub parameter: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PollutionAlert {
    pub id: Uuid,
    pub station_name: String,
    pub parameter: String,
    pub alert_type: String, // "exceedance", "trend", "anomaly"
    pub severity: String,   // "low", "medium", "high", "critical"
    pub message: String,
    pub threshold_value: f64,
    pub actual_value: f64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct MeasurementFilter {
    pub station_name: Option<String>,
    pub province: Option<String>,
    pub watershed: Option<String>,
    pub parameter: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WaterQualitySummary {
    pub total_stations: i64,
    pub total_measurements: i64,
    pub latest_measurement_time: Option<DateTime<Utc>>,
    pub grade_distribution: Vec<WaterQualityGradeDistribution>,
    pub top_pollutants: Vec<(String, f64)>, // parameter -> average concentration
    pub critical_stations: Vec<String>,
}

