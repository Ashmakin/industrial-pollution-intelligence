use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct ForecastRequest {
    pub station_name: Option<String>,
    pub parameters: Vec<String>,
    pub forecast_horizon: u32, // hours
    pub model_type: String,    // "lstm", "prophet", "ensemble"
    pub include_confidence_interval: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForecastResult {
    pub station_name: String,
    pub parameter: String,
    pub forecast_id: Uuid,
    pub model_type: String,
    pub predictions: Vec<PredictionPoint>,
    pub model_metrics: Option<ModelMetrics>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct PredictionResult {
    pub id: i64,
    pub station_name: String,
    pub parameter: String,
    pub forecast_time: DateTime<Utc>,
    pub prediction_value: Option<f64>,
    pub confidence_lower: Option<f64>,
    pub confidence_upper: Option<f64>,
    pub model_name: Option<String>,
    pub model_version: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionPoint {
    pub timestamp: DateTime<Utc>,
    pub predicted_value: f64,
    pub confidence_interval_lower: Option<f64>,
    pub confidence_interval_upper: Option<f64>,
    pub confidence_level: Option<f64>, // 0.95 for 95% CI
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
    pub r2_score: Option<f64>,
    pub training_samples: u32,
    pub validation_samples: u32,
    pub model_parameters: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EnsembleForecast {
    pub station_name: String,
    pub parameter: String,
    pub ensemble_id: Uuid,
    pub individual_forecasts: Vec<ForecastResult>,
    pub ensemble_prediction: Vec<PredictionPoint>,
    pub model_weights: Vec<f64>,
    pub ensemble_metrics: ModelMetrics,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForecastAccuracy {
    pub station_name: String,
    pub parameter: String,
    pub model_type: String,
    pub accuracy_metrics: ModelMetrics,
    pub forecast_period: String,
    pub evaluation_date: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub model_id: Uuid,
    pub model_type: String,
    pub parameter: String,
    pub training_date: DateTime<Utc>,
    pub performance_metrics: ModelMetrics,
    pub feature_importance: Vec<FeatureImportance>,
    pub hyperparameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub feature_name: String,
    pub importance_score: f64,
    pub rank: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionConfidence {
    pub station_name: String,
    pub parameter: String,
    pub prediction_horizon_hours: u32,
    pub confidence_score: f64,
    pub uncertainty_sources: Vec<String>,
    pub reliability_indicators: ReliabilityIndicators,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReliabilityIndicators {
    pub data_quality_score: f64,
    pub model_accuracy_score: f64,
    pub temporal_stability_score: f64,
    pub spatial_correlation_score: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ForecastFilter {
    pub station_name: Option<String>,
    pub parameter: Option<String>,
    pub model_type: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub min_accuracy: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForecastComparison {
    pub station_name: String,
    pub parameter: String,
    pub comparison_id: Uuid,
    pub model_comparisons: Vec<ModelComparison>,
    pub best_model: String,
    pub comparison_metrics: ComparisonMetrics,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_type: String,
    pub accuracy_metrics: ModelMetrics,
    pub computational_cost: f64, // seconds
    pub memory_usage: f64,       // MB
    pub interpretability_score: f64, // 0-1 scale
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub performance_ranking: Vec<(String, f64)>, // model_type -> composite_score
    pub statistical_significance: Vec<StatisticalTest>,
    pub practical_significance: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub test_name: String,
    pub model1: String,
    pub model2: String,
    pub p_value: f64,
    pub effect_size: f64,
    pub significant: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionAlert {
    pub id: Uuid,
    pub station_name: String,
    pub parameter: String,
    pub alert_type: String, // "threshold_exceedance", "trend_change", "model_drift"
    pub severity: String,   // "low", "medium", "high", "critical"
    pub predicted_value: f64,
    pub threshold: f64,
    pub confidence_interval: (f64, f64),
    pub forecast_horizon_hours: u32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelTrainingJob {
    pub job_id: Uuid,
    pub model_type: String,
    pub parameters: Vec<String>,
    pub training_data_range: (DateTime<Utc>, DateTime<Utc>),
    pub status: String, // "pending", "running", "completed", "failed"
    pub progress_percentage: u32,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}
