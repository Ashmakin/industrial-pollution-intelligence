
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct MLPrediction {
    pub station_name: String,
    pub parameter: String,
    pub forecast_time: chrono::DateTime<chrono::Utc>,
    pub prediction_value: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub model_name: String,
}

pub async fn call_python_model(model_type: &str, input_data: serde_json::Value) -> Result<MLPrediction, String> {
    
    
    Err("Python ML bridge not implemented yet".to_string())
}

