use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisQuery {
    pub analysis_type: Option<String>,
    pub station_name: Option<String>,
    pub parameter: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct AnalysisResult {
    pub id: i64,
    pub analysis_type: String,
    pub station_name: Option<String>,
    pub parameter: Option<String>,
    pub result_key: Option<String>,
    pub result_value: Option<f64>,
    pub result_text: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: Option<String>,
}

pub async fn get_analysis_results(
    State(_pool): State<PgPool>,
    Query(params): Query<AnalysisQuery>,
    axum::extract::Path(analysis_type): axum::extract::Path<String>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    // 调用真实的Python ML脚本进行数据分析
    println!("Analysis type from path: {}", analysis_type);
    
    // 构建Python脚本命令
    let mut command_parts = vec![
        "cd ../python && source venv/bin/activate && python3 run_analysis.py".to_string(),
        format!("--analysis-type {}", analysis_type),
        "--database-url postgres://pollution_user:pollution_pass@localhost:5432/pollution_db".to_string(),
        "--days 30".to_string()
    ];
    
    if let Some(station_name) = &params.station_name {
        if !station_name.is_empty() {
            command_parts.push(format!("--stations \"{}\"", station_name));
        }
    }
    
    if let Some(parameter) = &params.parameter {
        if !parameter.is_empty() {
            command_parts.push(format!("--parameters \"{}\"", parameter));
        }
    }
    
    let script_command = command_parts.join(" ");
    
    let output = std::process::Command::new("bash")
        .arg("-c")
        .arg(&script_command)
        .output();
    
    match output {
        Ok(output) => {
            if output.status.success() {
                let result_str = String::from_utf8_lossy(&output.stdout);
                println!("Python analysis output: {}", result_str);
                
                // 尝试解析JSON结果
                match serde_json::from_str::<serde_json::Value>(&result_str) {
                    Ok(analysis_result) => {
                        Ok(Json(ApiResponse {
                            success: true,
                            data: Some(analysis_result),
                            message: Some("Real data analysis completed successfully".to_string()),
                        }))
                    }
                    Err(e) => {
                        println!("Failed to parse analysis result: {}", e);
                        Ok(Json(ApiResponse {
                            success: false,
                            data: None,
                            message: None,
                        }))
                    }
                }
            } else {
                let error_str = String::from_utf8_lossy(&output.stderr);
                println!("Python analysis failed: {}", error_str);
                
                Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    message: None,
                }))
            }
        }
        Err(e) => {
            println!("Failed to execute Python analysis: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                message: None,
            }))
        }
    }
}

pub async fn get_statistics_summary(
    State(pool): State<PgPool>,
) -> Result<Json<ApiResponse<HashMap<String, serde_json::Value>>>, StatusCode> {
    // Get basic statistics
    let mut stats = HashMap::new();

    // Count total measurements
    let measurement_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM water_quality_data")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    stats.insert("total_measurements".to_string(), serde_json::Value::Number(measurement_count.into()));

    // Count stations
    let station_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(DISTINCT station_name) FROM water_quality_data")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    stats.insert("total_stations".to_string(), serde_json::Value::Number(station_count.into()));

    Ok(Json(ApiResponse {
        success: true,
        data: Some(stats),
        message: None,
    }))
}
