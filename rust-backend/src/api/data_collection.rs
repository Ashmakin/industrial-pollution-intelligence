use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Deserialize)]
pub struct DataCollectionRequest {
    pub areas: Option<Vec<String>>,
    pub basins: Option<Vec<String>>,
    pub stations: Option<Vec<String>>,
    pub max_records: Option<i32>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DataCollectionStatus {
    pub is_running: bool,
    pub progress: f32,
    pub total_records: i32,
    pub collected_records: i32,
    pub current_area: String,
    pub last_update: String,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: Option<String>,
    pub error: Option<String>,
}

pub fn router() -> Router<PgPool> {
    Router::new()
        .route("/api/data/collect", post(start_collection))
        .route("/api/data/status", get(get_status))
}

pub async fn start_collection(
    State(_pool): State<PgPool>,
    Json(request): Json<DataCollectionRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    println!("Starting data collection for areas: {:?}", request.areas);
    
    // 构建Python脚本命令
    let area_ids_str = if let Some(areas) = &request.areas {
        areas.join(" ")
    } else {
        "北京,上海,广东".to_string()
    };
    
    let max_records = request.max_records.unwrap_or(100);
    
    // 启动Python数据采集脚本
    let script_command = format!(
        "cd ../python && source venv/bin/activate && python3 real_cnemc_collector.py --area-ids {} --max-records {} --database-url postgres://pollution_user:pollution_pass@localhost:5432/pollution_db",
        area_ids_str, max_records
    );
    
    let output = Command::new("bash")
        .arg("-c")
        .arg(&script_command)
        .output();
    
    match output {
        Ok(output) => {
            if output.status.success() {
                let result_str = String::from_utf8_lossy(&output.stdout);
                println!("Data collection completed: {}", result_str);
                
                // 尝试解析结果
                if let Ok(result) = serde_json::from_str::<serde_json::Value>(&result_str) {
                    Ok(Json(ApiResponse {
                        success: true,
                        data: Some(serde_json::json!({
                            "collection_id": format!("coll_{}", chrono::Utc::now().timestamp()),
                            "estimated_duration": "2-5 minutes",
                            "areas": request.areas.as_ref().map(|a| a.len()).unwrap_or(0),
                            "result": result
                        })),
                        message: Some("Data collection completed successfully".to_string()),
                        error: None,
                    }))
                } else {
                    Ok(Json(ApiResponse {
                        success: true,
                        data: Some(serde_json::json!({
                            "collection_id": format!("coll_{}", chrono::Utc::now().timestamp()),
                            "estimated_duration": "2-5 minutes",
                            "areas": request.areas.as_ref().map(|a| a.len()).unwrap_or(0),
                            "raw_output": result_str
                        })),
                        message: Some("Data collection completed successfully".to_string()),
                        error: None,
                    }))
                }
            } else {
                let error_str = String::from_utf8_lossy(&output.stderr);
                println!("Data collection failed: {}", error_str);
                
                Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    message: None,
                    error: Some(format!("Data collection failed: {}", error_str)),
                }))
            }
        }
        Err(e) => {
            println!("Failed to start data collection: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                message: None,
                error: Some(format!("Failed to start data collection: {}", e)),
            }))
        }
    }
}

pub async fn get_status(
    State(pool): State<PgPool>,
) -> Result<Json<ApiResponse<DataCollectionStatus>>, StatusCode> {
    // 从数据库查询真实状态
    match sqlx::query_as::<_, (i64, String)>(
        "SELECT COUNT(*) as total_records, 
                STRING_AGG(DISTINCT province, ', ') as provinces 
         FROM water_quality_data 
         WHERE data_source = 'CNEMC_API'"
    )
    .fetch_one(&pool)
    .await
    {
        Ok((total_records, provinces)) => {
            let status = DataCollectionStatus {
                is_running: false,
                progress: 1.0,
                total_records: total_records as i32,
                collected_records: total_records as i32,
                current_area: provinces,
                last_update: chrono::Utc::now().to_rfc3339(),
                errors: vec![],
            };
            
            Ok(Json(ApiResponse {
                success: true,
                data: Some(status),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            println!("Failed to query database status: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                message: None,
                error: Some(format!("Failed to query database status: {}", e)),
            }))
        }
    }
}
