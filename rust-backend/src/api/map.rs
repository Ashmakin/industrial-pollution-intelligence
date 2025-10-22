use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use std::process::Command;

#[derive(Debug, Deserialize)]
pub struct MapQuery {
    pub parameter: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MapResponse {
    pub success: bool,
    pub data: Option<Value>,
    pub message: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DashboardResponse {
    pub success: bool,
    pub data: Option<Value>,
    pub message: Option<String>,
}

pub async fn generate_map(
    State(_pool): State<PgPool>,
    Query(params): Query<MapQuery>,
) -> Result<Json<MapResponse>, StatusCode> {
    let parameter = params.parameter.unwrap_or_else(|| "ph".to_string());
    
    println!("Generating map for parameter: {}", parameter);
    
        // 调用Python脚本生成复杂轮廓图地图
        let script_command = format!(
            "cd /Users/aphrodite/Desktop/Rustindp/python && source venv/bin/activate && python3 use_real_geojson_map.py {}",
            parameter
        );
    
    println!("Executing command: {}", script_command);
    
    let output = Command::new("bash")
        .arg("-c")
        .arg(&script_command)
        .output();
    
    match output {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("Python script output: {}", stdout);
                
                match serde_json::from_str::<Value>(&stdout) {
                    Ok(data) => Ok(Json(MapResponse {
                        success: true,
                        data: Some(data),
                        message: None,
                    })),
                    Err(e) => {
                        println!("JSON parse error: {}", e);
                        Ok(Json(MapResponse {
                            success: false,
                            data: None,
                            message: Some("Failed to parse map data".to_string()),
                        }))
                    }
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Python script error: {}", stderr);
                Ok(Json(MapResponse {
                    success: false,
                    data: None,
                    message: Some(format!("Script execution failed: {}", stderr)),
                }))
            }
        }
        Err(e) => {
            println!("Command execution error: {}", e);
            Ok(Json(MapResponse {
                success: false,
                data: None,
                message: Some(format!("Failed to execute script: {}", e)),
            }))
        }
    }
}

pub async fn generate_dashboard(
    State(_pool): State<PgPool>,
) -> Result<Json<DashboardResponse>, StatusCode> {
    println!("Generating dashboard data");
    
    // 调用Python脚本生成仪表盘数据
    let script_command = "cd /Users/aphrodite/Desktop/Rustindp/python && source venv/bin/activate && python3 china_map_visualization.py dashboard";
    
    println!("Executing command: {}", script_command);
    
    let output = Command::new("bash")
        .arg("-c")
        .arg(script_command)
        .output();
    
    match output {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("Python script output: {}", stdout);
                
                match serde_json::from_str::<Value>(&stdout) {
                    Ok(data) => Ok(Json(DashboardResponse {
                        success: true,
                        data: Some(data),
                        message: None,
                    })),
                    Err(e) => {
                        println!("JSON parse error: {}", e);
                        Ok(Json(DashboardResponse {
                            success: false,
                            data: None,
                            message: Some("Failed to parse dashboard data".to_string()),
                        }))
                    }
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Python script error: {}", stderr);
                Ok(Json(DashboardResponse {
                    success: false,
                    data: None,
                    message: Some(format!("Script execution failed: {}", stderr)),
                }))
            }
        }
        Err(e) => {
            println!("Command execution error: {}", e);
            Ok(Json(DashboardResponse {
                success: false,
                data: None,
                message: Some(format!("Failed to execute script: {}", e)),
            }))
        }
    }
}
