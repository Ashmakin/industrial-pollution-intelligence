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

#[derive(Debug, Deserialize)]
pub struct BasinQuery {
    pub area_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StationQuery {
    pub area_id: String,
}

pub fn router() -> Router<PgPool> {
    Router::new()
        .route("/api/data/collect", post(start_collection))
        .route("/api/data/status", get(get_status))
        .route("/api/areas", get(get_areas))
        .route("/api/basins", get(get_basins))
        .route("/api/stations", get(get_stations))
}

pub async fn start_collection(
    State(_pool): State<PgPool>,
    Json(request): Json<DataCollectionRequest>,
) -> Result<Json<ApiResponse<DataCollectionStatus>>, StatusCode> {
    println!("Starting enhanced data collection with request: {:?}", request);
    
    
    let areas_param = if let Some(areas) = &request.areas {
        areas.join(",")
    } else {
        "北京,上海,广东".to_string() 
    };
    
    
    let script_command = format!(
        "cd /Users/aphrodite/Desktop/Rustindp/python && source venv/bin/activate && python3 enhanced_cnemc_collector.py collect {}",
        areas_param
    );
    
    println!("Executing command: {}", script_command);
    
    let output = Command::new("bash")
        .arg("-c")
        .arg(&script_command)
        .output();
    
    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            
            if output.status.success() {
                println!("Enhanced data collection completed successfully");
                println!("Stdout: {}", stdout);
                
                
                let mut total_inserted = 0;
                let mut total_skipped = 0;
                
                if let Ok(result) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    if let Some(inserted) = result.get("total_inserted").and_then(|v| v.as_i64()) {
                        total_inserted = inserted as i32;
                    }
                    if let Some(skipped) = result.get("total_skipped").and_then(|v| v.as_i64()) {
                        total_skipped = skipped as i32;
                    }
                }
                
                Ok(Json(ApiResponse {
                    success: true,
                    data: Some(DataCollectionStatus {
                        is_running: false,
                        progress: 100.0,
                        total_records: total_inserted + total_skipped,
                        collected_records: total_inserted,
                        current_area: areas_param,
                        last_update: chrono::Utc::now().to_rfc3339(),
                        errors: vec![],
                    }),
                    message: Some(format!("数据采集完成，新增 {} 条记录，跳过 {} 条重复记录", total_inserted, total_skipped)),
                    error: None,
                }))
            } else {
                println!("Enhanced data collection failed: {}", stderr);
                Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    message: Some("数据采集失败".to_string()),
                    error: Some(stderr.to_string()),
                }))
            }
        },
        Err(e) => {
            println!("Failed to execute enhanced data collection script: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                message: Some("执行数据采集脚本失败".to_string()),
                error: Some(e.to_string()),
            }))
        }
    }
}

pub async fn get_status(
    State(pool): State<PgPool>,
) -> Result<Json<ApiResponse<DataCollectionStatus>>, StatusCode> {
    
    match sqlx::query_as::<_, (i64, String)>(
        "SELECT COUNT(*) as total_records, 
                STRING_AGG(DISTINCT province, ', ') as provinces 
         FROM water_quality_data 
         WHERE created_at >= NOW() - INTERVAL '1 hour'"
    )
    .fetch_one(&pool)
    .await
    {
        Ok((total_records, provinces)) => {
            let status = DataCollectionStatus {
                is_running: false,
                progress: 100.0,
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
            println!("Failed to query collection status: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                message: Some("查询采集状态失败".to_string()),
                error: Some(e.to_string()),
            }))
        }
    }
}

pub async fn get_areas(
    State(_pool): State<PgPool>,
) -> Result<Json<ApiResponse<Vec<HashMap<String, String>>>>, StatusCode> {
    
    let areas = vec![
        ("110000", "北京"),
        ("120000", "天津"),
        ("130000", "河北"),
        ("140000", "山西"),
        ("150000", "内蒙古"),
        ("210000", "辽宁"),
        ("220000", "吉林"),
        ("230000", "黑龙江"),
        ("310000", "上海"),
        ("320000", "江苏"),
        ("330000", "浙江"),
        ("340000", "安徽"),
        ("350000", "福建"),
        ("360000", "江西"),
        ("370000", "山东"),
        ("410000", "河南"),
        ("420000", "湖北"),
        ("430000", "湖南"),
        ("440000", "广东"),
        ("450000", "广西"),
        ("460000", "海南"),
        ("500000", "重庆"),
        ("510000", "四川"),
        ("520000", "贵州"),
        ("530000", "云南"),
        ("540000", "西藏"),
        ("610000", "陕西"),
        ("620000", "甘肃"),
        ("630000", "青海"),
        ("640000", "宁夏"),
        ("650000", "新疆"),
    ];
    
    let result: Vec<HashMap<String, String>> = areas
        .iter()
        .map(|(code, name)| {
            let mut map = HashMap::new();
            map.insert("code".to_string(), code.to_string());
            map.insert("name".to_string(), name.to_string());
            map
        })
        .collect();
    
    Ok(Json(ApiResponse {
        success: true,
        data: Some(result),
        message: None,
        error: None,
    }))
}

pub async fn get_basins(
    State(_pool): State<PgPool>,
) -> Result<Json<ApiResponse<Vec<HashMap<String, String>>>>, StatusCode> {
    
    let basins = vec![
        ("haihe", "海河流域"),
        ("yellow_river", "黄河流域"),
        ("yangtze", "长江流域"),
        ("pearl_river", "珠江流域"),
        ("songhua", "松花江流域"),
        ("liaohe", "辽河流域"),
        ("huaihe", "淮河流域"),
        ("taihu", "太湖流域"),
        ("chaohu", "巢湖流域"),
        ("dianchi", "滇池流域"),
        ("other", "其他"),
    ];
    
    let result: Vec<HashMap<String, String>> = basins
        .iter()
        .map(|(code, name)| {
            let mut map = HashMap::new();
            map.insert("code".to_string(), code.to_string());
            map.insert("name".to_string(), name.to_string());
            map
        })
        .collect();
    
    Ok(Json(ApiResponse {
        success: true,
        data: Some(result),
        message: None,
        error: None,
    }))
}

pub async fn get_stations(
    State(pool): State<PgPool>,
    Query(params): Query<StationQuery>,
) -> Result<Json<ApiResponse<Vec<HashMap<String, String>>>>, StatusCode> {
    
    match sqlx::query_as::<_, (String, String)>(
        "SELECT DISTINCT station_name, basin 
         FROM water_quality_data 
         WHERE province = (
             SELECT name FROM (
                 VALUES 
                 ('110000', '北京'), ('120000', '天津'), ('130000', '河北'), ('140000', '山西'), ('150000', '内蒙古'),
                 ('210000', '辽宁'), ('220000', '吉林'), ('230000', '黑龙江'), ('310000', '上海'), ('320000', '江苏'),
                 ('330000', '浙江'), ('340000', '安徽'), ('350000', '福建'), ('360000', '江西'), ('370000', '山东'),
                 ('410000', '河南'), ('420000', '湖北'), ('430000', '湖南'), ('440000', '广东'), ('450000', '广西'),
                 ('460000', '海南'), ('500000', '重庆'), ('510000', '四川'), ('520000', '贵州'), ('530000', '云南'),
                 ('540000', '西藏'), ('610000', '陕西'), ('620000', '甘肃'), ('630000', '青海'), ('640000', '宁夏'), ('650000', '新疆')
             ) AS area_mapping(code, name)
             WHERE code = $1
         )
         ORDER BY station_name"
    )
    .bind(&params.area_id)
    .fetch_all(&pool)
    .await
    {
        Ok(rows) => {
            let result: Vec<HashMap<String, String>> = rows
                .iter()
                .map(|(station_name, basin)| {
                    let mut map = HashMap::new();
                    map.insert("name".to_string(), station_name.clone());
                    map.insert("basin".to_string(), basin.clone());
                    map.insert("area_id".to_string(), params.area_id.clone());
                    map
                })
                .collect();
            
            Ok(Json(ApiResponse {
                success: true,
                data: Some(result),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            println!("Failed to query stations: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                message: Some("查询监测站失败".to_string()),
                error: Some(e.to_string()),
            }))
        }
    }
}
