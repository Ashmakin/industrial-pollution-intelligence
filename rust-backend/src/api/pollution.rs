use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            message: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            message: Some(message),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct PollutionQuery {
    pub station_name: Option<String>,
    pub province: Option<String>,
    pub watershed: Option<String>,
    pub parameter: Option<String>,
    pub page: Option<i32>,
    pub limit: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct WaterQualityRecord {
    pub id: i64,
    pub station_name: String,
    pub station_code: Option<String>,
    pub province: Option<String>,
    pub watershed: Option<String>,
    pub monitoring_time: chrono::DateTime<chrono::Utc>,
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
    pub water_quality_grade: Option<i32>,
    pub pollution_index: Option<f64>,
    pub data_source: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct StationInfo {
    pub station_name: String,
    pub station_code: Option<String>,
    pub province: Option<String>,
    pub watershed: Option<String>,
    pub measurement_count: i64,
    pub latest_measurement_time: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct ParameterStatistics {
    pub parameter: String,
    pub count: i64,
    pub mean: Option<f64>,
    pub std_dev: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

pub async fn get_stations(
    State(pool): State<PgPool>,
    Query(params): Query<PollutionQuery>,
) -> Result<Json<ApiResponse<Vec<StationInfo>>>, StatusCode> {
    let limit = params.limit.unwrap_or(50).min(100);
    let offset = (params.page.unwrap_or(1) - 1) * limit;

    let stations = match sqlx::query_as::<_, StationInfo>(
        r#"
        SELECT 
            station_name,
            station_code,
            province,
            watershed,
            COUNT(*) as measurement_count,
            MAX(monitoring_time) as latest_measurement_time
        FROM water_quality_data
        WHERE ($1::text IS NULL OR station_name ILIKE '%' || $1 || '%')
          AND ($2::text IS NULL OR province = $2)
          AND ($3::text IS NULL OR watershed = $3)
        GROUP BY station_name, station_code, province, watershed
        ORDER BY station_name
        LIMIT $4 OFFSET $5
        "#
    )
    .bind(&params.station_name)
    .bind(&params.province)
    .bind(&params.watershed)
    .bind(limit as i64)
    .bind(offset as i64)
    .fetch_all(&pool)
    .await
    {
        Ok(stations) => stations,
        Err(e) => {
            eprintln!("Error fetching stations: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Json(ApiResponse::success(stations)))
}

pub async fn get_measurements(
    State(pool): State<PgPool>,
    Query(params): Query<PollutionQuery>,
) -> Result<Json<ApiResponse<Vec<WaterQualityRecord>>>, StatusCode> {
    let limit = params.limit.unwrap_or(100).min(1000);
    let offset = (params.page.unwrap_or(1) - 1) * limit;

    let measurements = match sqlx::query_as::<_, WaterQualityRecord>(
        r#"
        SELECT id, station_name, station_code, province, watershed, monitoring_time,
               temperature, ph, dissolved_oxygen, conductivity, turbidity,
               permanganate_index, ammonia_nitrogen, total_phosphorus, total_nitrogen,
               chlorophyll_a, algae_density, water_quality_grade, pollution_index,
               data_source, created_at, updated_at
        FROM water_quality_data
        WHERE ($1::text IS NULL OR station_name ILIKE '%' || $1 || '%')
          AND ($2::text IS NULL OR province = $2)
          AND ($3::text IS NULL OR watershed = $3)
        ORDER BY monitoring_time DESC
        LIMIT $4 OFFSET $5
        "#
    )
    .bind(&params.station_name)
    .bind(&params.province)
    .bind(&params.watershed)
    .bind(limit as i64)
    .bind(offset as i64)
    .fetch_all(&pool)
    .await
    {
        Ok(measurements) => measurements,
        Err(e) => {
            eprintln!("Error fetching measurements: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Json(ApiResponse::success(measurements)))
}

pub async fn get_statistics(
    State(pool): State<PgPool>,
    Query(params): Query<PollutionQuery>,
) -> Result<Json<ApiResponse<Vec<ParameterStatistics>>>, StatusCode> {
    let parameters = [
        "temperature", "ph", "dissolved_oxygen", "conductivity", "turbidity",
        "permanganate_index", "ammonia_nitrogen", "total_phosphorus", "total_nitrogen",
        "chlorophyll_a", "algae_density"
    ];

    let mut statistics = Vec::new();

    for parameter in &parameters {
        let stats = match sqlx::query_as::<_, ParameterStatistics>(
            &format!(
                r#"
                SELECT 
                    '{}' as parameter,
                    COUNT({}) as count,
                    AVG({}) as mean,
                    STDDEV({}) as std_dev,
                    MIN({}) as min,
                    MAX({}) as max
                FROM water_quality_data
                WHERE ($1::text IS NULL OR station_name ILIKE '%' || $1 || '%')
                  AND ($2::text IS NULL OR province = $2)
                  AND ($3::text IS NULL OR watershed = $3)
                  AND {} IS NOT NULL
                "#,
                parameter, parameter, parameter, parameter, parameter, parameter, parameter
            )
        )
        .bind(&params.station_name)
        .bind(&params.province)
        .bind(&params.watershed)
        .fetch_one(&pool)
        .await
        {
            Ok(stats) => stats,
            Err(e) => {
                eprintln!("Error fetching statistics for {}: {}", parameter, e);
                continue;
            }
        };

        statistics.push(stats);
    }

    Ok(Json(ApiResponse::success(statistics)))
}

pub async fn get_health() -> Result<Json<ApiResponse<HashMap<String, String>>>, StatusCode> {
    let mut health_data = HashMap::new();
    health_data.insert("status".to_string(), "healthy".to_string());
    health_data.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
    
    Ok(Json(ApiResponse::success(health_data)))
}