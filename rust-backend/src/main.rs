use axum::{
    extract::State,
    routing::get,
    Router,
};
use sqlx::Row;
use sqlx::PgPool;
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use tracing::info;
use std::collections::HashMap;

mod api;
mod models;
mod db;
mod ml_bridge;

use api::pollution;
use api::forecast;
use api::enhanced_data_collection;

#[derive(Debug, serde::Serialize)]
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Initialize database pool
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://pollution_user:pollution_pass@localhost:5432/pollution_db".to_string());
    
    let pool = PgPool::connect(&database_url).await?;

    // Build application routes
    let app = Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/test-db", get(test_db))
        
        // Pollution data endpoints
        .route("/api/pollution/stations", get(pollution::get_stations))
        .route("/api/pollution/measurements", get(pollution::get_measurements))
        .route("/api/pollution/statistics", get(pollution::get_statistics))
        
        // Enhanced data collection endpoints
        .route("/api/data/collect", axum::routing::post(enhanced_data_collection::start_collection))
        .route("/api/data/status", axum::routing::get(enhanced_data_collection::get_status))
        .route("/api/areas", axum::routing::get(enhanced_data_collection::get_areas))
        .route("/api/basins", axum::routing::get(enhanced_data_collection::get_basins))
        .route("/api/stations", axum::routing::get(enhanced_data_collection::get_stations))
        
        // Analysis endpoints
        .route("/api/analysis/:analysis_type", axum::routing::get(api::analysis::get_analysis_results))
        
        // Forecasting endpoints
        .route("/api/forecasts", axum::routing::get(forecast::get_forecasts))
        .route("/api/forecasts/:id", axum::routing::get(forecast::get_forecast_by_id))
        .route("/api/forecast/generate", axum::routing::post(forecast::generate_forecast))
        .route("/api/forecast/list", axum::routing::get(forecast::get_forecast_list))
        
        // Map visualization endpoints
        .route("/api/map", axum::routing::get(api::map::generate_map))
        .route("/api/dashboard", axum::routing::get(api::map::generate_dashboard))
        
        // Add middleware
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        )
        .layer(TraceLayer::new_for_http())
        .with_state(pool);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    info!("Server running on http://0.0.0.0:8080");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> axum::response::Json<ApiResponse<HashMap<String, String>>> {
    let mut health_data = HashMap::new();
    health_data.insert("status".to_string(), "healthy".to_string());
    health_data.insert("version".to_string(), "1.0.0".to_string());
    health_data.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
    
    axum::response::Json(ApiResponse::success(health_data))
}

async fn test_db(State(pool): State<PgPool>) -> axum::response::Json<ApiResponse<HashMap<String, String>>> {
    let result = sqlx::query("SELECT current_database(), current_user, version()")
        .fetch_one(&pool)
        .await;
    
    match result {
        Ok(row) => {
            let mut db_info = HashMap::new();
            db_info.insert("database".to_string(), row.get::<String, _>("current_database"));
            db_info.insert("user".to_string(), row.get::<String, _>("current_user"));
            db_info.insert("version".to_string(), row.get::<String, _>("version"));
            axum::response::Json(ApiResponse::success(db_info))
        },
        Err(e) => {
            axum::response::Json(ApiResponse::error(format!("Database error: {}", e)))
        }
    }
}
