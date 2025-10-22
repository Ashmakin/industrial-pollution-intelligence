# 🏭 Industrial Pollution Intelligence System

A comprehensive, scientifically rigorous industrial pollution analysis and visualization system for advanced environmental monitoring and prediction.

## 🌟 Features

### 🔬 Advanced Analytics
- **Machine Learning Models**: LSTM, CNN-LSTM, Transformer networks using PyTorch
- **Time Series Analysis**: SARIMAX, Prophet, statistical forecasting
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Causal Inference**: Granger causality testing, Difference-in-Differences
- **Anomaly Detection**: Isolation Forest, One-Class SVM, Statistical Outliers
- **Network Analysis**: Pollution propagation networks, community detection

### 📊 Real-time Data Collection
- **CNEMC API Integration**: Automated data collection from Chinese environmental monitoring
- **Smart Deduplication**: Prevents duplicate data storage
- **Scheduled Collection**: Automatic data updates every 4 hours
- **Multi-region Support**: Coverage of all Chinese provinces and major cities

### 🗺️ Advanced Visualizations
- **Interactive China Map**: D3.js choropleth maps with pollution level visualization
- **Time Series Charts**: Interactive forecasting and trend analysis
- **Network Graphs**: Pollution correlation and causality networks
- **Radar Charts**: Multi-dimensional parameter analysis

### 🏗️ High-Performance Architecture
- **Backend**: Rust + Axum for ultra-fast API responses
- **Frontend**: React + TypeScript + Vite for modern UI
- **Database**: PostgreSQL with optimized indexing
- **ML Pipeline**: Python with PyTorch/TensorFlow integration

## 🚀 Quick Start

### Prerequisites
- PostgreSQL 17+ (`psql-17` command available)
- Python 3.8+ with virtual environment
- Node.js 18+ and npm
- Rust 1.70+

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Rustindp
```

2. **Database Setup**
```bash
# Create database and user
psql-17 -c "CREATE DATABASE pollution_db;"
psql-17 -c "CREATE USER pollution_user WITH PASSWORD 'pollution_pass';"
psql-17 -c "GRANT ALL PRIVILEGES ON DATABASE pollution_db TO pollution_user;"

# Run migrations
psql-17 -h localhost -p 5432 -U pollution_user -d pollution_db -f migrations/001_initial_schema.sql
```

3. **Backend Setup**
```bash
cd rust-backend
cargo build --release
```

4. **Python Environment**
```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. **Frontend Setup**
```bash
cd frontend
npm install
npm run build
```

### Running the System

1. **Start Backend**
```bash
cd rust-backend
DATABASE_URL="postgres://pollution_user:pollution_pass@localhost:5432/pollution_db" ./target/release/pollution-intelligence-backend
```

2. **Start Frontend**
```bash
cd frontend
npm run dev
```

3. **Data Collection**
```bash
cd python
source venv/bin/activate
python3 enhanced_cnemc_collector.py collect 北京,上海,广东
```

4. **Automatic Data Collection**
```bash
cd python
source venv/bin/activate
python3 auto_data_scheduler.py
```

## 📁 Project Structure

```
Rustindp/
├── rust-backend/          # Rust + Axum backend
│   ├── src/
│   │   ├── api/          # API endpoints
│   │   ├── db/           # Database connection
│   │   ├── models/       # Data models
│   │   └── ml_bridge/    # ML integration
│   └── Cargo.toml
├── frontend/              # React + TypeScript frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── services/     # API services
│   │   └── utils/        # Utilities
│   └── package.json
├── python/               # Python ML and data processing
│   ├── ml/              # Machine learning models
│   ├── processing/      # Data processing
│   ├── lifecycle/       # Product lifecycle analysis
│   └── enhanced_ml_forecasting.py
├── migrations/           # Database migrations
├── reports/             # Analysis reports
└── analysis/            # Jupyter notebooks
```

## 🔧 API Endpoints

### Core APIs
- `GET /health` - System health check
- `GET /api/pollution/stations` - Get monitoring stations
- `GET /api/pollution/measurements` - Get water quality data
- `GET /api/map` - Generate China pollution map

### Data Collection
- `POST /api/data/collect` - Start data collection
- `GET /api/data/status` - Collection status
- `GET /api/areas` - Available areas
- `GET /api/basins` - Available basins
- `GET /api/stations` - Available stations

### Analysis
- `POST /api/analysis` - Run data analysis
- `POST /api/forecast` - Generate predictions
- `GET /api/reports` - Generate reports

## 🧪 Machine Learning Models

### Time Series Forecasting
- **LSTM Networks**: Deep learning for complex patterns
- **CNN-LSTM**: Hybrid models for spatial-temporal data
- **Transformer**: Attention-based sequence modeling
- **SARIMAX**: Statistical time series analysis
- **Prophet**: Facebook's forecasting tool

### Analysis Methods
- **Principal Component Analysis (PCA)**: Dimensionality reduction
- **Granger Causality**: Causal relationship detection
- **Anomaly Detection**: Outlier identification
- **Correlation Analysis**: Parameter relationships
- **Network Analysis**: Pollution propagation

## 📊 Data Sources

- **CNEMC API**: Chinese National Environmental Monitoring Center
- **Real-time Data**: 4-hour interval updates
- **Geographic Coverage**: All Chinese provinces and major cities
- **Parameters**: pH, dissolved oxygen, ammonia nitrogen, total phosphorus, etc.

## 🎯 Use Cases

### Environmental Monitoring
- Real-time water quality assessment
- Pollution trend analysis
- Early warning systems
- Regulatory compliance monitoring

### Research Applications
- Environmental impact studies
- Pollution source identification
- Climate change effects analysis
- Policy effectiveness evaluation

### Industrial Applications
- Manufacturing process optimization
- Environmental risk assessment
- Supply chain sustainability
- Corporate environmental reporting

## 🔬 Scientific Methodology

### Data Processing
1. **ETL Pipeline**: Extract, Transform, Load from multiple sources
2. **Quality Control**: Missing value imputation, outlier detection
3. **Feature Engineering**: Temporal features, lag variables
4. **Normalization**: Standard scaling, robust scaling

### Model Validation
- **Cross-validation**: Time series split validation
- **Performance Metrics**: RMSE, MAE, MAPE, R²
- **Statistical Tests**: ADF test, Ljung-Box test
- **Ensemble Methods**: Model averaging and stacking

### Visualization
- **Interactive Maps**: D3.js choropleth visualizations
- **Time Series**: Multi-parameter trend analysis
- **Network Graphs**: Causality and correlation networks
- **Statistical Plots**: Distribution, correlation matrices

## 🚀 Performance

- **Backend**: < 100ms API response times
- **Database**: Optimized queries with proper indexing
- **ML Models**: GPU acceleration support
- **Frontend**: Modern React with efficient rendering

## 📈 Future Enhancements

- [ ] Real-time streaming data integration
- [ ] Advanced deep learning models (GANs, VAEs)
- [ ] Mobile application development
- [ ] Cloud deployment with Kubernetes
- [ ] Advanced visualization with WebGL
- [ ] Integration with IoT sensors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support, please open an issue in the GitHub repository.

---

**Industrial Pollution Intelligence System**  
*Advanced Environmental Monitoring and Prediction Platform*

Built with ❤️ using Rust, React, Python, and modern ML technologies.