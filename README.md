# 🚌 RedBus Demand Forecasting

A comprehensive machine learning system for forecasting bus seat demand, built with modern MLOps practices and deployed using Streamlit.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Advanced Feature Engineering**: Temporal, holiday, route-specific, and booking pattern features
- **Ensemble Models**: XGBoost and LightGBM with weighted predictions
- **Interactive Web Interface**: Streamlit-based UI for easy model training and prediction
- **Docker Support**: Containerized deployment for easy scaling
- **Comprehensive Testing**: Unit tests and model validation
- **MLOps Ready**: Structured codebase following software engineering best practices

## 🏗️ Architecture

```
redbus-demand-forecasting/
├── src/redbus_forecasting/          # Core ML package
│   ├── data/                        # Data loading and preprocessing
│   ├── features/                    # Feature engineering modules
│   ├── models/                      # ML models and ensemble
│   └── utils/                       # Utilities and configuration
├── streamlit_app/                   # Web application
│   ├── app.py                       # Main Streamlit app
│   └── pages/                       # Multi-page components
├── tests/                           # Unit tests
├── config/                          # Configuration files
├── data/                            # Data storage
├── docs/                            # Documentation
└── docker/                          # Docker configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/redbus-demand-forecasting.git
   cd redbus-demand-forecasting
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Running the Application

#### Local Development
```bash
streamlit run streamlit_app/app.py
```

#### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t redbus-forecasting .
docker run -p 8501:8501 redbus-forecasting
```

The application will be available at `http://localhost:8501`

## 📊 Data Requirements

The system expects three CSV files:

1. **train.csv**: Training data with historical demand
   - `srcid`: Source city ID
   - `destid`: Destination city ID
   - `doj`: Date of journey
   - `final_seatcount`: Target variable (seats sold)

2. **test.csv**: Test data for predictions
   - Same structure as train.csv but without `final_seatcount`

3. **transactions.csv**: Booking transaction data
   - `srcid`, `destid`, `doj`: Route and date information
   - `doi`: Date of inquiry/booking
   - `dbd`: Days before departure
   - `cumsum_seatcount`: Cumulative seats booked
   - `cumsum_searchcount`: Cumulative searches

## 🔧 Usage

### Web Interface

1. **Data Upload**: Upload your CSV files through the web interface
2. **Feature Engineering**: Generate temporal, holiday, route, and booking pattern features
3. **Model Training**: Train XGBoost, LightGBM, or ensemble models
4. **Predictions**: Generate predictions and download results

### Programmatic Usage

```python
from redbus_forecasting.data.data_loader import DataLoader
from redbus_forecasting.features.feature_engineering import *
from redbus_forecasting.models.forecasting_models import ModelManager

# Load data
loader = DataLoader()
train_df, test_df, transactions_df = loader.load_data()

# Engineer features
temporal_engineer = TemporalFeatureEngineer('doj')
holiday_engineer = HolidayFeatureEngineer('doj', 'India')
route_engineer = RouteFeatureEngineer(transactions_df)

# Fit and transform
train_features = temporal_engineer.fit_transform(train_df)
train_features = holiday_engineer.fit_transform(train_features)
train_features = route_engineer.fit_transform(train_features)

# Train model
model_manager = ModelManager()
ensemble = model_manager.train_ensemble(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/redbus_forecasting

# Run specific test file
pytest tests/test_models.py
```

## 📈 Model Performance

The system implements several advanced techniques:

- **Feature Engineering**: 25+ engineered features including temporal patterns, holidays, and booking velocities
- **Ensemble Learning**: Combines XGBoost and LightGBM with optimized weights
- **Cross-Validation**: Time series split validation for robust evaluation
- **Hyperparameter Tuning**: Configurable model parameters via YAML

Expected performance metrics:
- RMSE: < 15 seats (depending on data quality)
- MAE: < 10 seats
- R²: > 0.85

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
models:
  xgboost:
    n_estimators: 5000
    max_depth: 10
    learning_rate: 0.1

ensemble:
  weights:
    xgb: 0.6
    lgb: 0.4
```

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build production image
docker build -t redbus-forecasting:latest .

# Run with environment variables
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  --name redbus-app \
  redbus-forecasting:latest
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redbus-forecasting
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redbus-forecasting
  template:
    metadata:
      labels:
        app: redbus-forecasting
    spec:
      containers:
      - name: redbus-forecasting
        image: redbus-forecasting:latest
        ports:
        - containerPort: 8501
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [RedBus](https://www.redbus.in/) for the hackathon dataset
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/) teams

## 📞 Support

For questions and support:

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/redbus-demand-forecasting/issues)
- 📖 Documentation: [Full Documentation](docs/README.md)

---

**Built with ❤️ for better public transportation forecasting**
