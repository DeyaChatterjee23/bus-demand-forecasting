"""
Main Streamlit application for RedBus Demand Forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from redbus_forecasting.data.data_loader import DataLoader
from redbus_forecasting.features.feature_engineering import (
    TemporalFeatureEngineer, HolidayFeatureEngineer, 
    RouteFeatureEngineer, BookingPatternFeatureEngineer, CategoricalEncoder
)
from redbus_forecasting.models.forecasting_models import ModelManager, EnsembleModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="RedBus Demand Forecasting",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #DC143C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'features_engineered' not in st.session_state:
        st.session_state.features_engineered = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False

def load_data_section():
    """Data loading section."""
    st.markdown('<h2 class="sub-header">📊 Data Loading</h2>', unsafe_allow_html=True)

    # File upload option
    uploaded_files = st.file_uploader(
        "Upload your CSV files",
        accept_multiple_files=True,
        type=['csv'],
        help="Upload train.csv, test.csv, and transactions.csv files"
    )

    if uploaded_files and len(uploaded_files) >= 3:
        try:
            # Process uploaded files
            file_dict = {}
            for file in uploaded_files:
                if 'train' in file.name.lower():
                    file_dict['train'] = pd.read_csv(file)
                elif 'test' in file.name.lower():
                    file_dict['test'] = pd.read_csv(file)
                elif 'transaction' in file.name.lower():
                    file_dict['transactions'] = pd.read_csv(file)

            if len(file_dict) >= 3:
                st.session_state.train_df = file_dict['train']
                st.session_state.test_df = file_dict['test']
                st.session_state.transactions_df = file_dict['transactions']
                st.session_state.data_loaded = True

                st.success("✅ Data loaded successfully!")

                # Display data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train Records", len(st.session_state.train_df))
                with col2:
                    st.metric("Test Records", len(st.session_state.test_df))
                with col3:
                    st.metric("Transaction Records", len(st.session_state.transactions_df))

                # Data preview
                if st.checkbox("Show data preview"):
                    st.subheader("Train Data Preview")
                    st.dataframe(st.session_state.train_df.head())

                    st.subheader("Test Data Preview") 
                    st.dataframe(st.session_state.test_df.head())

                    st.subheader("Transactions Data Preview")
                    st.dataframe(st.session_state.transactions_df.head())

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    elif not st.session_state.data_loaded:
        st.info("Please upload train.csv, test.csv, and transactions.csv files to proceed.")

def feature_engineering_section():
    """Feature engineering section."""
    if not st.session_state.data_loaded:
        st.warning("Please load data first.")
        return

    st.markdown('<h2 class="sub-header">🔧 Feature Engineering</h2>', unsafe_allow_html=True)

    if st.button("Generate Features", type="primary"):
        try:
            with st.spinner("Engineering features..."):
                # Initialize feature engineers
                temporal_engineer = TemporalFeatureEngineer('doj')
                holiday_engineer = HolidayFeatureEngineer('doj', 'India')
                route_engineer = RouteFeatureEngineer(st.session_state.transactions_df)
                booking_engineer = BookingPatternFeatureEngineer(st.session_state.transactions_df)

                # Fit on train data
                temporal_engineer.fit(st.session_state.train_df)
                holiday_engineer.fit(st.session_state.train_df)
                route_engineer.fit(st.session_state.train_df)
                booking_engineer.fit(st.session_state.train_df)

                # Transform train data
                train_features = st.session_state.train_df.copy()
                train_features = temporal_engineer.transform(train_features)
                train_features = holiday_engineer.transform(train_features)
                train_features = route_engineer.transform(train_features)
                train_features = booking_engineer.transform(train_features)

                # Transform test data
                test_features = st.session_state.test_df.copy()
                test_features = temporal_engineer.transform(test_features)
                test_features = holiday_engineer.transform(test_features)
                test_features = route_engineer.transform(test_features)
                test_features = booking_engineer.transform(test_features)

                # Encode categorical features
                categorical_cols = ['tier_combination', 'region_combination']
                encoder = CategoricalEncoder(categorical_cols)
                encoder.fit(train_features)

                train_features = encoder.transform(train_features)
                test_features = encoder.transform(test_features)

                # Store engineered features
                st.session_state.train_features = train_features
                st.session_state.test_features = test_features
                st.session_state.features_engineered = True

                st.success("✅ Feature engineering completed!")

                # Show feature statistics
                feature_cols = [col for col in train_features.columns if col.endswith('_encoded') or 
                               any(prefix in col for prefix in ['doj_', 'cumsum_', 'booking_', 'conversion_'])]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Features", len(train_features.columns))
                with col2:
                    st.metric("Engineered Features", len(feature_cols))

                if st.checkbox("Show feature preview"):
                    st.dataframe(train_features[feature_cols].head())

        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
            logger.exception("Feature engineering failed")

def model_training_section():
    """Model training section."""
    if not st.session_state.features_engineered:
        st.warning("Please complete feature engineering first.")
        return

    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)

    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Ensemble (XGBoost + LightGBM)", "XGBoost Only", "LightGBM Only"]
    )

    if st.button("Train Model", type="primary"):
        try:
            with st.spinner("Training model..."):
                # Prepare training data
                exclude_cols = ['doj', 'final_seatcount', 'route_key']
                feature_cols = [col for col in st.session_state.train_features.columns 
                               if col not in exclude_cols and 
                               st.session_state.train_features[col].dtype in ['int64', 'float64']]

                X_train = st.session_state.train_features[feature_cols].fillna(0)
                y_train = st.session_state.train_features['final_seatcount']

                # Initialize model manager
                model_manager = ModelManager()

                if model_type == "Ensemble (XGBoost + LightGBM)":
                    model = model_manager.train_ensemble(X_train, y_train)
                else:
                    models = model_manager.create_models()
                    if "XGBoost" in model_type:
                        model = models['xgb']
                    else:
                        model = models['lgb']
                    model.fit(X_train, y_train)

                # Store model and feature columns
                st.session_state.model = model
                st.session_state.feature_cols = feature_cols
                st.session_state.model_trained = True

                # Evaluate model
                train_pred = model.predict(X_train)
                rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
                mae = np.mean(np.abs(y_train - train_pred))

                st.success("✅ Model training completed!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training RMSE", f"{rmse:.2f}")
                with col2:
                    st.metric("Training MAE", f"{mae:.2f}")
                with col3:
                    st.metric("Model Type", model_type)

                # Feature importance (if available)
                if hasattr(model, 'get_feature_importance'):
                    importance_df = model.get_feature_importance()

                    fig = px.bar(
                        importance_df.head(15), 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title="Top 15 Feature Importances"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            logger.exception("Model training failed")

def prediction_section():
    """Prediction section."""
    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
        return

    st.markdown('<h2 class="sub-header">🔮 Predictions</h2>', unsafe_allow_html=True)

    if st.button("Generate Predictions", type="primary"):
        try:
            with st.spinner("Generating predictions..."):
                # Prepare test data
                X_test = st.session_state.test_features[st.session_state.feature_cols].fillna(0)

                # Generate predictions
                predictions = st.session_state.model.predict(X_test)

                # Create submission dataframe
                submission_df = pd.DataFrame({
                    'route_key': st.session_state.test_df.get('route_key', range(len(predictions))),
                    'final_seatcount': predictions
                })

                st.session_state.predictions = predictions
                st.session_state.submission_df = submission_df
                st.session_state.predictions_made = True

                st.success("✅ Predictions generated successfully!")

                # Prediction statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Prediction", f"{predictions.min():.2f}")
                with col2:
                    st.metric("Max Prediction", f"{predictions.max():.2f}")
                with col3:
                    st.metric("Mean Prediction", f"{predictions.mean():.2f}")

                # Prediction distribution
                fig = px.histogram(
                    predictions,
                    nbins=50,
                    title="Distribution of Predictions"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download submission
                csv = submission_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Submission CSV",
                    data=csv,
                    file_name="redbus_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            logger.exception("Prediction generation failed")

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🚌 RedBus Demand Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("📋 Navigation")

    # Progress tracking
    progress_data = {
        "Data Loading": st.session_state.data_loaded,
        "Feature Engineering": st.session_state.features_engineered,
        "Model Training": st.session_state.model_trained,
        "Predictions": st.session_state.predictions_made
    }

    for step, completed in progress_data.items():
        icon = "✅" if completed else "⏳"
        st.sidebar.write(f"{icon} {step}")

    st.sidebar.markdown("---")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Loading", 
        "🔧 Feature Engineering", 
        "🤖 Model Training", 
        "🔮 Predictions"
    ])

    with tab1:
        load_data_section()

    with tab2:
        feature_engineering_section()

    with tab3:
        model_training_section()

    with tab4:
        prediction_section()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "RedBus Demand Forecasting System | Built with Streamlit 🚌"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
