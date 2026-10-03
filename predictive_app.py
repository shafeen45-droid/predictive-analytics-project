import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Predictive Analytics Dashboard", layout="wide")

st.title("📈 Predictive Analytics & Trend Forecasting Dashboard")
st.markdown("Train a machine learning model on historical baseline parameters to forecast future demands.")

# Load historical data configuration
@st.cache_data
def load_historical_data():
    try:
        return pd.read_csv("historical_sales.csv")
    except FileNotFoundError:
        st.error("Error: 'historical_sales.csv' not found. Please upload it to your repository.")
        return None

df = load_historical_data()

if df is not None:
    # Sidebar options grid configuration
    st.sidebar.header("🔮 Forecasting Adjustments")
    forecast_months = st.sidebar.slider("Months to Forecast into Future", min_value=3, max_value=12, value=6)
    
    # Train Linear Regression Engine
    X = df[['Month_Index']].values
    y = df['Historical_Sales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate training operational metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Generate future indices for forecast timeline
    last_idx = df['Month_Index'].max()
    future_indices = np.arange(last_idx + 1, last_idx + 1 + forecast_months).reshape(-1, 1)
    future_preds = model.predict(future_indices)
    
    # Construct complete baseline forecast matrix for plotting rows
    future_months_names = [f"Forecast Month +{i}" for i in range(1, forecast_months + 1)]
    
    # Build core layout cards indicators
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Total Historical Data Points", f"{len(df)} Months")
    with kpi2:
        st.metric("Model Prediction Accuracy (R² Score)", f"{r2*100:.1f}%")
    with kpi3:
        st.metric("Next Month Projected Sales", f"${int(future_preds[0]):,}")
        
    st.markdown("---")
    
    # Dynamic Plotly Visualization canvas
    st.subheader("📊 Historical Trends vs. Predictive Projections")
    
    fig = go.Figure()
    # Actual trend line
    fig.add_trace(go.Scatter(x=df['Month_Index'], y=df['Historical_Sales'], mode='lines+markers', name='Actual Historical Sales', line=dict(color='#00CC96', width=3)))
    # Fitted line
    fig.add_trace(go.Scatter(x=df['Month_Index'], y=y_pred, mode='lines', name='Model Training Fit', line=dict(color='#636EFA', dash='dash')))
    # Future projections line
    future_x = list(range(last_idx + 1, last_idx + 1 + forecast_months))
    fig.add_trace(go.Scatter(x=future_x, y=future_preds, mode='lines+markers', name='Future Projected Forecast', line=dict(color='#EF553B', width=3)))
    
    fig.update_layout(xaxis_title="Consecutive Timeline Index (Months)", yaxis_title="Sales Metrics ($)", height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Forecast Matrix Grid outputs
    st.subheader("📋 Forecast Data Matrix View")
    forecast_df = pd.DataFrame({
        'Timeline Period': future_months_names,
        'Timeline Index': future_x,
        'Projected Revenue Target ($)': [f"${int(val):,}" for val in future_preds]
    })
    st.dataframe(forecast_df, use_container_width=True)

