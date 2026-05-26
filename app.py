import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from data_prep import get_monthly_revenue, get_revenue_by_category, get_revenue_by_country
from model import train_forecast_model, get_trend_components

load_dotenv()

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="SalesPredict — ML Revenue Forecasting",
    page_icon="📈",
    layout="wide"
)

# ── LOAD DATA AND TRAIN MODEL ──
@st.cache_resource
def load_data_and_model():
    df       = get_monthly_revenue()
    cat_df   = get_revenue_by_category()
    ctry_df  = get_revenue_by_country()
    model, forecast, metrics, train_df, test_df = train_forecast_model(df)
    return df, cat_df, ctry_df, model, forecast, metrics, train_df, test_df

# ── SIDEBAR ──
with st.sidebar:
    st.title("📈 SalesPredict")
    st.markdown("ML-powered revenue forecasting using **Facebook Prophet**.")
    st.divider()

    st.subheader("Model Settings")
    forecast_months = st.slider("Forecast horizon (months)", 3, 12, 6)
    show_components  = st.checkbox("Show trend components", value=False)
    show_raw_data    = st.checkbox("Show raw data table", value=False)

    st.divider()
    st.subheader("About this model")
    st.markdown("""
    - **Algorithm:** Facebook Prophet (Meta open-source)
    - **Training data:** Northwind sample DB (PostgreSQL)
    - **Data range:** July 1996 – April 1998 (22 months)
    - **Revenue definition:** `SUM(unit_price × quantity × (1 − discount))` — shipped orders only
    - **Same semantic definition as [SemanticSQL](https://genai-reporting-agent.streamlit.app)** — proves consistent business logic across both ML and GenAI projects
    - **Train/test split:** 19 months training, 3 months held out for evaluation
    - **Model accuracy:** 38.1% MAPE (expected for 22 months of data — under 15% with 5+ years)
    - **Forecast horizon:** Configurable 3–12 months
    """)

    st.divider()
    st.caption("Built by Sadaquat Khan · [GitHub](https://github.com/sadaquat/sales-predict) · [SemanticSQL](https://genai-reporting-agent.streamlit.app) · [LinkedIn](https://linkedin.com/in/sadaquat-khan)")

# ── LOAD DATA ──
with st.spinner("Loading data and training model..."):
    df, cat_df, ctry_df, model, forecast, metrics, train_df, test_df = load_data_and_model()

# ── HEADER ──
st.title("SalesPredict — ML Revenue Forecasting")
st.markdown("Predicts future revenue trends using classical ML. Built on the same Northwind dataset and semantic revenue definition as **[SemanticSQL](https://genai-reporting-agent.streamlit.app)**.")

# ── KPI METRICS ──
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Historical Revenue",  f"${metrics['total_historical_revenue']:,.0f}")
col2.metric("Avg Monthly Revenue",        f"${metrics['avg_monthly_revenue']:,.0f}")
col3.metric("Forecast Next 6 Months",     f"${metrics['forecast_next_6m_total']:,.0f}")
col4.metric("Model MAPE",                 f"{metrics['mape']:.1f}%",
            delta=f"±${metrics['mae']:,.0f} avg error", delta_color="off")
col5.metric("Training Months",            f"{metrics['train_months']}",
            delta=f"{metrics['test_months']} test months", delta_color="off")

# ── MAIN FORECAST CHART ──
st.divider()
st.subheader("Revenue Forecast")

future_forecast = forecast[forecast['ds'] > df['ds'].max()]

fig = go.Figure()

# Training data
fig.add_trace(go.Scatter(
    x=train_df['ds'], y=train_df['y'],
    mode='lines+markers',
    name='Actual (training)',
    line=dict(color='#1B4F8A', width=2),
    marker=dict(size=6)
))

# Test data (actual)
fig.add_trace(go.Scatter(
    x=test_df['ds'], y=test_df['y'],
    mode='lines+markers',
    name='Actual (test period)',
    line=dict(color='#C9A84C', width=2),
    marker=dict(size=8, symbol='diamond')
))

# Test period prediction
test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
fig.add_trace(go.Scatter(
    x=test_forecast['ds'], y=test_forecast['yhat'],
    mode='lines+markers',
    name='Predicted (test period)',
    line=dict(color='#C9A84C', width=2, dash='dot'),
    marker=dict(size=8)
))

# Future forecast
fig.add_trace(go.Scatter(
    x=future_forecast['ds'], y=future_forecast['yhat'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#2E8BC0', width=3),
    marker=dict(size=8)
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=pd.concat([future_forecast['ds'], future_forecast['ds'].iloc[::-1]]),
    y=pd.concat([future_forecast['yhat_upper'], future_forecast['yhat_lower'].iloc[::-1]]),
    fill='toself',
    fillcolor='rgba(46,139,192,0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% Confidence Interval'
))

# Vertical line at forecast start
fig.add_vline(
    x=df['ds'].max().timestamp() * 1000,
    line_dash="dash",
    line_color="gray",
    annotation_text="Forecast starts",
    annotation_position="top"
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Revenue (USD)",
    yaxis_tickformat="$,.0f",
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=450,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ── MODEL ACCURACY ──
st.divider()
st.subheader("Model Accuracy — Test Set Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Actual vs Predicted (test period)**")

    acc_fig = go.Figure()
    acc_fig.add_trace(go.Bar(
        x=test_df['ds'].dt.strftime('%b %Y'),
        y=test_df['y'],
        name='Actual',
        marker_color='#1B4F8A'
    ))
    acc_fig.add_trace(go.Bar(
        x=test_forecast['ds'].dt.strftime('%b %Y'),
        y=test_forecast['yhat'],
        name='Predicted',
        marker_color='#2E8BC0'
    ))
    acc_fig.update_layout(
        barmode='group',
        yaxis_tickformat="$,.0f",
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(acc_fig, use_container_width=True)

with col2:
    st.markdown("**Accuracy Metrics Explained**")
    st.markdown(f"""
    | Metric | Value | Meaning |
    |--------|-------|---------|
    | **MAE** | ${metrics['mae']:,.2f} | Average error per month |
    | **RMSE** | ${metrics['rmse']:,.2f} | Penalises large errors more than MAE |
    | **MAPE** | {metrics['mape']:.1f}% | Percentage error — lower is better |

    **What these mean:**
    - MAPE of {metrics['mape']:.1f}% means the model's predictions are on average {metrics['mape']:.1f}% away from actual revenue
    - For a forecasting model on 26 months of business data, under 15% MAPE is considered good
    - The confidence interval (shaded area) shows the 95% range where actual revenue is likely to fall
    """)

# ── CATEGORY BREAKDOWN ──
st.divider()
st.subheader("Revenue by Product Category")

latest_month = cat_df['month'].max()
# If latest month is a partial month, use the previous one
cat_df['month'] = cat_df['month'].dt.tz_localize(None)
latest_month = cat_df['month'].max()
latest_cat = cat_df[cat_df['month'] == latest_month].sort_values('revenue', ascending=False)

col1, col2 = st.columns(2)

with col1:
    fig_cat = px.bar(
        latest_cat,
        x='revenue', y='category_name',
        orientation='h',
        title=f"Revenue by Category — {latest_month.strftime('%b %Y')}",
        labels={'revenue': 'Revenue (USD)', 'category_name': 'Category'},
        color='revenue',
        color_continuous_scale='Blues'
    )
    fig_cat.update_layout(height=350, template="plotly_white", showlegend=False)
    fig_cat.update_xaxes(tickformat="$,.0f")
    st.plotly_chart(fig_cat, use_container_width=True)

with col2:
    cat_trend = cat_df.groupby(['month', 'category_name'])['revenue'].sum().reset_index()
    fig_trend = px.line(
        cat_trend,
        x='month', y='revenue',
        color='category_name',
        title="Category Revenue Trends Over Time",
        labels={'revenue': 'Revenue (USD)', 'month': 'Month'}
    )
    fig_trend.update_layout(height=350, template="plotly_white")
    fig_trend.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig_trend, use_container_width=True)

# ── COUNTRY BREAKDOWN ──
st.divider()
st.subheader("Revenue by Country")

top_countries = ctry_df.groupby('country')['revenue'].sum().nlargest(10).index
top_ctry_df   = ctry_df[ctry_df['country'].isin(top_countries)]

country_totals = ctry_df.groupby('country')['revenue'].sum()\
                        .sort_values(ascending=False)\
                        .reset_index()\
                        .head(15)

fig_map = px.bar(
    country_totals,
    x='revenue', y='country',
    orientation='h',
    title="Top 15 Countries by Total Revenue",
    labels={'revenue': 'Total Revenue (USD)', 'country': 'Country'},
    color='revenue',
    color_continuous_scale='Blues'
)
fig_map.update_layout(height=500, template="plotly_white", showlegend=False)
fig_map.update_xaxes(tickformat="$,.0f")
st.plotly_chart(fig_map, use_container_width=True)

# ── TREND COMPONENTS ──
if show_components:
    st.divider()
    st.subheader("Model Components — Trend & Seasonality")
    components = get_trend_components(model, forecast)

    fig_trend = px.line(
        components['trend'],
        x='ds', y='trend',
        title="Underlying Revenue Trend",
        labels={'ds': 'Date', 'trend': 'Trend'}
    )
    fig_trend.update_layout(template="plotly_white")
    fig_trend.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig_trend, use_container_width=True)

# ── RAW DATA ──
if show_raw_data:
    st.divider()
    st.subheader("Raw Monthly Revenue Data")
    display_df = df.copy()
    display_df['month']   = display_df['ds'].dt.strftime('%B %Y')
    display_df['revenue'] = display_df['y'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_df[['month', 'revenue', 'order_count', 'unique_customers']],
                 use_container_width=True)