import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def train_forecast_model(df, forecast_months=6):
    """
    Train a Prophet forecasting model on monthly revenue data.
    Returns: model, forecast dataframe, and accuracy metrics.
    """

    # ── TRAIN / TEST SPLIT ──
    # Use last 3 months as test set, everything before as training
    # Northwind has ~26 months of data — we'll train on 23, test on 3
    test_size = 3
    
    df['ds'] = df['ds'].dt.tz_localize(None)
    train_df = df.iloc[:-test_size].copy()
    test_df  = df.iloc[-test_size:].copy()

    print(f"Training on {len(train_df)} months, testing on {len(test_df)} months")

    # ── BUILD PROPHET MODEL ──
    # Prophet is Facebook's open-source forecasting tool
    # It handles: seasonality, holidays, trend changes, missing data
    model = Prophet(
        seasonality_mode='multiplicative',   # Revenue growth is multiplicative (% changes)
        yearly_seasonality=True,             # Annual patterns (Q4 peaks etc.)
        weekly_seasonality=False,            # Monthly data — no weekly pattern
        daily_seasonality=False,             # Monthly data — no daily pattern
        changepoint_prior_scale=0.05,        # How flexible the trend line is (0.05 = moderate)
        interval_width=0.95                  # 95% confidence interval
    )

    # ── FIT THE MODEL ──
    model.fit(train_df[['ds', 'y']])

    # ── GENERATE FORECAST ──
    # Create future dates: test period + forecast period
    future_months = test_size + forecast_months
    future = model.make_future_dataframe(
        periods=future_months,
        freq='MS'  # MS = Month Start
    )
    forecast = model.predict(future)

    # ── CALCULATE ACCURACY ON TEST SET ──
    # Merge actual vs predicted for the test period
    test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]

    mae  = mean_absolute_error(test_df['y'], test_forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
    mape = np.mean(np.abs((test_df['y'].values - test_forecast['yhat'].values)
                          / test_df['y'].values)) * 100

    metrics = {
        'mae':          round(mae, 2),
        'rmse':         round(rmse, 2),
        'mape':         round(mape, 2),
        'train_months': len(train_df),
        'test_months':  len(test_df),
        'forecast_months': forecast_months,
        'total_historical_revenue': round(df['y'].sum(), 2),
        'avg_monthly_revenue':      round(df['y'].mean(), 2),
        'forecast_next_6m_total':   round(
            forecast[forecast['ds'] > df['ds'].max()]['yhat'].sum(), 2
        )
    }

    print(f"\nModel accuracy on test set ({test_size} months):")
    print(f"  MAE:  ${mae:,.2f}  (average error per month)")
    print(f"  RMSE: ${rmse:,.2f}  (penalises large errors more)")
    print(f"  MAPE: {mape:.1f}%   (percentage error)")

    return model, forecast, metrics, train_df, test_df


def get_trend_components(model, forecast):
    """Extract trend and seasonality components from the model."""
    components = {
        'trend':  forecast[['ds', 'trend']].copy(),
        'yearly': forecast[['ds', 'yearly']].copy() if 'yearly' in forecast.columns else None,
    }
    return components


if __name__ == "__main__":
    df = pd.read_csv("data/monthly_revenue.csv", parse_dates=['ds'])
    model, forecast, metrics, train_df, test_df = train_forecast_model(df)

    print(f"\nForecast for next 6 months: ${metrics['forecast_next_6m_total']:,.2f}")
    print(f"Average monthly revenue (historical): ${metrics['avg_monthly_revenue']:,.2f}")

    # Save forecast
    forecast.to_csv("data/forecast.csv", index=False)
    print("\nForecast saved to data/forecast.csv")