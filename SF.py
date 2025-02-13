import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet  # If using an older version, use: from fbprophet import Prophet
from datetime import timedelta

# Set the Streamlit page configuration
st.set_page_config(page_title="Retail Sales Forecasting Dashboard", layout="wide")

st.title("Retail Sales Forecasting Dashboard")
st.markdown("This dashboard forecasts future retail sales using **ARIMA** and **Prophet** models.")

# ----------------------------
# Data Acquisition & Preprocessing
# ----------------------------
#@st.cache(allow_output_mutation=True)
def load_data():
    # Load the CSV file; adjust the file path if needed.
    df = pd.read_csv("train.csv", parse_dates=["Order Date"])
    # Convert 'Order Date' using the correct format (day/month/year in this example)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y", errors="coerce")
    df.dropna(subset=["Order Date"], inplace=True)
    df.set_index("Order Date", inplace=True)
    # Aggregate sales by day (if multiple orders per day)
    daily_sales = df["Sales"].resample("D").sum()
    # Create a complete date range and fill missing dates with 0 sales
    full_range = pd.date_range(start=daily_sales.index.min(), end=daily_sales.index.max(), freq="D")
    daily_sales = daily_sales.reindex(full_range, fill_value=0)
    return daily_sales

sales_ts = load_data()
st.subheader("Historical Daily Sales")
st.line_chart(sales_ts)

# ----------------------------
# Forecast Settings (Sidebar)
# ----------------------------
st.sidebar.header("Forecast Settings")
forecast_horizon = st.sidebar.slider("Select forecast horizon (days):", min_value=7, max_value=90, value=30, step=7)
model_choice = st.sidebar.radio("Select Forecast Model(s):", ("ARIMA", "Prophet", "Both"))

# ----------------------------
# Forecasting and Visualization
# ----------------------------
if model_choice == "Both":
    tab_arima, tab_prophet = st.tabs(["ARIMA Forecast", "Prophet Forecast"])
else:
    tab_arima = st.container()

# ---- ARIMA Forecast Section ----
with tab_arima:
    if model_choice in ("ARIMA", "Both"):
        st.subheader("ARIMA Forecast")
        try:
            # Fit ARIMA model on the entire sales time series
            model_arima = sm.tsa.ARIMA(sales_ts, order=(1,1,1))
            model_arima_fit = model_arima.fit()
            # Forecast future sales for the selected horizon
            forecast_arima = model_arima_fit.forecast(steps=forecast_horizon)
            forecast_index = pd.date_range(start=sales_ts.index[-1] + timedelta(days=1),
                                           periods=forecast_horizon, freq='D')
            forecast_arima_series = pd.Series(forecast_arima, index=forecast_index)
            
            # Plot historical sales and ARIMA forecast
            fig_arima, ax_arima = plt.subplots(figsize=(10,4))
            ax_arima.plot(sales_ts.index, sales_ts.values, label="Historical Sales")
            ax_arima.plot(forecast_arima_series.index, forecast_arima_series.values, 
                          label="ARIMA Forecast", color="red")
            ax_arima.set_title("ARIMA Sales Forecast")
            ax_arima.set_xlabel("Date")
            ax_arima.set_ylabel("Sales")
            ax_arima.legend()
            st.pyplot(fig_arima)
            
            # Display forecasted values in a table below the plot
            st.markdown("### Predicted Values and Dates (ARIMA)")
            # Reset index to turn the Series into a DataFrame
            arima_df = forecast_arima_series.reset_index()
            arima_df.columns = ["Date", "Predicted Sales"]
            st.dataframe(arima_df)
            
            st.markdown("**ARIMA Model Summary:**")
            st.text(model_arima_fit.summary())
        except Exception as e:
            st.error(f"ARIMA model error: {e}")

# ---- Prophet Forecast Section ----
if model_choice in ("Prophet", "Both"):
    if model_choice == "Prophet":
        container_prophet = st.container()
    else:
        container_prophet = tab_prophet

    with container_prophet:
        st.subheader("Prophet Forecast")
        try:
            # Prepare data for Prophet: DataFrame with 'ds' (date) and 'y' (sales)
            df_prophet = sales_ts.reset_index()
            df_prophet.columns = ['ds', 'y']
            
            # Fit the Prophet model
            model_prophet = Prophet()
            model_prophet.fit(df_prophet)
            
            # Create a future DataFrame and forecast
            future = model_prophet.make_future_dataframe(periods=forecast_horizon)
            forecast_prophet = model_prophet.predict(future)
            
            # Plot the Prophet forecast
            fig_prophet = model_prophet.plot(forecast_prophet)
            plt.title("Prophet Sales Forecast")
            st.pyplot(fig_prophet)
            
            # Display forecasted values for the future period
            st.markdown("### Predicted Values and Dates (Prophet)")
            # Filter forecast_prophet to future dates (after the last historical date)
            future_forecast = forecast_prophet[forecast_prophet['ds'] > sales_ts.index[-1]]
            # Show columns: ds (date), yhat (prediction), yhat_lower, yhat_upper
            st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True))
        except Exception as e:
            st.error(f"Prophet model error: {e}")
