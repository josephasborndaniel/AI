import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# --------------------------
# Data Loader Function
# --------------------------
def fetch_stock_data(symbol, start="2015-01-01", end="2024-01-01"):
    """Fetch historical stock data from Yahoo Finance."""
    return yf.download(symbol, start=start, end=end)

# --------------------------
# Load and Preprocess Data
# --------------------------
symbol = "AAPL"
data = fetch_stock_data(symbol)
df = data[["Close"]].copy()

# Save original values for ARIMA (ARIMA may perform better on original scale)
df_original = df.copy()

# Normalize data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df["Close"] = scaler.fit_transform(df)

# --------------------------
# Prepare Test Data for LSTM
# --------------------------
window_size = 60
test_data = df[-window_size:].values
test_data = np.reshape(test_data, (1, test_data.shape[0], 1))

# --------------------------
# Load the Trained LSTM Model
# --------------------------
lstm_model = load_model("models/stock_lstm.h5")

# LSTM prediction (in normalized scale, then inverse transform)
lstm_pred_norm = lstm_model.predict(test_data)
lstm_pred = scaler.inverse_transform(lstm_pred_norm)
print(f"LSTM Predicted Next Day Price: ${lstm_pred[0][0]:.2f}")

# --------------------------
# Train ARIMA Model on Original Data
# --------------------------
# Use the original 'Close' price (non-normalized) for ARIMA modeling.
arima_data = df_original["Close"].values

# For simplicity, we choose an ARIMA order (p,d,q) of (5,1,0). 
# (In practice, you would use ACF/PACF plots or automated selection methods.)
arima_model = ARIMA(arima_data, order=(5, 1, 0))
arima_result = arima_model.fit()

# Forecast the next day using ARIMA
arima_forecast = arima_result.forecast(steps=1)
print(f"ARIMA Predicted Next Day Price: ${arima_forecast[0]:.2f}")

# --------------------------
# Combine Predictions (Ensemble)
# --------------------------
# For example, using an equal weight average.
ensemble_pred = (lstm_pred[0][0] + arima_forecast[0]) / 2
print(f"Ensemble Predicted Next Day Price: ${ensemble_pred:.2f}")

# --------------------------
# Visualize Predictions (Optional)
# --------------------------
plt.figure(figsize=(10, 5))
plt.plot(df_original.index[-100:], df_original["Close"].tail(100), label="Historical Close Price")
plt.scatter(df_original.index[-1] + pd.Timedelta(days=1), lstm_pred[0][0], color='red', label="LSTM Prediction")
plt.scatter(df_original.index[-1] + pd.Timedelta(days=1), arima_forecast[0], color='green', label="ARIMA Prediction")
plt.scatter(df_original.index[-1] + pd.Timedelta(days=1), ensemble_pred, color='blue', label="Ensemble Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title(f"{symbol} Stock Prediction Ensemble")
plt.legend()
plt.show()
