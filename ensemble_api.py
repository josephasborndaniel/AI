from fastapi import FastAPI
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import uvicorn

app = FastAPI()

# Load the trained LSTM model once when the app starts
lstm_model = load_model("models/stock_lstm.h5")
scaler = MinMaxScaler(feature_range=(0, 1))

def fetch_stock_data(symbol: str, start="2015-01-01", end="2024-01-01"):
    """Fetch historical stock data from Yahoo Finance."""
    return yf.download(symbol, start=start, end=end)

@app.get("/predict/{symbol}")
def predict_stock(symbol: str):
    # Fetch stock data for the given symbol
    data = fetch_stock_data(symbol)
    if data.empty:
        return {"error": "No data found for symbol"}
    
    df = data[["Close"]].copy()
    df_original = df.copy()  # For ARIMA, we use the original data
    
    # Check for NaN values and clean the data
    if df.isnull().values.any():
        df = df.dropna()

    # Normalize the data for LSTM
    df["Close"] = scaler.fit_transform(df)
    
    # Prepare test data for LSTM using the last 60 days
    window_size = 60
    if len(df) < window_size:
        return {"error": "Not enough data to make a prediction"}
    
    test_data = df[-window_size:].values
    test_data = np.reshape(test_data, (1, test_data.shape[0], 1))
    
    try:
        # LSTM prediction (normalize then inverse-transform)
        lstm_pred_norm = lstm_model.predict(test_data)
        lstm_pred = scaler.inverse_transform(lstm_pred_norm)[0][0]
    except Exception as e:
        return {"error": f"LSTM prediction error: {str(e)}"}
    
    # ARIMA prediction on original (non-normalized) data
    arima_data = df_original["Close"].values
    try:
        # ARIMA order (p,d,q) is set to (5,1,0) for this example.
        arima_model = ARIMA(arima_data, order=(5, 1, 0))
        arima_result = arima_model.fit()
        arima_forecast = arima_result.forecast(steps=1)[0]
    except Exception as e:
        return {"error": f"ARIMA error: {str(e)}"}
    
    # Ensemble: simple average of LSTM and ARIMA predictions
    ensemble_pred = (lstm_pred + arima_forecast) / 2
    
    # Convert numpy.float32 to float for JSON serialization
    return {
        "symbol": symbol,
        "lstm_prediction": float(lstm_pred),  # Convert numpy.float32 to native float
        "arima_prediction": float(arima_forecast),  # Convert numpy.float32 to native float
        "ensemble_prediction": float(ensemble_pred)  # Convert numpy.float32 to native float
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
