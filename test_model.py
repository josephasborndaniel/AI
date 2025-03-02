import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from data_loader import fetch_stock_data

# Load trained model
model = load_model("models/stock_lstm.h5")

# Load stock data
symbol = "AAPL"
df = fetch_stock_data(symbol)[["Close"]].copy()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
df["Close"] = scaler.fit_transform(df)

# Prepare test data (last 60 days)
window_size = 60
test_data = df[-window_size:].values
test_data = np.reshape(test_data, (1, test_data.shape[0], 1))

# Predict next day's price
predicted_price = model.predict(test_data)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted Next Day Price for {symbol}: ${predicted_price[0][0]:.2f}")
