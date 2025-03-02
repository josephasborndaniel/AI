import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from data_loader import fetch_stock_data

# Load stock data
symbol = "AAPL"
df = fetch_stock_data(symbol)[["Close"]].copy()

# Normalize the data (scale values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0,1))
df["Close"] = scaler.fit_transform(df)

# Prepare training data
window_size = 60
X, Y = [], []

for i in range(window_size, len(df)):
    X.append(df["Close"].iloc[i-window_size:i].values)
    Y.append(df["Close"].iloc[i])

X, Y = np.array(X), np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

# Build LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Train Model
model.fit(X, Y, epochs=50, batch_size=32)

# Save model
model.save("models/stock_lstm.h5")

print("Model trained and saved successfully!")
