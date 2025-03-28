import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2
import pandas_ta as ta
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import requests
from datetime import datetime, date

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configurable stock symbol
stock_symbol = "AAPL"  # Change this to test other stocks (e.g., "MSFT", "GOOGL")
local_file = f"{stock_symbol}_data.csv"
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
api_key = "U05ZXJI8JWTSYBNP"  # Your Alpha Vantage API key
sequence_length = 15

# Function to check if local data is up-to-date
def is_data_up_to_date(file_path, end_date):
    if not os.path.exists(file_path):
        return False
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    latest_date = data.index.max().date()
    return latest_date >= date.fromisoformat(end_date)

# Load or fetch data
if is_data_up_to_date(local_file, end_date):
    data = pd.read_csv(local_file, index_col=0, parse_dates=True)
    data = data.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
else:
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url).json()
    if "Time Series (Daily)" not in response:
        raise ValueError(f"Alpha Vantage fetch failed: {response.get('Information', 'Unknown error')}")
    data = pd.DataFrame(response["Time Series (Daily)"]).T
    data = data.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
    data.index = pd.to_datetime(data.index)
    data = data.astype(float)
    data.to_csv(local_file)

# Ensure chronological order, remove duplicates, and filter trading days
data = data.sort_index(ascending=True).drop_duplicates()
data = data[data['Volume'] > 0]
data = data.loc[start_date:end_date]

# Initial preprocessing
data['Returns'] = data['Close'].pct_change()
data['Lag1'] = data['Returns'].shift(1)

# Add SP500 and VIX if not present
if 'SP500' not in data.columns or 'VIX' not in data.columns:
    try:
        sp500 = yf.download("^GSPC", start=start_date, end=end_date)['Close']
        data['SP500'] = sp500.reindex(data.index).ffill()
        vix = yf.download("^VIX", start=start_date, end=end_date)['Close']
        data['VIX'] = vix.reindex(data.index).ffill()
        data.to_csv(local_file)
    except Exception as e:
        pass  # Proceed without SP500/VIX if fetch fails

# Add technical indicators
if len(data) < 26 or data['Close'].isna().sum() > 0:
    raise ValueError("Not enough data or NaNs in Close column for MACD calculation.")
data['RSI'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
if macd is None:
    raise ValueError("MACD calculation failed. Check data integrity.")
data['MACD'] = macd['MACD_12_26_9']
data['MACD_Signal'] = macd['MACDs_12_26_9']
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
data['ROC'] = ta.roc(data['Close'], length=10)

# Select features without NaN drop yet
features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']
full_data = data[features].copy()

# Create sequences with explicit trading day offset
X = []
y = []
raw_prices = full_data['Close'].values
dates = full_data.index

for i in range(len(full_data) - sequence_length * 2):
    future_idx = i + sequence_length
    if future_idx >= len(dates):
        break
    future_return = (raw_prices[future_idx] - raw_prices[i]) / raw_prices[i]
    label = 1 if future_return > 0.01 else 0  # "buy" for gains > 1%
    y.append(label)
    X.append(full_data[features].iloc[i:i+sequence_length].values)

# Convert to arrays and drop NaN sequences
X = np.array(X)
y = np.array(y)
valid_indices = ~np.isnan(X).any(axis=(1, 2))
X = X[valid_indices]
y = y[valid_indices]
print(f"Sequence buy percentage: {np.mean(y) * 100}")

# Scale features
X_scaled = np.zeros_like(X)
for i in range(X.shape[2]):
    scaler = RobustScaler()
    X_scaled[:, :, i] = scaler.fit_transform(X[:, :, i])

# Split data
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_prices = raw_prices[train_size + sequence_length:]
future_prices = raw_prices[train_size + 2 * sequence_length:]

# Train GRU Model
model = Sequential()
model.add(Bidirectional(GRU(units=64, return_sequences=False), input_shape=(sequence_length, len(features))))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer=Nadam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

class_weights = {0: 1.0, 1: 1.6}
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), 
                    callbacks=[early_stop, lr_scheduler], class_weight=class_weights)

# 1. LSTM Prediction
test_predictions = model.predict(X_test)
lstm_signals = ["buy" if p > 0.5 else "sell" for p in test_predictions]
lstm_profits = []
for signal, price, future in zip(lstm_signals, test_prices, future_prices):
    actual_return = (future - price) / price
    if signal == "buy":
        capped_return = max(actual_return, -0.02)
        lstm_profits.append(price * capped_return)
    else:
        lstm_profits.append(price * -actual_return if actual_return < 0 else 0)
lstm_accuracy = sum(1 for p, a in zip(lstm_signals, y_test) if (p == "buy" and a == 1) or (p == "sell" and a == 0)) / len(y_test)
print(f"LSTM Accuracy: {lstm_accuracy*100:.2f}%")
print(f"LSTM Backtest Profit: {float(sum(lstm_profits)):.2f}")

# 2. MACD Strategy
macd_signals = []
for i in range(len(full_data) - sequence_length):
    if i < 1:
        macd_signals.append("sell")
        continue
    if full_data['MACD'].iloc[i] > full_data['MACD_Signal'].iloc[i] and full_data['MACD'].iloc[i-1] <= full_data['MACD_Signal'].iloc[i-1]:
        macd_signals.append("buy")
    elif full_data['MACD'].iloc[i] < full_data['MACD_Signal'].iloc[i] and full_data['MACD'].iloc[i-1] >= full_data['MACD_Signal'].iloc[i-1]:
        macd_signals.append("sell")
    else:
        macd_signals.append(macd_signals[-1])
macd_test_signals = macd_signals[train_size + sequence_length:]
macd_profits = []
for signal, price, future in zip(macd_test_signals, test_prices, future_prices):
    actual_return = (future - price) / price
    if signal == "buy":
        capped_return = max(actual_return, -0.02)
        macd_profits.append(price * capped_return)
    else:
        macd_profits.append(price * -actual_return if actual_return < 0 else 0)
macd_accuracy = sum(1 for p, a in zip(macd_test_signals, y_test) if (p == "buy" and a == 1) or (p == "sell" and a == 0)) / len(y_test)
print(f"MACD Accuracy: {macd_accuracy*100:.2f}%")
print(f"MACD Backtest Profit: {float(sum(macd_profits)):.2f}")

# 3. RSI Strategy (tuned thresholds)
rsi_signals = []
for i in range(len(full_data) - sequence_length):
    if i < 1:
        rsi_signals.append("sell")
        continue
    if full_data['RSI'].iloc[i] < 45:  # Adjusted from 40
        rsi_signals.append("buy")
    elif full_data['RSI'].iloc[i] > 55:  # Adjusted from 60
        rsi_signals.append("sell")
    else:
        rsi_signals.append(rsi_signals[-1])
rsi_test_signals = rsi_signals[train_size + sequence_length:]
rsi_profits = []
for signal, price, future in zip(rsi_test_signals, test_prices, future_prices):
    actual_return = (future - price) / price
    if signal == "buy":
        capped_return = max(actual_return, -0.02)
        rsi_profits.append(price * capped_return)
    else:
        rsi_profits.append(price * -actual_return if actual_return < 0 else 0)
rsi_accuracy = sum(1 for p, a in zip(rsi_test_signals, y_test) if (p == "buy" and a == 1) or (p == "sell" and a == 0)) / len(y_test)
print(f"RSI Accuracy: {rsi_accuracy*100:.2f}%")
print(f"RSI Backtest Profit: {float(sum(rsi_profits)):.2f}")

# Latest Predictions
latest_data = X_scaled[-1, :, :].reshape(1, sequence_length, len(features))
prediction = model.predict(latest_data)[0][0]
lstm_flag = "buy" if prediction > 0.5 else "sell"
print(f"LSTM Predicted trade flag (next 15 days): {lstm_flag} (Probability: {prediction:.4f})")
macd_flag = "buy" if full_data['MACD'].iloc[-1] > full_data['MACD_Signal'].iloc[-1] else "sell"
print(f"MACD Predicted trade flag (next 15 days): {macd_flag}")
rsi_flag = "buy" if full_data['RSI'].iloc[-1] < 45 else "sell" if full_data['RSI'].iloc[-1] > 55 else "hold"
print(f"RSI Predicted trade flag (next 15 days): {rsi_flag} (RSI: {full_data['RSI'].iloc[-1]:.2f})")

# Plotting
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'GRU Model Accuracy for {stock_symbol}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()