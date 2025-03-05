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
from datetime import datetime, date

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Local file path
local_file = "AAPL_data.csv"
stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-03-03"  # Today’s date as per your system time

# Function to check if local data is up-to-date
def is_data_up_to_date(file_path, end_date):
    if not os.path.exists(file_path):
        return False
    data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    latest_date = data.index.max().date()
    return latest_date >= date.fromisoformat(end_date)

# Load or fetch data
if is_data_up_to_date(local_file, end_date):
    print(f"Loading up-to-date data from {local_file}")
    data = pd.read_csv(local_file, index_col="Date", parse_dates=True)
else:
    print(f"Fetching fresh {stock_symbol} data from {start_date} to {end_date}...")
    data = yf.download(stock_symbol, start=start_date, end=end_date, progress=True)
    if data.empty:
        raise ValueError(f"Failed to fetch {stock_symbol} data from yfinance. Check network or manually download to {local_file}.")
    data.to_csv(local_file)
    print(f"Data saved to {local_file}")

print("Data length after initial fetch/load:", len(data))
print("NaN in Close:", data['Close'].isna().sum())

# Ensure chronological order
data = data.sort_index(ascending=True)

# Initial preprocessing
data['Returns'] = data['Close'].pct_change()
data['Lag1'] = data['Returns'].shift(1)

# Fetch additional data (only if not already in CSV)
if 'SP500' not in data.columns or 'VIX' not in data.columns:
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)['Close']
    data['SP500'] = sp500.reindex(data.index).ffill()
    vix = yf.download("^VIX", start=start_date, end=end_date)['Close']
    data['VIX'] = vix.reindex(data.index).ffill()
    data.to_csv(local_file)  # Update CSV with new columns
    print(f"Updated {local_file} with SP500 and VIX data")

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
print("Full data length before NaN drop:", len(full_data))

# Create sequences (5% in 15 days) before scaling
sequence_length = 30
X, y = [], []
raw_prices = full_data['Close'].values
for i in range(sequence_length, len(full_data) - 15):
    X.append(full_data[features].iloc[i-sequence_length:i].values)
    future_return = (raw_prices[i + 15] - raw_prices[i]) / raw_prices[i]
    y.append(1 if future_return > 0.05 else 0)  # Buy if >5% increase

# Convert to arrays and drop NaN sequences
X = np.array(X)
y = np.array(y)
valid_indices = ~np.isnan(X).any(axis=(1, 2))
X = X[valid_indices]
y = y[valid_indices]
print("Data length after sequence creation and NaN removal:", len(X))

# Scale features
X_scaled = np.zeros_like(X)
for i in range(X.shape[2]):  # Scale each feature
    scaler = RobustScaler()
    X_scaled[:, :, i] = scaler.fit_transform(X[:, :, i])

# Verify buy percentage
print("Sequence buy percentage:", np.mean(y) * 100)

# Debug: Print extended sample sequences and labels
print("Sample X shape:", X_scaled[:10].shape)
print("Sample y:", y[:10])
print("Sample raw prices (current vs. 15 days later):", [(raw_prices[i], raw_prices[i+15], (raw_prices[i+15] - raw_prices[i]) / raw_prices[i] * 100) for i in range(30, 40)])

# Split data
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_prices = raw_prices[train_size + sequence_length:]
future_prices = raw_prices[train_size + sequence_length + 15:]

# Train GRU Model
model = Sequential()
model.add(Bidirectional(GRU(units=64, return_sequences=False), input_shape=(sequence_length, len(features))))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer=Nadam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

# Class weights
class_weights = {0: 1.0, 1: (1 - np.mean(y)) / np.mean(y)}

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), 
                    callbacks=[early_stop, lr_scheduler], class_weight=class_weights)

# 1. LSTM Prediction
test_predictions = model.predict(X_test)
lstm_signals = ["buy" if p > 0.5 else "sell" for p in test_predictions]
lstm_profits = []
for i, (signal, price, future) in enumerate(zip(lstm_signals, test_prices, future_prices)):
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
for i in range(len(full_data) - 15):
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
for i, (signal, price, future) in enumerate(zip(macd_test_signals, test_prices, future_prices)):
    actual_return = (future - price) / price
    if signal == "buy":
        capped_return = max(actual_return, -0.02)
        macd_profits.append(price * capped_return)
    else:
        macd_profits.append(price * -actual_return if actual_return < 0 else 0)
macd_accuracy = sum(1 for p, a in zip(macd_test_signals, y_test) if (p == "buy" and a == 1) or (p == "sell" and a == 0)) / len(y_test)
print(f"MACD Accuracy: {macd_accuracy*100:.2f}%")
print(f"MACD Backtest Profit: {float(sum(macd_profits)):.2f}")

# 3. RSI Strategy
rsi_signals = []
for i in range(len(full_data) - 15):
    if i < 1:
        rsi_signals.append("sell")
        continue
    if full_data['RSI'].iloc[i] < 40:
        rsi_signals.append("buy")
    elif full_data['RSI'].iloc[i] > 60:
        rsi_signals.append("sell")
    else:
        rsi_signals.append(rsi_signals[-1])
rsi_test_signals = rsi_signals[train_size + sequence_length:]
rsi_profits = []
for i, (signal, price, future) in enumerate(zip(rsi_test_signals, test_prices, future_prices)):
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
latest_data = X_scaled[-sequence_length:].reshape(1, sequence_length, len(features))
prediction = model.predict(latest_data)[0][0]
lstm_flag = "buy" if prediction > 0.5 else "sell"
print(f"LSTM Predicted trade flag (next 15 days): {lstm_flag} (Probability: {prediction:.4f})")
macd_flag = "buy" if full_data['MACD'].iloc[-1] > full_data['MACD_Signal'].iloc[-1] else "sell"
print(f"MACD Predicted trade flag (next 15 days): {macd_flag}")
rsi_flag = "buy" if full_data['RSI'].iloc[-1] < 40 else "sell" if full_data['RSI'].iloc[-1] > 60 else "hold"
print(f"RSI Predicted trade flag (next 15 days): {rsi_flag} (RSI: {full_data['RSI'].iloc[-1]:.2f})")

# Plotting
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('GRU Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()