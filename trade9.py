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
from mylib.download_data import download_data  # Import the download_data function from your module
from datetime import datetime

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configurable parameters
stock_symbol = "AAPL"
data_dir = os.path.join(os.path.dirname(__file__), "data")
local_file = os.path.join(data_dir, f"{stock_symbol}_data.csv")
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
sequence_length = 15
threshold = 0.01  # Reduced to 1% to capture more price movements, adjust as needed
initial_amount = 10000  # $10,000 starting capital

# Load data using the download_data module (Polygon.io only, no modifications to source file)
data = download_data(stock_symbol, source="polygon")  # Use Polygon.io as specified

# Ensure chronological order only for analysis (don't save back to file), ascending for processing
data = data.sort_index(ascending=True).drop_duplicates()
data = data[data['Volume'] > 0]
data = data.loc[start_date:end_date]

# Store the sorted dates for later use in plotting
dates = data.index

# Initial preprocessing
data['Returns'] = data['Close'].pct_change()
data['Lag1'] = data['Returns'].shift(1)

# Add SP500 and VIX if not present (using yfinance, no save to source file)
if 'SP500' not in data.columns or 'VIX' not in data.columns:
    try:
        sp500 = pd.read_csv(os.path.join(data_dir, "SP500_data.csv"), index_col=0, parse_dates=True) if os.path.exists(os.path.join(data_dir, "SP500_data.csv")) else yf.download("^GSPC", start=start_date, end=end_date)['Close']
        data['SP500'] = sp500.reindex(data.index).ffill()
        vix = pd.read_csv(os.path.join(data_dir, "VIX_data.csv"), index_col=0, parse_dates=True) if os.path.exists(os.path.join(data_dir, "VIX_data.csv")) else yf.download("^VIX", start=start_date, end=end_date)['Close']
        data['VIX'] = vix.reindex(data.index).ffill()
        # Do not save back to the source data file in ./data
    except Exception as e:
        print(f"Error downloading SP500/VIX: {e}")

# Add technical indicators
if len(data) < 26 or data['Close'].isna().sum() > 0:
    raise ValueError("Not enough data or NaNs in Close column.")
data['RSI'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
if macd is None:
    raise ValueError("MACD calculation failed.")
data['MACD'] = macd['MACD_12_26_9']
data['MACD_Signal'] = macd['MACDs_12_26_9']
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
data['ROC'] = ta.roc(data['Close'], length=10)

# Select features
features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']
full_data = data[features].copy()

# Add flag columns initialized as NaN for both LSTM and MACD
full_data['lstm_flag'] = np.nan
full_data['macd_flag'] = np.nan

# Create sequences for LSTM with corrected labels
X = []
y = []
sequence_starts = []  # Track indices of sequence starts
for i in range(len(full_data) - sequence_length * 2):
    future_idx = i + sequence_length
    if future_idx >= len(full_data):
        break
    future_return = (full_data['Close'].iloc[future_idx] - full_data['Close'].iloc[i]) / full_data['Close'].iloc[i]
    if future_return > threshold:
        label = 1  # Buy: rise > threshold
    elif future_return < -threshold:
        label = -1  # Sell: drop < -threshold
    else:
        label = 0  # Hold: -threshold to +threshold
    y.append(label)
    X.append(full_data[features].iloc[i:i+sequence_length].values)
    sequence_starts.append(i)

# Convert to arrays and drop NaN sequences
X = np.array(X)
y = np.array(y)
sequence_starts = np.array(sequence_starts)
valid_indices = ~np.isnan(X).any(axis=(1, 2))
X = X[valid_indices]
y = y[valid_indices]
sequence_starts = sequence_starts[valid_indices]
print(f"Sequence buy percentage: {np.mean(y == 1) * 100:.2f}%")
print(f"Sequence sell percentage: {np.mean(y == -1) * 100:.2f}%")
print(f"Sequence hold percentage: {np.mean(y == 0) * 100:.2f}%")

# Debug: Verify labels with raw returns
print("Sample labels vs. returns:")
for i in range(15, 25):
    if i < len(full_data) - sequence_length:
        ret = (full_data['Close'].iloc[i + sequence_length] - full_data['Close'].iloc[i]) / full_data['Close'].iloc[i] * 100
        print(f"Date: {dates[i]}, Return: {ret:.2f}%, Label: {y[i]}")

# Scale features
X_scaled = np.zeros_like(X)
for i in range(X.shape[2]):
    scaler = RobustScaler()
    X_scaled[:, :, i] = scaler.fit_transform(X[:, :, i])

# Split data, but save the last sequence for prediction
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:-1]  # Exclude last sequence for test
y_train, y_test = y[:train_size], y[train_size:-1]
test_sequence_starts = sequence_starts[train_size:-1]

# Keep the last sequence for prediction
X_latest = X_scaled[-1:, :]  # Last 15 days for prediction

# Convert labels to categorical for multi-class
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train + 1, num_classes=3)  # Shift: -1 -> 0, 0 -> 1, 1 -> 2
y_test_cat = to_categorical(y_test + 1, num_classes=3)

# Train GRU Model (multi-class) with balanced weights
model = Sequential()
model.add(Bidirectional(GRU(units=64, return_sequences=False), input_shape=(sequence_length, len(features))))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(units=3, activation='softmax'))  # 3 classes: sell, hold, buy
model.compile(optimizer=Nadam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])

class_weights_dict = {0: 1.0, 1: 2.0, 2: 1.0}  # Boost "hold" to capture more price stability
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
history = model.fit(X_train, y_train_cat, epochs=50, batch_size=4, validation_data=(X_test, y_test_cat), 
                    callbacks=[early_stop, lr_scheduler], class_weight=class_weights_dict)

# 1. LSTM Trading Strategy with $10,000
test_predictions = model.predict(X_test)
lstm_signals = ["sell" if np.argmax(p) == 0 else "hold" if np.argmax(p) == 1 else "buy" for p in test_predictions]

# Assign LSTM signals to the flag column at sequence start indices
for idx, signal in zip(test_sequence_starts, lstm_signals):
    full_data.iloc[idx, full_data.columns.get_loc('lstm_flag')] = signal

# Define test period DataFrame for LSTM (non-NaN lstm_flags), keeping dates as index
test_df = full_data.iloc[test_sequence_starts[0]:test_sequence_starts[-1] + sequence_length + 1].dropna(subset=['lstm_flag'])

amount = initial_amount
shares = 0
print("\nLSTM Trades (Test Period):")
print("Test period LSTM signals:", test_df['lstm_flag'].tolist())
for i in range(len(test_df)):
    signal = test_df['lstm_flag'].iloc[i]
    price = test_df['Close'].iloc[i]
    # Use next day's price for selling if available, otherwise last price
    future = test_df['Close'].iloc[i + 1] if i + 1 < len(test_df) else test_df['Close'].iloc[-1]
    date = test_df.index[i]
    if signal == "buy" and amount > 0:
        shares = amount / price
        amount = 0
        print(f"{date}: Bought {shares:.2f} shares at ${price:.2f}")
    elif signal == "sell" and shares > 0:
        amount = shares * future
        print(f"{date}: Sold {shares:.2f} shares at ${future:.2f}, Amount: ${amount:.2f}")
        shares = 0

lstm_accuracy = sum(1 for p, a in zip(test_df['lstm_flag'], y_test) if (p == "buy" and a == 1) or (p == "sell" and a == -1) or (p == "hold" and a == 0)) / len(y_test)
total_profit = amount - initial_amount if amount > 0 else (shares * test_df['Close'].iloc[-1] - initial_amount if shares > 0 else 0)
profit_percentage = (total_profit / initial_amount) * 100
print(f"LSTM Accuracy: {lstm_accuracy*100:.2f}%")
print(f"LSTM Total Profit on ${initial_amount} (Mar 2024 - Mar 2025): ${total_profit:.2f}")
print(f"LSTM Profit Percentage: {profit_percentage:.2f}%")

# Predict for the last 15 days
latest_prediction = model.predict(X_latest)[0]
latest_lstm_flag = "sell" if np.argmax(latest_prediction) == 0 else "hold" if np.argmax(latest_prediction) == 1 else "buy"
print(f"\nLSTM Predicted trade flag (next 15 days): {latest_lstm_flag} (Probabilities: Sell: {latest_prediction[0]:.4f}, Hold: {latest_prediction[1]:.4f}, Buy: {latest_prediction[2]:.4f})")

# 2. MACD Strategy with $10,000 and Corrected Flags
macd_signals = []
macd_flags = []  # To store flags only at crossovers
for i in range(len(full_data) - sequence_length):
    if i < 1:
        macd_signals.append("sell")
        macd_flags.append(np.nan)  # No flag at the start
        continue
    current_macd = full_data['MACD'].iloc[i]
    current_signal = full_data['MACD_Signal'].iloc[i]
    prev_macd = full_data['MACD'].iloc[i-1]
    prev_signal = full_data['MACD_Signal'].iloc[i-1]
    
    if (prev_macd <= prev_signal and current_macd > current_signal):  # MACD crosses above Signal (Buy)
        macd_signals.append("buy")
        macd_flags.append("buy")
    elif (prev_macd >= prev_signal and current_macd < current_signal):  # MACD crosses below Signal (Sell)
        macd_signals.append("sell")
        macd_flags.append("sell")
    else:
        macd_signals.append(macd_signals[-1])  # Maintain previous signal
        macd_flags.append(np.nan)  # No flag unless crossover

# Assign MACD signals and flags to the macd_flag column
for idx, (signal, flag) in enumerate(zip(macd_signals, macd_flags), 0):
    full_data.iloc[idx, full_data.columns.get_loc('macd_flag')] = flag  # Use flag for plotting, signal for trading

# Define test period DataFrame for MACD (non-NaN macd_flags), keeping dates as index
macd_test_df = full_data.iloc[test_sequence_starts[0]:test_sequence_starts[-1] + sequence_length + 1].dropna(subset=['macd_flag'])

amount = initial_amount
shares = 0
print("\nMACD Trades:")
print("Test period MACD signals:", macd_test_df['macd_flag'].tolist())
for i in range(len(macd_test_df)):
    signal = macd_test_df['macd_flag'].iloc[i]  # Use macd_flag for trading (only crossovers)
    if pd.isna(signal):  # Skip if no flag (no crossover)
        continue
    price = macd_test_df['Close'].iloc[i]
    # Use next day's price for selling if available, otherwise last price
    future = macd_test_df['Close'].iloc[i + 1] if i + 1 < len(macd_test_df) else macd_test_df['Close'].iloc[-1]
    date = macd_test_df.index[i]
    if signal == "buy" and amount > 0:
        shares = amount / price
        amount = 0
        print(f"{date}: Bought {shares:.2f} shares at ${price:.2f}")
    elif signal == "sell" and shares > 0:
        amount = shares * future
        print(f"{date}: Sold {shares:.2f} shares at ${future:.2f}, Amount: ${amount:.2f}")
        shares = 0

# Corrected MACD accuracy calculation
# Get non-NaN indices from macd_test_df['macd_flag']
valid_macd_flags = macd_test_df['macd_flag'].dropna().values
valid_positions = np.arange(len(macd_test_df))[macd_test_df['macd_flag'].notna()]  # Get integer positions of non-NaN flags

# Align y_test with valid MACD flags using integer positions relative to test_sequence_starts
# Get the positions relative to the full DataFrame or test period
full_data_positions = np.arange(len(full_data))[test_sequence_starts[0]:test_sequence_starts[-1] + sequence_length + 1]
y_test_positions = full_data_positions[valid_positions]  # Align with y_test indices

# Ensure y_test_positions are within bounds of y_test
y_test_positions = y_test_positions[y_test_positions < len(y_test)]
y_test_aligned = y_test[y_test_positions]  # Use integer positions directly

# Calculate accuracy only for non-NaN pairs
macd_accuracy = sum(1 for p, a in zip(valid_macd_flags, y_test_aligned) if (p == "buy" and a == 1) or (p == "sell" and a == -1)) / len(valid_macd_flags) if len(valid_macd_flags) > 0 else 0
macd_total_profit = amount - initial_amount if amount > 0 else (shares * macd_test_df['Close'].iloc[-1] - initial_amount if shares > 0 else 0)
macd_profit_percentage = (macd_total_profit / initial_amount) * 100
print(f"MACD Accuracy: {macd_accuracy*100:.2f}%")
print(f"MACD Total Profit on ${initial_amount} (Mar 2024 - Mar 2025): ${macd_total_profit:.2f}")
print(f"MACD Profit Percentage: {macd_profit_percentage:.2f}%")

# 3. RSI Strategy with Debug (unchanged for now, can be added similarly)
rsi_signals = []
for i in range(len(full_data) - sequence_length):
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
amount = initial_amount
shares = 0
print("\nRSI Trades:")
print("All RSI test period signals:", rsi_test_signals)
for i, (signal, price, future, date) in enumerate(zip(rsi_test_signals, test_df['Close'], test_df['Close'].shift(-1).fillna(test_df['Close'].iloc[-1]), test_df.index)):
    if signal == "buy" and amount > 0:
        shares = amount / price
        amount = 0
        print(f"{date}: Bought {shares:.2f} shares at ${price:.2f}")
    elif signal == "sell" and shares > 0:
        amount = shares * future
        print(f"{date}: Sold {shares:.2f} shares at ${future:.2f}, Amount: ${amount:.2f}")
        shares = 0
rsi_accuracy = sum(1 for p, a in zip(rsi_test_signals, y_test) if (p == "buy" and a == 1) or (p == "sell" and a in [0, -1])) / len(y_test)
rsi_total_profit = amount - initial_amount if amount > 0 else (shares * test_df['Close'].iloc[-1] - initial_amount if shares > 0 else 0)
rsi_profit_percentage = (rsi_total_profit / initial_amount) * 100
print(f"RSI Accuracy: {rsi_accuracy*100:.2f}%")
print(f"RSI Total Profit on ${initial_amount} (Mar 2024 - Mar 2025): ${rsi_total_profit:.2f}")
print(f"RSI Profit Percentage: {rsi_profit_percentage:.2f}%")

# Plotting: Two subplots sharing x-axis, using dates as index
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top subplot: Price with LSTM signals
ax1.plot(test_df.index, test_df['Close'], label="Price", color="blue")

# Calculate offset for markers (e.g., 2% of price range)
price_range = test_df['Close'].max() - test_df['Close'].min()
offset = price_range * 0.02  # Adjust this value to position markers closer/farther

# Plot LSTM buy signals (^) above price
buy_df_lstm = test_df[test_df['lstm_flag'] == 'buy']
ax1.scatter(buy_df_lstm.index, buy_df_lstm['Close'] + offset, marker='^', color='green', label='LSTM Buy', s=100)

# Plot LSTM sell signals (v) below price
sell_df_lstm = test_df[test_df['lstm_flag'] == 'sell']
ax1.scatter(sell_df_lstm.index, sell_df_lstm['Close'] - offset, marker='v', color='red', label='LSTM Sell', s=100)

ax1.set_title("LSTM Signals vs. Price")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)

# Bottom subplot: MACD with MACD signals (only at crossovers)
ax2.plot(macd_test_df.index, macd_test_df['MACD'], label="MACD", color="blue", linestyle='--')
ax2.plot(macd_test_df.index, macd_test_df['MACD_Signal'], label="MACD Signal", color="orange")
macd_range = max(macd_test_df['MACD'].max(), macd_test_df['MACD_Signal'].max()) - min(macd_test_df['MACD'].min(), macd_test_df['MACD_Signal'].min())
macd_offset = macd_range * 0.02  # Adjust for MACD scale

# Plot MACD buy signals (^) above MACD line (only at crossovers)
buy_df_macd = macd_test_df[macd_test_df['macd_flag'] == 'buy']
ax2.scatter(buy_df_macd.index, buy_df_macd['MACD'] + macd_offset, marker='^', color='green', label='MACD Buy', s=100)

# Plot MACD sell signals (v) below MACD line (only at crossovers)
sell_df_macd = macd_test_df[macd_test_df['macd_flag'] == 'sell']
ax2.scatter(sell_df_macd.index, sell_df_macd['MACD'] - macd_offset, marker='v', color='red', label='MACD Sell', s=100)

ax2.set_title("MACD and Signals (Crossovers Only)")
ax2.set_ylabel("MACD Value")
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Latest Predictions
latest_data = X_scaled[-1:, :]  # Last 15 days for prediction
prediction = model.predict(latest_data)[0]
lstm_flag = "sell" if np.argmax(prediction) == 0 else "hold" if np.argmax(prediction) == 1 else "buy"
print(f"\nLSTM Predicted trade flag (next 15 days): {lstm_flag} (Probabilities: Sell: {prediction[0]:.4f}, Hold: {prediction[1]:.4f}, Buy: {prediction[2]:.4f})")
macd_flag = "buy" if full_data['MACD'].iloc[-1] > full_data['MACD_Signal'].iloc[-1] else "sell"
print(f"MACD Predicted trade flag (next 15 days): {macd_flag}")
rsi_flag = "buy" if full_data['RSI'].iloc[-1] < 40 else "sell" if full_data['RSI'].iloc[-1] > 60 else "hold"
print(f"RSI Predicted trade flag (next 15 days): {rsi_flag} (RSI: {full_data['RSI'].iloc[-1]:.2f})")