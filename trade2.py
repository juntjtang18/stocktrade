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
import tensorflow.keras.backend as K

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Fetch stock data
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2020-01-01", end="2025-03-02")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Initial preprocessing
data['Returns'] = data['Close'].pct_change()
data['Lag1'] = data['Returns'].shift(1)
data = data.dropna()

# Fetch additional data
sp500 = yf.download("^GSPC", start="2020-01-01", end="2025-03-02")['Close']
data['SP500'] = sp500.reindex(data.index).ffill()
vix = yf.download("^VIX", start="2020-01-01", end="2025-03-02")['Close']
data['VIX'] = vix.reindex(data.index).ffill()

# Add technical indicators
print("Data length after initial preprocessing:", len(data))
print("NaN in Close:", data['Close'].isna().sum())
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

# Select and scale features
features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']
data = data[features].dropna()
scalers = {feat: RobustScaler() for feat in features}
scaled_data = np.column_stack([scalers[feat].fit_transform(data[[feat]]) for feat in features])

# Create sequences (5% in 15 days)
sequence_length = 30
X, y = [], []
for i in range(sequence_length, len(scaled_data) - 15):
    X.append(scaled_data[i-sequence_length:i])
    future_return = (scaled_data[i+15, 0] - scaled_data[i, 0]) / scaled_data[i, 0]
    y.append(1 if future_return > 0.05 else 0)
X, y = np.array(X), np.array(y)
print("Buy percentage:", np.mean(y) * 100)

# Add noise
noise = np.random.normal(0, 0.01, X.shape)  # Reduced from 0.02 to 0.01
X_noisy = X + noise

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X_noisy[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_prices = data['Close'].values[train_size + sequence_length:]
future_prices = data['Close'].values[train_size + sequence_length + 15:]

# Focal Loss
def focal_loss(gamma=1.5, alpha=0.6):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss, axis=-1)
    return focal_loss_fixed

# Train GRU Model with increased capacity and reduced regularization
model = Sequential()
model.add(Bidirectional(GRU(units=128, return_sequences=True), input_shape=(sequence_length, len(features))))
model.add(Dropout(0.2))  # Reduced from 0.3 to 0.2
model.add(Bidirectional(GRU(units=64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(units=32, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.001)))  # Reduced from 0.005 to 0.001
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer=Nadam(learning_rate=0.0001), loss=focal_loss(gamma=1.5, alpha=0.6), metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), 
                    callbacks=[early_stop, lr_scheduler])

# 1. LSTM Prediction
test_predictions = model.predict(X_test)
lstm_signals = ["buy" if p > 0.5 else "sell" for p in test_predictions]  # Adjusted threshold from 0.55 to 0.5
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
for i in range(len(data) - 15):
    if i < 1:
        macd_signals.append("sell")
        continue
    if data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and data['MACD'].iloc[i-1] <= data['MACD_Signal'].iloc[i-1]:
        macd_signals.append("buy")
    elif data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and data['MACD'].iloc[i-1] >= data['MACD_Signal'].iloc[i-1]:
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
for i in range(len(data) - 15):
    if i < 1:
        rsi_signals.append("sell")
        continue
    if data['RSI'].iloc[i] < 40:
        rsi_signals.append("buy")
    elif data['RSI'].iloc[i] > 60:
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
latest_data = scaled_data[-sequence_length-15:-15].reshape(1, sequence_length, len(features))
prediction = model.predict(latest_data)[0][0]
lstm_flag = "buy" if prediction > 0.5 else "sell"
print(f"LSTM Predicted trade flag (next 15 days): {lstm_flag} (Probability: {prediction:.4f})")
macd_flag = "buy" if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else "sell"
print(f"MACD Predicted trade flag (next 15 days): {macd_flag}")
rsi_flag = "buy" if data['RSI'].iloc[-1] < 40 else "sell" if data['RSI'].iloc[-1] > 60 else "hold"
print(f"RSI Predicted trade flag (next 15 days): {rsi_flag} (RSI: {data['RSI'].iloc[-1]:.2f})")

# Plotting
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('GRU Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()