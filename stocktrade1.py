import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Download Data
def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Calculate Indicators
def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.rolling(window=window).mean()
    ema_down = down.rolling(window=window).mean()
    rsi = 100 - (100 / (1 + (ema_up / ema_down)))
    return rsi

# Prepare Data
def prepare_data(df, lookback=60):
    df = df.dropna()
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Build Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train Model
def train_model(X_train, y_train, epochs=25, batch_size=32):
    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)  # Reduced verbosity
    return model

# Predict
def predict(model, X_test, scaler):
    predictions = model.predict(X_test, verbose=0)  # Reduced verbosity
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 3))), axis=1))[:, 0]
    return predictions

# Generate Trading Signals (Corrected)
def generate_trading_signals(df, predictions, lookback, train_size):
    predictions_shifted = pd.Series(predictions).shift(1).values
    df_close = df['Close'].values

    signals_start_index = lookback + train_size  # Correct start index
    signals_end_index = signals_start_index + len(predictions)  # Correct end index (no -1)

    signals_index = df.index[signals_start_index:signals_end_index]

    buy_signals = (predictions[1:] > df_close[signals_start_index+1:signals_end_index]) & (predictions_shifted[:-1] <= df_close[signals_start_index:signals_end_index-1])
    sell_signals = (predictions[1:] < df_close[signals_start_index+1:signals_end_index]) & (predictions_shifted[:-1] >= df_close[signals_start_index:signals_end_index-1])

    signals = pd.Series(['Hold'] * len(predictions[1:]), index=signals_index[1:])  # ***KEY CORRECTION HERE***

    # Correct boolean indexing
    buy_indices = np.where(buy_signals)[0]
    sell_indices = np.where(sell_signals)[0]

    signals.iloc[buy_indices] = 'Buy'
    signals.iloc[sell_indices] = 'Sell'

    return list(zip(signals.index, signals))

# --- Main execution ---
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-10-27"

df = download_data(ticker, start_date, end_date)
df = calculate_indicators(df)

lookback = 60
X, y, scaler = prepare_data(df, lookback)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = train_model(X_train, y_train)
predictions = predict(model, X_test, scaler)

trading_signals = generate_trading_signals(df, predictions, lookback, train_size)

# Print signals (for verification)
for date, signal in trading_signals:
    print(f"{date}: {signal}")

# Convert datetime indices to integer indices for plotting
buy_indices = [df.index.get_loc(date) for date, signal in trading_signals if signal == 'Buy']
sell_indices = [df.index.get_loc(date) for date, signal in trading_signals if signal == 'Sell']

# Plotting (optional)
plt.figure(figsize=(12, 6))
plt.plot(df.index[lookback+train_size:lookback+train_size+len(predictions)], df['Close'][lookback+train_size:lookback+train_size+len(predictions)], label='Actual Close')  # Corrected slicing
plt.plot(df.index[lookback+train_size:lookback+train_size+len(predictions)], predictions, label='Predicted Close') # Corrected slicing

plt.scatter(df.index[buy_indices], df['Close'].iloc[buy_indices], marker='^', color='green', label='Buy')
plt.scatter(df.index[sell_indices], df['Close'].iloc[sell_indices], marker='v', color='red', label='Sell')

plt.title('Stock Price Prediction with Trading Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()