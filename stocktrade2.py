import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Download Data
def download_data(ticker, start_date, end_date):
    file_path = f'./data/{ticker}_data.csv'  # Construct file path based on ticker
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)  # Read CSV with datetime index
    # data.sort_index(inplace=True)  # Ensure chronological order
    return data

# Calculate Indicators
def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.rolling(window=window).mean()
    ema_down = down.rolling(window=window).mean()
    rsi = 100 - (100 / (1 + (ema_up / ema_down)))
    return rsi

def calculate_macd(prices):
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Prepare Data
def prepare_data(df, lookback=60):
    df = df.dropna()
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'EMA_20']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        price_change = scaled_data[i, 0] - scaled_data[i - 1, 0]
        if price_change > 0:
            y.append(1)  # Buy
        elif price_change < 0:
            y.append(-1)  # Sell
        else:
            y.append(0)  # Hold

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Flatten 3D data to 2D
def flatten_data(X):
    num_samples, time_steps, num_features = X.shape
    return X.reshape(num_samples, time_steps * num_features)

# Build and Train Random Forest Model
def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Model Mean Squared Error: {mse}")
    print(f"Model Root Mean Squared Error: {rmse}")
    print(f"Model Mean Absolute Error: {mae}")
    print(f"Model Accuracy: {accuracy}")
    print(f"Model Precision: {precision}")
    print(f"Model Recall: {recall}")
    print(f"Model F1 Score: {f1}")

# Generate Trading Signals
def generate_trading_signals(df, predictions, lookback, train_size):
    df_close = df['Close'].values
    signals_start_index = lookback + train_size  # Correct start index
    signals_end_index = signals_start_index + len(predictions)  # Correct end index

    signals_index = df.index[signals_start_index:signals_end_index]

    buy_signals = (predictions == 1)
    sell_signals = (predictions == -1)

    signals = pd.Series(['Hold'] * len(predictions), index=signals_index)

    # Correct boolean indexing
    buy_indices = np.where(buy_signals)[0]
    sell_indices = np.where(sell_signals)[0]

    signals.iloc[buy_indices] = 'Buy'
    signals.iloc[sell_indices] = 'Sell'

    return list(zip(signals.index, signals))

# Calculate Returns
def calculate_returns(trading_signals, df):
    initial_balance = 10000
    balance = initial_balance
    position = 0.0
    for date, signal in trading_signals:
        if signal == 'Buy' and position == 0.0:
            position = float(balance / df.loc[date, 'Close'].iloc[0])
            balance = 0.0
        elif signal == 'Sell' and position > 0.0:
            balance = float(position * df.loc[date, 'Close'].iloc[0])
            position = 0.0
    if position > 0.0:
        balance += float(position * df.iloc[-1]['Close'])
    return (balance - initial_balance) / initial_balance

# Verify Strategy Returns and Predictions
def verify_results(df, trading_signals):
    df['Strategy'] = 0.0
    df['Market'] = 0.0
    df['Signal'] = 'Hold'

    position = 0.0
    balance = 10000

    for date, signal in trading_signals:
        if signal == 'Buy' and position == 0.0:
            position = float(balance / df.loc[date, 'Close'])
            balance = 0.0
            df.loc[date, 'Signal'] = 'Buy'
        elif signal == 'Sell' and position > 0.0:
            balance = float(position * df.loc[date, 'Close'])
            position = 0.0
            df.loc[date, 'Signal'] = 'Sell'

        df.loc[date, 'Strategy'] = (balance + position * df.loc[date, 'Close']) / 10000 - 1

    df['Market'] = df['Close'] / df['Close'].iloc[0] - 1

    plt.figure(figsize=(12, 6))
    plt.plot(df['Strategy'], label='Strategy Return')
    plt.plot(df['Market'], label='Market Return')
    plt.legend()
    plt.title('Cumulative Returns')
    plt.show()

    print("Verification of Trading Signals:")
    print(df[['Close', 'Signal', 'Strategy', 'Market']].tail(10))

# --- Main execution ---
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-02-20"

df = download_data(ticker, start_date, end_date)
df = calculate_indicators(df)

lookback = 60
X, y, scaler = prepare_data(df, lookback)

X_flattened = flatten_data(X)
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, shuffle=False)

rf_model = train_rf_model(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluate Model
evaluate_model(y_test, y_pred)

trading_signals = generate_trading_signals(df, y_pred, lookback, len(X_train))

# Print signals (for verification)
for date, signal in trading_signals:
    print(f"{date}: {signal}")

# Calculate Returns
returns = calculate_returns(trading_signals, df)
print(f"Trading Strategy Return: {returns * 100:.2f}%")

# Verify Results
verify_results(df, trading_signals)