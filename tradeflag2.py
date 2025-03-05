# use rand forest model to predict.

import numpy as np  # Add this import at the top of your script
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Download Stock Data
def get_stock_data(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    return data

# Step 2: Calculate Technical Indicators
def add_technical_indicators(df):
    # Ensure 'Close' is a 1D series and pass it to the RSIIndicator
    close_prices = df['Close'].squeeze()  # .squeeze() makes sure it's 1D
    rsi_indicator = ta.momentum.RSIIndicator(close_prices, window=14)
    df['RSI'] = rsi_indicator.rsi()

    # Same for MACD
    macd_indicator = ta.trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()

    # Same for SMA
    sma_indicator = ta.trend.SMAIndicator(close_prices, window=20)
    df['SMA'] = sma_indicator.sma_indicator()

    df.dropna(inplace=True)
    return df

# Step 3: Prepare Features and Labels
def prepare_data(df):
    X = df[['RSI', 'MACD', 'MACD_signal', 'SMA']]
    df['Target'] = df['Close'].shift(-1) > df['Close']  # 1 if price goes up next day, else 0
    y = df['Target'].astype(int)
    return X, y

# Step 4: Train the Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model

def simulate_trading(df, model):
    X = df[['RSI', 'MACD', 'MACD_signal', 'SMA']]
    predictions = model.predict(X)

    # If predictions is a DataFrame with more than one column, pick the first one.
    if isinstance(predictions, pd.DataFrame):
        if predictions.shape[1] > 1:
            predictions = predictions.iloc[:, 0]
        else:
            predictions = predictions.squeeze()
    # If predictions is a numpy array, flatten it if needed and convert to a Series.
    elif isinstance(predictions, np.ndarray):
        if predictions.ndim > 1:
            if predictions.shape[1] > 1:
                predictions = predictions[:, 0]
            else:
                predictions = predictions.flatten()
        predictions = pd.Series(predictions, index=df.index)

    # Make sure the Prediction column is a Series.
    df['Prediction'] = pd.Series(predictions, index=df.index).squeeze()

    # Force the multiplication result to be a Series:
    pred_series = df['Prediction'].squeeze()
    pct_change_series = df['Close'].pct_change().squeeze()
    strat_return = pred_series.shift(1) * pct_change_series

    # Now, explicitly convert the result to a Series with the same index:
    df['Strategy_Return'] = pd.Series(strat_return, index=df.index).squeeze()

    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Market_Return'] = (1 + df['Close'].pct_change()).cumprod()

    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Market_Return'], label='Market Return')
    plt.plot(df['Cumulative_Strategy_Return'], label='Strategy Return')
    plt.legend()
    plt.title('Trading Strategy vs Market Performance')
    plt.show()

    # Generate trade flags based on predictions
    df['Signal_Change'] = df['Prediction'].diff()
    df['Trade_Flag'] = df['Signal_Change'].apply(lambda x: 'Buy' if x == 1 else ('Sell' if x == -1 else 'Hold'))

    # Plot trade signals on price chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Price')
    buy_signals = df[df['Trade_Flag'] == 'Buy']
    sell_signals = df[df['Trade_Flag'] == 'Sell']
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal', s=100)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal', s=100)
    plt.legend()
    plt.title('Price Chart with Trade Signals')
    plt.show()

# Example usage:
def main():
    ticker = 'AAPL'
    # ticker = 'QQQ'
    data = get_stock_data(ticker)
    data = add_technical_indicators(data)
    X, y = prepare_data(data)
    model = train_model(X, y)
    
    # Simulate trading and plot both returns and trade signals
    simulate_trading(data, model)

if __name__ == "__main__":
    main()