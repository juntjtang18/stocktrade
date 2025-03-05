import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# Step 1: Download Stock Data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Calculate Technical Indicators
def add_technical_indicators(df):
    close_prices = df['Close'].squeeze()  # Ensure it's a 1-dimensional series
    volume = df['Volume'].squeeze()  # Ensure it's a 1-dimensional series

    df['RSI'] = ta.momentum.RSIIndicator(close_prices, window=14).rsi()
    df['MACD'] = ta.trend.MACD(close_prices).macd()
    df['MACD_signal'] = ta.trend.MACD(close_prices).macd_signal()
    df['SMA_20'] = ta.trend.SMAIndicator(close_prices, window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close_prices, window=50).sma_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(close_prices, window=20).ema_indicator()
    df['Volume_SMA_20'] = ta.trend.SMAIndicator(volume, window=20).sma_indicator()

    df.dropna(inplace=True)
    return df

# Step 3: Identify Local Maxima and Minima to Create Labels
def identify_local_extrema(df, order=5):
    df['Min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=order)[0]]['Close']
    df['Max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]]['Close']
    df['Trade_Label'] = 0
    df.loc[df['Min'].notna(), 'Trade_Label'] = 1  # Buy signal
    df.loc[df['Max'].notna(), 'Trade_Label'] = -1  # Sell signal
    
    # Count the number of buy and sell signals
    buy_count = df['Trade_Label'].value_counts().get(1, 0)
    sell_count = df['Trade_Label'].value_counts().get(-1, 0)
    
    print(f"Number of Buy signals: {buy_count}")
    print(f"Number of Sell signals: {sell_count}")
    
    return df

# Step 4: Prepare Features and Labels
def prepare_data(df):
    X = df[['RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50', 'EMA_20', 'Volume_SMA_20']]
    y = df['Trade_Label']
    return X, y

# Step 5: Handle Class Imbalance with Oversampling
def handle_class_imbalance(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print("Resampled training labels distribution:")
    print(pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

# Step 6: Train the Model
def train_model(X, y, df):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Handle class imbalance in the training set
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    # Define the hyperparameters to be tuned
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    # Initialize GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    # Fit the model to the resampled training data
    grid_search.fit(X_train_resampled, y_train_resampled)
    # Get the best model
    model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = model.predict(X_test)
    # Print the predictions
    print("Predictions on the test set:")
    print(y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print evaluation metrics
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Precision: {precision:.2f}")
    print(f"Model Recall: {recall:.2f}")
    print(f"Model F1 Score: {f1:.2f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Make predictions for the entire dataset
    df['Trade_Flag'] = model.predict(df[['RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50', 'EMA_20', 'Volume_SMA_20']])

    # Print the entire range of predictions for debugging
    print("Predictions for the entire dataset:")
    print(df[['Trade_Flag']])

    # Print the distribution of predicted labels
    print("Distribution of predicted labels:")
    print(df['Trade_Flag'].value_counts())

    return model

# Step 7: Verify Trade Pairs
def verify_trade_pairs(df):
    buy_indices = df[df['Trade_Flag'] == 1].index
    sell_indices = df[df['Trade_Flag'] == -1].index

    # Print the buy and sell indices
    print("Buy indices found by the model:")
    print(buy_indices)
    print("Sell indices found by the model:")
    print(sell_indices)

    buy_sell_pairs = []

    for buy_idx in buy_indices:
        # Find the first sell index that comes after the buy index
        sell_idx = sell_indices[sell_indices > buy_idx]
        if not sell_idx.empty:
            sell_idx = sell_idx[0]
            buy_sell_pairs.append((buy_idx, sell_idx))
            print(f"buy_idx= {buy_idx} sell_idx= {sell_idx}")

    # Initialize the 'Profit' column
    df['Profit'] = 0.0

    # Calculate profit/loss for each pair
    for buy_idx, sell_idx in buy_sell_pairs:
        # Directly use the datetime labels with .loc[]
        df.loc[sell_idx, 'Profit'] = df.loc[sell_idx, 'Close'] - df.loc[buy_idx, 'Close']

    # Identify faulty flags (pairs that result in a loss)
    df['Faulty_Flag'] = 0
    print("df=", df)
    for buy_idx, sell_idx in buy_sell_pairs:
        if df.loc[sell_idx, 'Profit'].item() < 0:
            df.loc[buy_idx, 'Faulty_Flag'] = 1
            df.loc[sell_idx, 'Faulty_Flag'] = 1

    return df, buy_sell_pairs

# Step 8: Simulate Trading
def simulate_trading(df, initial_capital, buy_sell_pairs):
    capital = initial_capital
    trades = []

    for buy_idx, sell_idx in buy_sell_pairs:
        buy_price = df.loc[buy_idx, 'Close'].item()
        sell_price = df.loc[sell_idx, 'Close'].item()
        shares = capital / buy_price
        trade_profit = shares * (sell_price - buy_price)
        trade_return = (trade_profit / capital) * 100
        capital += trade_profit

        trades.append({
            'Buy Date': buy_idx,
            'Sell Date': sell_idx,
            'Buy Price': buy_price,
            'Sell Price': sell_price,
            'Profit': trade_profit,
            'Return (%)': trade_return
        })

    return trades, capital

# Step 9: Plot Results
def plot_results(df, trades):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    for trade in trades:
        if trade['Profit'] > 0:
            plt.scatter(trade['Buy Date'], trade['Buy Price'], marker='^', color='g', label='Buy Signal', s=100)
            plt.scatter(trade['Sell Date'], trade['Sell Price'], marker='v', color='r', label='Sell Signal', s=100)
        else:
            plt.scatter(trade['Buy Date'], trade['Buy Price'], marker='^', color='orange', label='Faulty Buy Signal', s=100)
            plt.scatter(trade['Sell Date'], trade['Sell Price'], marker='v', color='purple', label='Faulty Sell Signal', s=100)
    plt.legend()
    plt.title('Price Chart with Trade Signals')
    plt.show()

# Example usage:
def main():
    ticker = 'AAPL'
    start_date = "2024-01-01"
    end_date = "2025-02-20"
    initial_capital = 30000

    data = get_stock_data(ticker, start_date, end_date)
    data = add_technical_indicators(data)
    data = identify_local_extrema(data)
    X, y = prepare_data(data)
    model = train_model(X, y, data)

    data, buy_sell_pairs = verify_trade_pairs(data)
    
    # Print the last few rows of the data to ensure predictions cover the entire dataset
    print(data.tail())

    trades, final_capital = simulate_trading(data, initial_capital, buy_sell_pairs)

    print(f"Initial Capital: ${initial_capital}")
    print(f"Final Capital: ${final_capital}")
    print("Trades Summary:")
    for trade in trades:
        print(trade)

    plot_results(data, trades)

if __name__ == "__main__":
    main()