import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Download Stock Data
def get_stock_data(ticker, start_date, end_date):
    file_path = f'./data/{ticker}_data.csv'  # Construct file path based on ticker
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)  # Read CSV with datetime index
    #data.sort_index(inplace=True)  # Ensure chronological order
    return data

# Step 2: Calculate Technical Indicators
def add_technical_indicators(df):
    close_prices = df['Close'].squeeze()
    volume = df['Volume'].squeeze()

    df['RSI'] = ta.momentum.RSIIndicator(close_prices, window=14).rsi()
    df['MACD'] = ta.trend.MACD(close_prices).macd()
    df['MACD_signal'] = ta.trend.MACD(close_prices).macd_signal()
    df['SMA_20'] = ta.trend.SMAIndicator(close_prices, window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close_prices, window=50).sma_indicator()
    df['Volume_SMA_20'] = ta.trend.SMAIndicator(volume, window=20).sma_indicator()

    df.dropna(inplace=True)
    return df

# Step 3: Mark Buy/Sell Points Using MACD
def mark_buy_sell_points(df):
    df['Buy_Signal_MACD'] = np.where(df['MACD'] > df['MACD_signal'], 1, 0)
    df['Sell_Signal_MACD'] = np.where(df['MACD'] < df['MACD_signal'], -1, 0)
    return df

# Step 4: Calculate Profits for Each Pair
def calculate_profits(df):
    buy_signals = df[df['Buy_Signal_MACD'] == 1].index
    sell_signals = df[df['Sell_Signal_MACD'] == -1].index
    df['Profit_MACD'] = 0.0

    for buy_idx in buy_signals:
        sell_idx = sell_signals[sell_signals > buy_idx]
        if not sell_idx.empty:
            sell_idx = sell_idx[0]
            df.loc[sell_idx, 'Profit_MACD'] = df.loc[sell_idx, 'Close'] - df.loc[buy_idx, 'Close']

    return df

# Step 5: Correlation Analysis
def correlation_analysis(df):
    df['Profit_Flag'] = np.where(df['Profit_MACD'] > 0, 1, 0)
    correlation_matrix = df[['Profit_Flag', 'RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50', 'Volume_SMA_20']].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)
    return df

# Step 6: Prepare Features and Labels for Model Training
def prepare_data_for_model(df):
    X = df[['RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50', 'Volume_SMA_20']]
    y = df['Profit_Flag']
    return X, y

# Step 7: Train the Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Precision: {precision:.2f}")
    print(f"Model Recall: {recall:.2f}")
    print(f"Model F1 Score: {f1:.2f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model

# Step 8: Verify Profit Earned by Following the Model
def verify_profit(df, model):
    df['Trade_Flag'] = model.predict(df[['RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50', 'Volume_SMA_20']])
    
    buy_indices = df[df['Trade_Flag'] == 1].index
    sell_indices = df[df['Trade_Flag'] == -1].index

    buy_sell_pairs = []
    for buy_idx in buy_indices:
        sell_idx = sell_indices[sell_indices > buy_idx]
        if not sell_idx.empty:
            sell_idx = sell_idx[0]
            buy_sell_pairs.append((buy_idx, sell_idx))

    df['Profit'] = 0.0
    for buy_idx, sell_idx in buy_sell_pairs:
        df.loc[sell_idx, 'Profit'] = df.loc[sell_idx, 'Close'] - df.loc[buy_idx, 'Close']

    return df, buy_sell_pairs

# Step 9: Simulate Trading
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

# Step 10: Plot Results
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

# Main Function
def main():
    ticker = 'AAPL'
    start_date = "2024-01-01"
    end_date = "2025-02-20"
    initial_capital = 30000

    data = get_stock_data(ticker, start_date, end_date)
    data = add_technical_indicators(data)
    data = mark_buy_sell_points(data)
    data = calculate_profits(data)
    data = correlation_analysis(data)
    X, y = prepare_data_for_model(data)
    model = train_model(X, y)

    data, buy_sell_pairs = verify_profit(data, model)
    
    trades, final_capital = simulate_trading(data, initial_capital, buy_sell_pairs)

    print(f"Initial Capital: ${initial_capital}")
    print(f"Final Capital: ${final_capital}")
    print("Trades Summary:")
    for trade in trades:
        print(trade)

    plot_results(data, trades)

if __name__ == "__main__":
    main()