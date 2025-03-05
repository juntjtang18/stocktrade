import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import pandas_ta as ta
import tensorflow as tf
import os
from mylib.download_data import download_data
from datetime import datetime
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configurable parameters
stock_symbol = "AAPL"
data_dir = os.path.join(os.path.dirname(__file__), "data")
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")  # March 5, 2025
sequence_length = 10
threshold = 0.01
tagging_method = 3

# Tagging Method 3 (Slope-based)
def tagging_method_3(data, i, sequence_length, threshold):
    if i + sequence_length >= len(data):
        return None
    prices = data['Close'].iloc[i:i + sequence_length]
    slope, _, _, _, _ = linregress(range(len(prices)), prices)
    price_std = prices.std()
    boundary = threshold * price_std / len(prices)
    if slope > boundary:
        return 1  # Buy
    elif slope < -boundary:
        return -1  # Sell
    return 0  # Hold

# Load and preprocess data
def load_and_preprocess_data(stock_symbol, start_date, end_date):
    data = download_data(stock_symbol, source="polygon")
    data = data.sort_index(ascending=True).drop_duplicates()
    data = data[data['Volume'] > 0].loc[start_date:end_date]
    dates = data.index
    print(f"Data range: {dates[0]} to {dates[-1]}")
    
    data['Returns'] = data['Close'].pct_change()
    data['Lag1'] = data['Returns'].shift(1)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_Signal'] = macd['MACDs_12_26_9']
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data['ROC'] = ta.roc(data['Close'], length=10)
    
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']
    full_data = data[features].copy()
    full_data['lstm_flag'] = np.nan
    full_data['lstm_flag'] = full_data['lstm_flag'].astype('object')
    return full_data, dates

# Prepare training and test data
def prepare_training_test_data(full_data, sequence_length, threshold):
    X, y, sequence_starts = [], [], []
    for i in range(len(full_data) - sequence_length):
        label = tagging_method_3(full_data, i, sequence_length, threshold)
        if label is None:
            continue
        X.append(full_data[features].iloc[i:i + sequence_length].values)
        y.append(label)
        sequence_starts.append(i)
    
    X, y, sequence_starts = np.array(X), np.array(y), np.array(sequence_starts)
    valid_indices = ~np.isnan(X).any(axis=(1, 2))
    X, y, sequence_starts = X[valid_indices], y[valid_indices], sequence_starts[valid_indices]

    # Scale features
    X_scaled = np.zeros_like(X)
    for i in range(X.shape[2]):
        scaler = RobustScaler()
        X_scaled[:, :, i] = scaler.fit_transform(X[:, :, i])

    # Split into training and test sets
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    train_sequence_starts, test_sequence_starts = sequence_starts[:train_size], sequence_starts[train_size:]

    # Print class distribution
    print("\nClass Distribution:")
    print(f"Training - Buy (1): {np.sum(y_train == 1)}, Sell (-1): {np.sum(y_train == -1)}, Hold (0): {np.sum(y_train == 0)}")
    print(f"Test - Buy (1): {np.sum(y_test == 1)}, Sell (-1): {np.sum(y_test == -1)}, Hold (0): {np.sum(y_test == 0)}")
    
    return X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts

# Build and train LSTM model
def build_and_train_lstm(X_train, y_train, X_test, sequence_length):
    if len(X_train) == 0 or len(y_train) == 0:
        print("No training data available.")
        return None, [], []
    
    y_train_cat = to_categorical(y_train + 1, num_classes=3)
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=False), input_shape=(sequence_length, len(features))),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Nadam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=4,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ],
        class_weight={0: 1.0, 1: 2.0, 2: 1.0},
        verbose=1
    )
    
    test_predictions = model.predict(X_test, verbose=0) if len(X_test) > 0 else np.array([])
    lstm_signals_test = [np.argmax(p) - 1 for p in test_predictions] if len(X_test) > 0 else []
    return model, lstm_signals_test, history

# Predict for the last 10 days
def predict_last_10_days(model, full_data, dates, sequence_length):
    last_start_idx = len(full_data) - sequence_length - 9
    last_sequences = []
    for i in range(last_start_idx, len(full_data) - sequence_length + 1):
        sequence = full_data[features].iloc[i:i + sequence_length].values
        scaled_sequence = np.zeros((1, sequence_length, len(features)))
        for j in range(len(features)):
            scaler = RobustScaler()
            scaled_sequence[0, :, j] = scaler.fit_transform(sequence[:, j].reshape(-1, 1)).flatten()
        last_sequences.append(scaled_sequence)
    
    last_sequences = np.vstack(last_sequences)
    last_predictions = model.predict(last_sequences, verbose=0)
    lstm_signals_last = [np.argmax(p) - 1 for p in last_predictions]
    return lstm_signals_last

# Plotting function with future trend check
def plot_results(dates, full_data, train_sequence_starts, y_train, test_sequence_starts, y_test, lstm_signals_test, lstm_signals_last, sequence_length):
    plt.figure(figsize=(14, 8))
    plt.plot(dates, full_data['Close'], label='Closing Price', color='blue', linewidth=1)
    
    price_range = full_data['Close'].max() - full_data['Close'].min()
    offset = price_range * 0.02

    # Training tags
    train_dates = dates[train_sequence_starts + sequence_length]
    train_prices = full_data['Close'].iloc[train_sequence_starts + sequence_length]
    for date, price, tag in zip(train_dates, train_prices, y_train):
        if tag == 1:
            plt.scatter(date, price + offset, marker='^', color='green', label='Train Buy' if tag == y_train[0] else '', s=50)
        elif tag == -1:
            plt.scatter(date, price - offset, marker='v', color='red', label='Train Sell' if tag == y_train[0] else '', s=50)

    # Test tags and predictions
    test_dates = dates[test_sequence_starts + sequence_length]
    test_prices = full_data['Close'].iloc[test_sequence_starts + sequence_length]
    for date, price, tag, pred in zip(test_dates, test_prices, y_test, lstm_signals_test):
        if tag == 1:
            plt.scatter(date, price + offset, marker='^', color='darkgreen', label='Test Buy (Actual)' if tag == y_test[0] else '', s=50)
        elif tag == -1:
            plt.scatter(date, price - offset, marker='v', color='darkred', label='Test Sell (Actual)' if tag == y_test[0] else '', s=50)
        if pred == 1:
            plt.scatter(date, price + offset * 2, marker='^', color='lime', label='Test Buy (Pred)' if pred == lstm_signals_test[0] else '', s=50)
        elif pred == -1:
            plt.scatter(date, price - offset * 2, marker='v', color='pink', label='Test Sell (Pred)' if pred == lstm_signals_test[0] else '', s=50)

    # Last 10 days predictions
    last_10_days = dates[-10:]
    last_10_prices = full_data['Close'].iloc[-10:]
    for date, price, pred in zip(last_10_days, last_10_prices, lstm_signals_last):
        if pred == 1:
            plt.scatter(date, price + offset * 3, marker='^', color='cyan', label='Final Buy (Pred)' if pred == lstm_signals_last[0] else '', s=50)
        elif pred == -1:
            plt.scatter(date, price - offset * 3, marker='v', color='magenta', label='Final Sell (Pred)' if pred == lstm_signals_last[0] else '', s=50)

    plt.title(f"{stock_symbol} Closing Price and Tags (Sequence Length: {sequence_length}, Method: {tagging_method})")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main execution
print("Loading and preprocessing data...")
full_data, dates = load_and_preprocess_data(stock_symbol, start_date, end_date)
features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']

print(f"\nPreparing data with Sequence Length: {sequence_length}, Tagging Method: {tagging_method}")
X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts = prepare_training_test_data(
    full_data, sequence_length, threshold
)

if len(X_train) == 0 or len(X_test) == 0:
    print("No valid data to process.")
else:
    # Sample tagging debug
    print("\nSample Tagging Check:")
    for i in [0, len(full_data)//2, len(full_data)-sequence_length-1]:
        prices = full_data['Close'].iloc[i:i + sequence_length].values
        slope, _, _, _, _ = linregress(range(sequence_length), prices)
        tag = tagging_method_3(full_data, i, sequence_length, threshold)
        print(f"Index {i}: Prices {prices}, Slope {slope:.4f}, Tag {tag}")

    print("\nTraining LSTM model...")
    model, lstm_signals_test, history = build_and_train_lstm(X_train, y_train, X_test, sequence_length)

    # Test accuracy
    correct_predictions = 0
    for p, a in zip(lstm_signals_test, y_test):
        if p == a:
            correct_predictions += 1
    test_accuracy = correct_predictions / len(y_test) if len(y_test) > 0 else 0.0
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Save test results to CSV
    test_dates = dates[test_sequence_starts + sequence_length]
    test_results = pd.DataFrame({
        "Date": test_dates,
        "Closing Price": full_data['Close'].iloc[test_sequence_starts + sequence_length].values,
        "Actual Tag": y_test,
        "Predicted Signal": lstm_signals_test
    })
    test_results.to_csv("test_results.csv", index=False)
    print("\nTest results saved to 'test_results.csv'")

    # Predictions for the last 10 days
    lstm_signals_last = predict_last_10_days(model, full_data, dates, sequence_length)
    last_10_days = dates[-10:]
    last_10_prices = full_data['Close'].iloc[-10:]
    print("\nPredictions for the last 10 days (2025-02-19 to 2025-03-04):")
    print("Date       | Closing Price | Prediction")
    print("----------------------------------------")
    for date, price, signal in zip(last_10_days, last_10_prices, lstm_signals_last):
        print(f"{date.strftime('%Y-%m-%d')} | {price:>12.2f} | {signal:>3d}")

    # Plot the results
    plot_results(dates, full_data, train_sequence_starts, y_train, test_sequence_starts, y_test, lstm_signals_test, lstm_signals_last, sequence_length)