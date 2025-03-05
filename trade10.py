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
from datetime import datetime
import tqdm  # For progress bars
import matplotlib.pyplot as plt
import sys  # For command-line arguments

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configurable parameters
stock_symbol = "AAPL"
sequence_length = 10  # Fixed for test mode
threshold = 0.03  # 1% threshold

# Load your dataset directly (CSV format)
try:
    data = pd.read_csv("./data/AAPL_data.csv", parse_dates=['Date'], index_col='Date')
except FileNotFoundError:
    print("Error: aapl_data.csv not found. Please ensure the file exists in the current directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading aapl_data.csv: {e}")
    sys.exit(1)

# Debug: Check the loaded DataFrame
print("Columns after loading:", data.columns.tolist())
print("Shape after loading:", data.shape)

# Ensure Date index is correct and handle potential format issues
if data.empty:
    print("Warning: DataFrame is empty after loading. Attempting to load without index_col...")
    data = pd.read_csv("./data/AAPL_data.csv", parse_dates=['Date'])
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    else:
        print("Error: 'Date' column not found in aapl_data.csv. Please check the CSV format.")
        sys.exit(1)

start_date = "2023-03-03"  # First date in your data
end_date = "2025-03-04"  # Extend to include March 4, 2025

# Tagging method 4 (only used for simplicity)
def tagging_method_4(data, i, sequence_length, threshold):
    if i + sequence_length >= len(data):
        return None  # No label for the last sequence_length days
    future_start = i + 2
    if future_start >= len(data):
        future_start = i + 1
        if future_start >= len(data):
            return None
    
    max_future = min(i + sequence_length + 1, len(data))
    first_rise, first_drop = None, None
    
    for j in range(future_start, max_future):
        current_return = (data['Close'].iloc[j] - data['Close'].iloc[i]) / data['Close'].iloc[i]
        if current_return > threshold and first_rise is None:
            first_rise = current_return
            print(f"Found a rise at {data.index[j]}, Close={data['Close'].iloc[j]}, Return={current_return}")
            return 1  # Buy
        elif current_return < -threshold and first_drop is None:
            first_drop = current_return
            print(f"Found a drop at {data.index[j]}, Close={data['Close'].iloc[j]}, Return={current_return}")
            return -1  # Sell
    
    return 0  # Hold

# Load and preprocess data, filling NaN values
def load_and_preprocess_data(data, start_date, end_date):
    data = data.sort_index(ascending=True).drop_duplicates()
    data = data.loc[start_date:end_date]  # Ensure full range
    dates = data.index

    if data.empty:
        print(f"Warning: No data found between {start_date} and {end_date}. Check date format or CSV content.")
        sys.exit(1)

    print(f"Data range: {dates[0]} to {dates[-1]}")
    print(f"Initial data shape: {data.shape}")  # Should be 600 rows

    # Fill NaN values with 0 (though your data appears clean)
    data['Close'] = data['Close'].fillna(0)
    data['Volume'] = data['Volume'].fillna(0)
    data['Open'] = data['Open'].fillna(0)
    data['High'] = data['High'].fillna(0)
    data['Low'] = data['Low'].fillna(0)
    if 'Adj Close' in data.columns:
        data['Adj Close'] = data['Adj Close'].fillna(0)  # Handle Adj Close if present

    # Calculate returns and lag
    data['Returns'] = data['Close'].pct_change().fillna(0)
    data['Lag1'] = data['Returns'].shift(1).fillna(0)

    # Add technical indicators, filling NaN with 0
    data['RSI'] = ta.rsi(data['Close'], length=14).fillna(0)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data['MACD'] = macd['MACD_12_26_9'].fillna(0)
    data['MACD_Signal'] = macd['MACDs_12_26_9'].fillna(0)
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14).fillna(0)
    data['ROC'] = ta.roc(data['Close'], length=10).fillna(0)

    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']
    full_data = data[features].copy()  # Use only required features, drop 'Adj Close' if present
    full_data['lstm_flag'] = np.nan
    full_data['macd_flag'] = np.nan
    full_data['lstm_flag'] = full_data['lstm_flag'].astype('object')
    full_data['macd_flag'] = full_data['macd_flag'].astype('object')

    print(f"Processed data shape: {full_data.shape}")
    print(f"NaN check in full_data:\n{full_data.isna().sum()}")
    return full_data, dates

# Prepare training and test data
def prepare_training_test_data(full_data, sequence_length, threshold):
    X, y, sequence_starts = [], [], []
    labels = []  # To store all labels for plotting and printing
    print(f"Preparing sequences for sequence_length={sequence_length}")
    for i in range(len(full_data) - sequence_length):
        label = tagging_method_4(full_data, i, sequence_length, threshold)
        date = full_data.index[i]
        close = full_data['Close'].iloc[i]
        print(f"Index {i}, Date: {date}, Label: {label}, Close: {close}")
        labels.append((date, label, close))  # Store date, label, and close for all rows
        sequence = full_data[features].iloc[i:i + sequence_length].fillna(0).values
        X.append(sequence)
        y.append(label if label is not None else 0)  # Default to 0 (hold) for predictions
        sequence_starts.append(i)
    
    # Print all labels
    print("\nAll Labels (Date, Label, Close):")
    for date, label, close in labels:
        signal = "buy" if label == 1 else "sell" if label == -1 else "hold"
        print(f"Date: {date}, Signal: {signal}, Close: {close}")

    print(f"Before filtering - X shape: {len(X)}, y shape: {len(y)}, sequence_starts shape: {len(sequence_starts)}")
    X, y, sequence_starts = np.array(X), np.array(y), np.array(sequence_starts)
    
    # No NaN filtering; use all data
    X_scaled = np.zeros_like(X, dtype=float)
    for i in range(X.shape[2]):
        scaler = RobustScaler()
        X_scaled[:, :, i] = scaler.fit_transform(X[:, :, i])

    print(f"X_scaled shape: {X_scaled.shape}")
    if len(X_scaled) > 0:
        train_size = max(1, int(len(X_scaled) * 0.8))  # Ensure at least 1 test sample
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        train_sequence_starts, test_sequence_starts = sequence_starts[:train_size], sequence_starts[train_size:]
    else:
        X_train, X_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])
        train_sequence_starts, test_sequence_starts = np.array([]), np.array([])

    print(f"After split - Train size: {len(X_train)}, Test size: {len(X_test)}, Test sequence_starts: {len(test_sequence_starts)}")

    # Prepare the last sequence_length days for prediction
    last_sequence = full_data[features].iloc[len(full_data) - sequence_length:].fillna(0).values
    last_sequence_scaled = np.zeros((1, sequence_length, len(features)), dtype=float)
    for i in range(len(features)):
        scaler = RobustScaler()
        last_sequence_scaled[0, :, i] = scaler.fit_transform(last_sequence[:, i].reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts, last_sequence_scaled, labels

# Build and train LSTM model
def build_and_train_lstm(X_train, y_train, X_test, y_test, sequence_length, last_sequence_scaled):
    if len(X_train) == 0 or len(y_train) == 0:
        return None, [], [], []
    
    y_train_cat = to_categorical(y_train + 1, num_classes=3)  # -1, 0, 1 -> 0, 1, 2
    y_test_cat = to_categorical(y_test + 1, num_classes=3) if len(y_test) > 0 else np.array([])
    
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=False), input_shape=(sequence_length, len(features))),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=Nadam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train_cat, 
        epochs=50,  # Reverted to 50 epochs
        batch_size=4,  # Reverted to original batch size
        validation_data=(X_test, y_test_cat) if len(X_test) > 0 else None,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ],
        class_weight={0: 1.0, 1: 2.0, 2: 1.0},  # Balance classes
        verbose=1  # Show training progress
    )
    
    # Predict on test set
    test_predictions = model.predict(X_test, verbose=0) if len(X_test) > 0 else np.array([])
    lstm_signals_test = ["sell" if np.argmax(p) == 0 else "hold" if np.argmax(p) == 1 else "buy" 
                         for p in test_predictions] if len(X_test) > 0 else []
    
    # Predict for the last sequence_length days
    last_predictions = model.predict(last_sequence_scaled, verbose=0)
    lstm_signals_last = ["sell" if np.argmax(p) == 0 else "hold" if np.argmax(p) == 1 else "buy" 
                         for p in last_predictions]
    
    return model, lstm_signals_test, lstm_signals_last, y_test

# Evaluate MACD signals
def evaluate_macd(full_data, test_sequence_starts, y_test):
    macd_flags = []
    for i in range(1, len(full_data)):
        prev_macd, prev_signal = full_data['MACD'].iloc[i - 1], full_data['MACD_Signal'].iloc[i - 1]
        current_macd, current_signal = full_data['MACD'].iloc[i], full_data['MACD_Signal'].iloc[i]
        if prev_macd <= prev_signal and current_macd > current_signal:
            macd_flags.append("buy")
        elif prev_macd >= prev_signal and current_macd < current_signal:
            macd_flags.append("sell")
        else:
            macd_flags.append(np.nan)
    macd_flags = [np.nan] + macd_flags
    
    valid_macd_flags = [f for f in macd_flags if not pd.isna(f)]
    valid_indices = [i for i, f in enumerate(macd_flags) if not pd.isna(f)]
    if len(valid_macd_flags) > 0 and len(valid_indices) <= len(y_test) and len(y_test) > 0:
        y_test_aligned = y_test[[i for i in range(len(y_test)) if test_sequence_starts[i] - test_sequence_starts[0] in valid_indices]]
        macd_accuracy = sum(1 for p, a in zip(valid_macd_flags, y_test_aligned) if 
                            (p == "buy" and a == 1) or (p == "sell" and a == -1)) / len(valid_macd_flags)
    else:
        macd_accuracy = 0
    return macd_flags, macd_accuracy

# Plotting function to show all days' closing prices and labels
def plot_signals(full_data, dates, sequence_length, test_sequence_starts, lstm_signals_test, lstm_signals_last, macd_flags, config, all_labels):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top subplot: Price with all labels (buy/sell/hold) for all days
    ax1.plot(full_data.index, full_data['Close'], label="Price", color="blue")
    price_range = full_data['Close'].max() - full_data['Close'].min()
    offset = price_range * 0.02

    # Plot all labels from the dataset (rows 0 to 589)
    for date, label, close in all_labels:
        if label == 1:  # Buy
            ax1.scatter(date, close + offset, marker='^', color='green', label='Buy' if date == all_labels[0][0] else "", s=100)
        elif label == -1:  # Sell
            ax1.scatter(date, close - offset, marker='v', color='red', label='Sell' if date == all_labels[0][0] else "", s=100)
        elif label == 0:  # Hold
            ax1.scatter(date, close, marker='o', color='gray', label='Hold' if date == all_labels[0][0] else "", s=50, alpha=0.5)

    # Plot LSTM test signals (if available)
    if len(test_sequence_starts) > 0 and len(lstm_signals_test) > 0:
        signal_indices = test_sequence_starts + sequence_length
        test_dates = full_data.index[signal_indices]
        for i, signal in enumerate(lstm_signals_test):
            price = full_data['Close'].iloc[signal_indices[i]]
            if signal == "buy":
                ax1.scatter(test_dates[i], price + offset, marker='^', color='green', 
                            label='LSTM Buy' if i == 0 else "", s=100, edgecolor='black')
            elif signal == "sell":
                ax1.scatter(test_dates[i], price - offset, marker='v', color='red', 
                            label='LSTM Sell' if i == 0 else "", s=100, edgecolor='black')

    # Plot LSTM predictions for the last sequence_length days
    last_start_idx = len(full_data) - sequence_length
    last_dates = full_data.index[last_start_idx:]
    for i, signal in enumerate(lstm_signals_last):
        price = full_data['Close'].iloc[last_start_idx + i]
        if signal == "buy":
            ax1.scatter(last_dates[i], price + offset, marker='^', color='green', 
                        label='LSTM Buy Prediction' if i == 0 else "", s=100, edgecolor='black')
        elif signal == "sell":
            ax1.scatter(last_dates[i], price - offset, marker='v', color='red', 
                        label='LSTM Sell Prediction' if i == 0 else "", s=100, edgecolor='black')

    ax1.set_title(f"Price and Signals (Sequence Length {config['Sequence Length']}, Method 4)")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # Bottom subplot: MACD with MACD Buy/Sell Flags
    ax2.plot(full_data.index, full_data['MACD'], label="MACD", color="blue", linestyle='--')
    ax2.plot(full_data.index, full_data['MACD_Signal'], label="MACD Signal", color="orange")
    macd_range = max(full_data['MACD'].max(), full_data['MACD_Signal'].max()) - min(full_data['MACD'].min(), full_data['MACD_Signal'].min())
    macd_offset = macd_range * 0.02

    for i in range(1, len(full_data)):
        if full_data['macd_flag'].iloc[i] == "buy":
            ax2.scatter(full_data.index[i], full_data['MACD'].iloc[i] + macd_offset, 
                        marker='^', color='green', label='MACD Buy' if i == 1 else "", s=100)
        elif full_data['macd_flag'].iloc[i] == "sell":
            ax2.scatter(full_data.index[i], full_data['MACD'].iloc[i] - macd_offset, 
                        marker='v', color='red', label='MACD Sell' if i == 1 else "", s=100)

    ax2.set_title(f"MACD and Signals (Sequence Length {config['Sequence Length']}, Method 4)")
    ax2.set_ylabel("MACD Value")
    ax2.legend()
    ax2.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main execution for debugging (test mode with Method 4)
if __name__ == "__main__":
    test_single = True  # Force test mode
    full_data, dates = load_and_preprocess_data(data, start_date, end_date)
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']

    X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts, last_sequence_scaled, all_labels = prepare_training_test_data(
        full_data, sequence_length, threshold
    )
    print(f"Train labels: {y_train[:10]}...")  # First 10 labels for debugging
    print(f"Test labels: {y_test[:10]}...")  # First 10 test labels

    model, lstm_signals_test, lstm_signals_last, y_test = build_and_train_lstm(
        X_train, y_train, X_test, y_test, sequence_length, last_sequence_scaled
    )

    # Assign signals for debugging
    for idx, signal in zip(test_sequence_starts, lstm_signals_test):
        full_data.iloc[idx + sequence_length, full_data.columns.get_loc('lstm_flag')] = signal
    
    last_start_idx = len(full_data) - sequence_length
    for i, signal in enumerate(lstm_signals_last):
        full_data.iloc[last_start_idx + i, full_data.columns.get_loc('lstm_flag')] = signal + " Prediction"

    # Evaluate and assign MACD signals
    macd_flags, _ = evaluate_macd(full_data, test_sequence_starts, y_test)
    for i, flag in enumerate(macd_flags[1:]):  # Skip first NaN
        if not pd.isna(flag):
            full_data.iloc[i + 1, full_data.columns.get_loc('macd_flag')] = flag

    config = {"Sequence Length": sequence_length, "Tagging Method": 4}
    plot_signals(full_data, dates, sequence_length, test_sequence_starts, lstm_signals_test, lstm_signals_last, macd_flags, config, all_labels)

    # Print final signals for verification
    print("\nFinal LSTM Flags:")
    print(full_data['lstm_flag'].tail(20))  # Last 20 rows to check predictions
    print("\nFinal MACD Flags:")
    print(full_data['macd_flag'].tail(20))  # Last 20 rows to check MACD signals