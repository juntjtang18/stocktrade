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
from mylib.download_data import download_data  # Import your download_data function
from datetime import datetime
import tqdm  # For progress bars
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys  # For command-line arguments

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configurable parameters
stock_symbol = "AAPL"
data_dir = os.path.join(os.path.dirname(__file__), "data")
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")  # March 4, 2025
sequence_lengths = [5, 10, 15]  # Reduced to specific lengths
threshold = 0.01  # 1% threshold
initial_amount = 10000  # $10,000 starting capital

# Command-line argument for testing a single configuration
test_single = len(sys.argv) > 1 and sys.argv[1] == "--test"
if test_single:
    sequence_lengths = [10]  # Test with sequence length 10
    test_method = 3  # Test with Method 3 (highest accuracy from your results)

# Tagging methods
def tagging_method_1(data, i, sequence_length, threshold):
    future_idx = i + sequence_length
    if future_idx >= len(data):
        return None
    future_return = (data['Close'].iloc[future_idx] - data['Close'].iloc[i]) / data['Close'].iloc[i]
    if future_return > threshold:
        return 1  # Buy
    elif future_return < -threshold:
        return -1  # Sell
    return 0  # Hold

def tagging_method_2(data, i, sequence_length, threshold):
    if i + sequence_length >= len(data):
        return None
    moving_avg = data['Close'].iloc[i + 1:i + sequence_length + 1].mean()
    future_return = (moving_avg - data['Close'].iloc[i]) / data['Close'].iloc[i]
    if future_return > threshold:
        return 1  # Buy
    elif future_return < -threshold:
        return -1  # Sell
    return 0  # Hold

def tagging_method_3(data, i, sequence_length, threshold):
    if i + sequence_length >= len(data):
        return None
    prices = data['Close'].iloc[i:i + sequence_length]
    slope, _, _, _, _ = linregress(range(len(prices)), prices)
    price_std = prices.std()
    if slope > threshold * price_std / len(prices):  # Adjust threshold based on scale
        return 1  # Buy
    elif slope < -threshold * price_std / len(prices):
        return -1  # Sell
    return 0  # Hold

def tagging_method_4(data, i, sequence_length, threshold):  # Opportunity-Based (Method 9 from previous)
    if i + sequence_length >= len(data):
        return None
    # Find the earliest future day starting from i + 2 (buffer for trading)
    future_start = i + 2
    if future_start >= len(data):
        # If i + 2 is beyond data, use the last available day (i + 1)
        future_start = i + 1
        if future_start >= len(data):
            return None
    
    # Look for the first significant movement from i to future_start or later within sequence_length
    max_future = min(i + sequence_length + 1, len(data))
    first_rise = None
    first_drop = None
    
    for j in range(future_start, max_future):
        current_return = (data['Close'].iloc[j] - data['Close'].iloc[i]) / data['Close'].iloc[i]
        if current_return > threshold and first_rise is None:
            first_rise = current_return
        elif current_return < -threshold and first_drop is None:
            first_drop = current_return
    
    # Handle consecutive minor drops (optional, adjust parameters as needed)
    if first_rise is None and first_drop is None:
        # Check for consecutive minor drops (e.g., 3+ drops summing to > -threshold)
        minor_drops = []
        for j in range(future_start, max_future):
            current_return = (data['Close'].iloc[j] - data['Close'].iloc[i]) / data['Close'].iloc[i]
            if current_return < 0 and abs(current_return) < threshold:  # Minor drop
                minor_drops.append(current_return)
            else:
                minor_drops = []  # Reset if a non-drop or significant drop occurs
            if len(minor_drops) >= 3 and sum(minor_drops) < -threshold:  # 3 minor drops summing to > -1%
                return -1  # Sell due to trend of minor drops
    
    # Label based on first significant movement
    if first_rise is not None:
        return 1  # Buy (first rise, even if drop follows later)
    elif first_drop is not None:
        return -1  # Sell (first drop, even if rise follows later)
    return 0  # Hold (no significant movement)

# Load and preprocess data
def load_and_preprocess_data(stock_symbol, start_date, end_date):
    # Use download_data without force_update, relying on local or API update logic
    data = download_data(stock_symbol, source="polygon")
    data = data.sort_index(ascending=True).drop_duplicates()
    data = data[data['Volume'] > 0].loc[start_date:end_date]
    dates = data.index

    # Print data range for debugging
    print(f"Data range: {dates[0]} to {dates[-1]}")

    data['Returns'] = data['Close'].pct_change()
    data['Lag1'] = data['Returns'].shift(1)

    # Add technical indicators
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_Signal'] = macd['MACDs_12_26_9']
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data['ROC'] = ta.roc(data['Close'], length=10)

    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']
    full_data = data[features].copy()
    # Create lstm_flag and macd_flag as object dtype columns with NaN
    full_data['lstm_flag'] = np.nan
    full_data['macd_flag'] = np.nan
    # Ensure they are object dtype to handle strings
    full_data['lstm_flag'] = full_data['lstm_flag'].astype('object')
    full_data['macd_flag'] = full_data['macd_flag'].astype('object')
    return full_data, dates

# Prepare training and test data with full data utilization and predictions for the last sequence_length days
def prepare_training_test_data(full_data, sequence_length, tagging_method, threshold):
    X, y, sequence_starts = [], [], []
    # Use all data up to len(data) - sequence_length for training/validation
    for i in range(len(full_data) - sequence_length):
        label = [tagging_method_1, tagging_method_2, tagging_method_3, 
                 tagging_method_4][tagging_method - 1](full_data, i, sequence_length, threshold)
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

    # Split into training and test sets (use all data up to the last sequence_length days for prediction)
    if len(X_scaled) > 0:  # Ensure there are valid sequences
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        train_sequence_starts, test_sequence_starts = sequence_starts[:train_size], sequence_starts[train_size:]
    else:
        X_train, X_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])
        train_sequence_starts, test_sequence_starts = np.array([]), np.array([])

    # Prepare the last sequence_length days for prediction (no labels, just predict)
    last_sequence = full_data[features].iloc[len(full_data) - sequence_length:].values
    last_sequence_scaled = np.zeros((1, sequence_length, len(features)))
    for i in range(len(features)):
        scaler = RobustScaler()
        last_sequence_scaled[0, :, i] = scaler.fit_transform(last_sequence[:, i].reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts, last_sequence_scaled

# Build and train LSTM model with progress indicators, and predict for the last sequence_length days
def build_and_train_lstm(X_train, y_train, X_test, y_test, sequence_length, last_sequence_scaled):
    if len(X_train) == 0 or len(y_train) == 0:
        return None, [], [], []
    
    y_train_cat = to_categorical(y_train + 1, num_classes=3)  # Assuming -1, 0, 1 -> 0, 1, 2
    y_test_cat = to_categorical(y_test + 1, num_classes=3)
    
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
        batch_size=4,  # Reverted to original batch size for accuracy
        validation_data=(X_test, y_test_cat),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ],
        class_weight={0: 1.0, 1: 2.0, 2: 1.0}, 
        verbose=1  # Show training progress
    )
    
    # Predict on test set
    test_predictions = model.predict(X_test, verbose=0) if len(X_test) > 0 else np.array([])
    lstm_signals_test = ["sell" if np.argmax(p) == 0 else "hold" if np.argmax(p) == 1 else "buy" for p in test_predictions] if len(X_test) > 0 else []
    
    # Predict for the last sequence_length days
    last_predictions = model.predict(last_sequence_scaled, verbose=0)
    lstm_signals_last = ["sell" if np.argmax(p) == 0 else "hold" if np.argmax(p) == 1 else "buy" for p in last_predictions]
    
    return model, lstm_signals_test, lstm_signals_last, y_test

# Evaluate MACD signals
def evaluate_macd(test_df, test_sequence_starts, y_test):
    macd_flags = []
    for i in range(1, len(test_df)):
        prev_macd, prev_signal = test_df['MACD'].iloc[i - 1], test_df['MACD_Signal'].iloc[i - 1]
        current_macd, current_signal = test_df['MACD'].iloc[i], test_df['MACD_Signal'].iloc[i]
        if prev_macd <= prev_signal and current_macd > current_signal:
            macd_flags.append("buy")
        elif prev_macd >= prev_signal and current_macd < current_signal:
            macd_flags.append("sell")
        else:
            macd_flags.append(np.nan)
    macd_flags = [np.nan] + macd_flags  # Align with test_df
    
    # Align MACD flags with test period indices
    valid_macd_flags = [f for f in macd_flags if not pd.isna(f)]
    valid_indices = [i for i, f in enumerate(macd_flags) if not pd.isna(f)]
    if len(valid_macd_flags) > 0 and len(valid_indices) <= len(y_test) and len(y_test) > 0:
        y_test_aligned = y_test[[i for i in range(len(y_test)) if test_sequence_starts[i] - test_sequence_starts[0] in valid_indices]]
        macd_accuracy = sum(1 for p, a in zip(valid_macd_flags, y_test_aligned) if 
                            (p == "buy" and a == 1) or (p == "sell" and a == -1)) / len(valid_macd_flags)
    else:
        macd_accuracy = 0
    return macd_flags, macd_accuracy

# Plotting function for price, LSTM, MACD, and signals, including predictions for the last sequence_length days
def plot_signals(full_data, dates, sequence_length, test_sequence_starts, lstm_signals_test, lstm_signals_last, macd_flags, config):
    # Define test period DataFrame (including the last day, March 4, 2025)
    test_df = full_data.iloc[:].dropna(subset=['lstm_flag'])  # Use all data up to end_date

    # Plotting: Two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top subplot: Price with LSTM Buy/Sell Flags (including predictions for the last sequence_length days)
    ax1.plot(test_df.index, test_df['Close'], label="Price", color="blue")
    price_range = test_df['Close'].max() - test_df['Close'].min()
    offset = price_range * 0.02  # Adjust for marker positioning

    # Plot LSTM signals from test set, if available
    if len(test_sequence_starts) > 0 and len(lstm_signals_test) > 0:
        test_start_idx = test_sequence_starts[0] + sequence_length
        test_end_idx = test_start_idx + len(lstm_signals_test)
        test_dates = test_df.index[test_start_idx:test_end_idx]
        for i, signal in enumerate(lstm_signals_test):
            if signal == "buy":
                ax1.scatter(test_dates[i], test_df['Close'].iloc[test_start_idx + i] + offset, marker='^', color='green', label='LSTM Buy' if i == 0 else "", s=100)
            elif signal == "sell":
                ax1.scatter(test_dates[i], test_df['Close'].iloc[test_start_idx + i] - offset, marker='v', color='red', label='LSTM Sell' if i == 0 else "", s=100)
    else:
        print("Warning: No test signals available for plotting.")

    # Plot LSTM predictions for the last sequence_length days
    last_start_idx = len(full_data) - sequence_length
    last_dates = full_data.index[last_start_idx:]
    for i, signal in enumerate(lstm_signals_last):
        if signal == "buy":
            ax1.scatter(last_dates[i], full_data['Close'].iloc[last_start_idx + i] + offset, marker='^', color='green', label='LSTM Buy Prediction' if i == 0 else "", s=100)
        elif signal == "sell":
            ax1.scatter(last_dates[i], full_data['Close'].iloc[last_start_idx + i] - offset, marker='v', color='red', label='LSTM Sell Prediction' if i == 0 else "", s=100)

    ax1.set_title(f"Price and LSTM Signals (Sequence Length {config['Sequence Length']}, Method {config['Tagging Method']})")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # Bottom subplot: MACD with MACD Buy/Sell Flags (including the full range)
    ax2.plot(test_df.index, test_df['MACD'], label="MACD", color="blue", linestyle='--')
    ax2.plot(test_df.index, test_df['MACD_Signal'], label="MACD Signal", color="orange")
    macd_range = max(test_df['MACD'].max(), test_df['MACD_Signal'].max()) - min(test_df['MACD'].min(), test_df['MACD_Signal'].min())
    macd_offset = macd_range * 0.02  # Adjust for MACD scale

    # Plot MACD signals for the full range
    for i in range(1, len(test_df)):
        if test_df['macd_flag'].iloc[i] == "buy":
            ax2.scatter(test_df.index[i], test_df['MACD'].iloc[i] + macd_offset, marker='^', color='green', label='MACD Buy' if i == 1 else "", s=100)
        elif test_df['macd_flag'].iloc[i] == "sell":
            ax2.scatter(test_df.index[i], test_df['MACD'].iloc[i] - macd_offset, marker='v', color='red', label='MACD Sell' if i == 1 else "", s=100)

    ax2.set_title(f"MACD and Signals (Sequence Length {config['Sequence Length']}, Method {config['Tagging Method']})")
    ax2.set_ylabel("MACD Value")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main execution
full_data, dates = load_and_preprocess_data(stock_symbol, start_date, end_date)
features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'Lag1']
results = []

# Progress bar for sequence lengths
for sequence_length in tqdm.tqdm(sequence_lengths if not test_single else sequence_lengths, desc="Sequence Lengths"):
    for method in range(1, 5) if not test_single else [test_method]:  # Methods 1–4, or just test_method if testing
        print(f"\nProcessing Sequence Length: {sequence_length}, Tagging Method: {method}")
        
        X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts, last_sequence_scaled = prepare_training_test_data(
            full_data, sequence_length, method, threshold
        )
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping Sequence Length {sequence_length}, Method {method} due to empty data")
            continue
        
        model, lstm_signals_test, lstm_signals_last, y_test = build_and_train_lstm(X_train, y_train, X_test, y_test, sequence_length, last_sequence_scaled)
        
        # Ensure lstm_signals_test and y_test are lists for compatibility
        lstm_signals_test_list = list(lstm_signals_test)  # Convert to list if not already
        y_test_list = list(y_test)  # Convert to list if not already
        
        # Calculate LSTM accuracy for test set, handling empty or mismatched lists
        correct_predictions = 0
        if lstm_signals_test_list and y_test_list and len(lstm_signals_test_list) == len(y_test_list):
            for p, a in zip(lstm_signals_test_list, y_test_list):
                if (p == "buy" and a == 1) or (p == "sell" and a == -1) or (p == "hold" and a == 0):
                    correct_predictions += 1
            lstm_accuracy = correct_predictions / len(y_test_list)
        else:
            lstm_accuracy = 0.0
        
        # Assign test signals to full_data
        for idx, signal in zip(test_sequence_starts, lstm_signals_test):
            full_data.iloc[idx + sequence_length, full_data.columns.get_loc('lstm_flag')] = signal  # Align with test period
        
        # Assign predictions for the last sequence_length days to full_data
        last_start_idx = len(full_data) - sequence_length
        for i, signal in enumerate(lstm_signals_last):
            full_data.iloc[last_start_idx + i, full_data.columns.get_loc('lstm_flag')] = signal + " Prediction"

        test_df = full_data.iloc[test_sequence_starts[0] + sequence_length:len(full_data)] if len(test_sequence_starts) > 0 else full_data
        macd_flags, macd_accuracy = evaluate_macd(test_df, test_sequence_starts, y_test)
        
        # Assign MACD signals to full_data for the test and prediction periods
        macd_idx = test_sequence_starts[0] + 1 if len(test_sequence_starts) > 0 else 1  # Start after the first NaN
        for i, flag in enumerate(macd_flags[1:]):  # Skip the first NaN
            if not pd.isna(flag):
                full_data.iloc[macd_idx + i, full_data.columns.get_loc('macd_flag')] = flag
        
        results.append({
            "Sequence Length": sequence_length,
            "Tagging Method": method,
            "LSTM Accuracy": lstm_accuracy,
            "MACD Accuracy": macd_accuracy
        })

# Display and save results
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df = results_df.pivot(index="Sequence Length", columns="Tagging Method", 
                                  values=["LSTM Accuracy", "MACD Accuracy"])
    results_df.columns = [f"{metric} (Method {method})" for metric, method in results_df.columns]
    results_df = results_df.round(4)
    print("\nAccuracy Comparison Across Tagging Methods and Sequence Lengths:")
    print(results_df.to_string())  # Full table printed without truncation
    results_df.to_csv("accuracy_comparison.csv")
    print("\nResults saved to 'accuracy_comparison.csv'")

    # For test mode, directly use the single configuration (Sequence Length 10, Method 3) to draw the graph
    if test_single:
        sequence_length = 10  # Hardcode sequence length for test mode
        method = 3  # Hardcode method for test mode
        print(f"\nUsing Test Configuration: Sequence Length {sequence_length}, Method {method}")

        # Use the configuration for LSTM to prepare data and plot
        config = {"Sequence Length": sequence_length, "Tagging Method": method}
        X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts, last_sequence_scaled = prepare_training_test_data(
            full_data, sequence_length, method, threshold
        )
        model, lstm_signals_test, lstm_signals_last, y_test = build_and_train_lstm(X_train, y_train, X_test, y_test, sequence_length, last_sequence_scaled)

        # Assign LSTM signals to full_data
        for idx, signal in zip(test_sequence_starts, lstm_signals_test):
            full_data.iloc[idx + sequence_length, full_data.columns.get_loc('lstm_flag')] = signal  # Align with test period
        
        # Assign predictions for the last sequence_length days to full_data
        last_start_idx = len(full_data) - sequence_length
        for i, signal in enumerate(lstm_signals_last):
            full_data.iloc[last_start_idx + i, full_data.columns.get_loc('lstm_flag')] = signal + " Prediction"

        # Use all available data for MACD (including up to March 4, 2025)
        full_test_df = full_data.iloc[:].dropna(subset=['lstm_flag'])  # Use all data up to end_date
        macd_flags, _ = evaluate_macd(full_test_df, test_sequence_starts, y_test)

        # Assign MACD signals to full_data for the full range
        macd_idx = 1  # Start after the first NaN
        for i, flag in enumerate(macd_flags[1:]):  # Skip the first NaN
            if not pd.isna(flag):
                full_data.iloc[macd_idx + i, full_data.columns.get_loc('macd_flag')] = flag

        # Plot signals using the configuration, including predictions for the last sequence_length days
        plot_signals(full_data, dates, sequence_length, test_sequence_starts, lstm_signals_test, lstm_signals_last, macd_flags, config)
    else:  # Full mode
        lstm_cols = [col for col in results_df.columns if col[0] == 'LSTM Accuracy']
        if lstm_cols and not results_df[lstm_cols].empty:
            best_lstm_acc = results_df[lstm_cols].stack().max()
            best_lstm = results_df[lstm_cols].stack().idxmax()
            best_lstm_seq, best_lstm_method = best_lstm[0], best_lstm[1]

            print(f"\nBest LSTM Configuration: Sequence Length {best_lstm_seq}, Method {best_lstm_method}, Accuracy {best_lstm_acc:.4f}")

            # Use the best configuration for LSTM to prepare data and plot
            best_config_lstm = {"Sequence Length": best_lstm_seq, "Tagging Method": best_lstm_method}
            X_train, X_test, y_train, y_test, train_sequence_starts, test_sequence_starts, last_sequence_scaled = prepare_training_test_data(
                full_data, best_lstm_seq, best_lstm_method, threshold
            )
            model, lstm_signals_test, lstm_signals_last, y_test = build_and_train_lstm(X_train, y_train, X_test, y_test, best_lstm_seq, last_sequence_scaled)

            # Assign LSTM signals to full_data
            for idx, signal in zip(test_sequence_starts, lstm_signals_test):
                full_data.iloc[idx + best_lstm_seq, full_data.columns.get_loc('lstm_flag')] = signal  # Align with test period
            
            # Assign predictions for the last sequence_length days to full_data
            last_start_idx = len(full_data) - best_lstm_seq
            for i, signal in enumerate(lstm_signals_last):
                full_data.iloc[last_start_idx + i, full_data.columns.get_loc('lstm_flag')] = signal + " Prediction"

            # Use all available data for MACD (including up to March 4, 2025)
            full_test_df = full_data.iloc[:].dropna(subset=['lstm_flag'])  # Use all data up to end_date
            macd_flags, _ = evaluate_macd(full_test_df, test_sequence_starts, y_test)

            # Assign MACD signals to full_data for the full range
            macd_idx = 1  # Start after the first NaN
            for i, flag in enumerate(macd_flags[1:]):  # Skip the first NaN
                if not pd.isna(flag):
                    full_data.iloc[macd_idx + i, full_data.columns.get_loc('macd_flag')] = flag

            # Plot signals using the best LSTM configuration, including predictions for the last sequence_length days
            plot_signals(full_data, dates, best_lstm_seq, test_sequence_starts, lstm_signals_test, lstm_signals_last, macd_flags, best_config_lstm)
        else:
            print("No valid LSTM accuracy data available for finding the best configuration.")
else:
    print("No results available to process.")