import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Fetch stock data
stock_symbol = "AAPL"  # Example: Apple stock - change this as needed
data = yf.download(stock_symbol, start="2020-01-01", end="2025-03-02")  # Up to current date
prices = data['Close'].values.reshape(-1, 1)  # Use closing prices

# Step 2: Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences (e.g., 60 days of data to predict the next day)
sequence_length = 60  # Number of days to look back
X, y = [], []
for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i-sequence_length:i])  # Past 60 days
    y.append(1 if scaled_prices[i] > scaled_prices[i-1] else 0)  # 1 = buy, 0 = sell
X, y = np.array(X), np.array(y)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))  # Prevent overfitting
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Output: 0 or 1 (sell/buy)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Step 6: Make a prediction
latest_data = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
prediction = model.predict(latest_data)[0][0]
trade_flag = "buy" if prediction > 0.5 else "sell"
print(f"Predicted trade flag for next day: {trade_flag} (Probability: {prediction:.4f})")

# Step 7: Visualize training progress (optional)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 8: Backtest (simple example)
test_predictions = model.predict(X_test)
test_signals = ["buy" if p > 0.5 else "sell" for p in test_predictions]
correct_signals = sum(1 for pred, actual in zip(test_signals, y_test) if (pred == "buy" and actual == 1) or (pred == "sell" and actual == 0))
print(f"Correct signals in test set: {correct_signals}/{len(y_test)} ({correct_signals/len(y_test)*100:.2f}%)")

