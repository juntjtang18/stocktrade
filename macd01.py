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
import os
from mylib.download_data import download_data
from datetime import datetime, timedelta

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configurable parameters
stock_symbol = "AAPL"
data_dir = os.path.join(os.path.dirname(__file__), "data")
local_file = os.path.join(data_dir, f"{stock_symbol}_data.csv")
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
buffer_days = 5  # Add 5 days buffer after today
sequence_length = 15
threshold = 0.01
initial_amount = 10000

# Load data
data = download_data(stock_symbol, source="polygon")

# Ensure chronological order and trim to exact date range
data = data.sort_index(ascending=True).drop_duplicates()
data = data[data['Volume'] > 0]
data = data.loc[start_date:end_date]

# Store dates and define exact limits
dates = data.index
start_date_exact = dates[0]  # First date in the data
end_date_exact = dates[-1]  # Last date in the data
# Calculate buffer end date (5 days after today)
buffer_end_date = datetime.now() + timedelta(days=buffer_days)

# Calculate MACD
macd = data.ta.macd(fast=12, slow=26, signal=9)
data = pd.concat([data, macd], axis=1)

# Define column names
macd_line = 'MACD_12_26_9'
signal_line = 'MACDs_12_26_9'
histogram = 'MACDh_12_26_9'

# Generate trading signals
data['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
data['MACD_Diff'] = data[macd_line] - data[signal_line]
data['MACD_Diff_Prev'] = data['MACD_Diff'].shift(1)

# Detect crossovers
data.loc[(data['MACD_Diff'] > 0) & (data['MACD_Diff_Prev'] < 0), 'Signal'] = 1  # Buy
data.loc[(data['MACD_Diff'] < 0) & (data['MACD_Diff_Prev'] > 0), 'Signal'] = -1  # Sell

# Create visualization
plt.figure(figsize=(15, 10))

# Price plot (top)
ax1 = plt.subplot(2, 1, 1)
plt.plot(dates, data['Close'], label='Close Price', color='blue')
buy_signals = data[data['Signal'] == 1]['Close']
sell_signals = data[data['Signal'] == -1]['Close']
plt.scatter(buy_signals.index, buy_signals, color='green', label='Buy', marker='^', s=100)
plt.scatter(sell_signals.index, sell_signals, color='red', label='Sell', marker='v', s=100)
plt.title(f'{stock_symbol} Stock Price with Trade Signals')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
# Set x-axis limits with buffer
ax1.set_xlim(start_date_exact, buffer_end_date)

# MACD plot (bottom)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)  # Share x-axis with ax1
plt.plot(dates, data[macd_line], label='MACD', color='blue')
plt.plot(dates, data[signal_line], label='Signal Line', color='orange')
plt.bar(dates, data[histogram], label='Histogram', color='grey', alpha=0.3)
plt.scatter(buy_signals.index, data.loc[buy_signals.index, macd_line], 
           color='green', label='Buy', marker='^', s=100)
plt.scatter(sell_signals.index, data.loc[sell_signals.index, macd_line], 
           color='red', label='Sell', marker='v', s=100)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.title('MACD Indicator')
plt.ylabel('MACD Value')
plt.legend()
plt.grid(True)
# Set x-axis limits with buffer
ax2.set_xlim(start_date_exact, buffer_end_date)

plt.tight_layout()
plt.show()