# Implemented 4 technical indicators:
# MACD, RSI, Rolling Band, and Stochastic

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
buffer_days = 5
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
start_date_exact = dates[0]
end_date_exact = dates[-1]
buffer_end_date = datetime.now() + timedelta(days=buffer_days)

# Calculate MACD
macd = data.ta.macd(fast=12, slow=26, signal=9)
data = pd.concat([data, macd], axis=1)

# Calculate Bollinger Bands (20-day period, 2 standard deviations)
bb = data.ta.bbands(length=20, std=2)
data = pd.concat([data, bb], axis=1)

# Calculate Stochastic Oscillator (14, 3, 3)
stoch = data.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, smooth_k=3)
data = pd.concat([data, stoch], axis=1)

# Define column names
macd_line = 'MACD_12_26_9'
signal_line = 'MACDs_12_26_9'
histogram = 'MACDh_12_26_9'
bb_lower = 'BBL_20_2.0'
bb_middle = 'BBM_20_2.0'
bb_upper = 'BBU_20_2.0'
stoch_k = 'STOCHk_14_3_3'
stoch_d = 'STOCHd_14_3_3'

# Generate MACD trading signals
data['MACD_Signal'] = 0
data['MACD_Diff'] = data[macd_line] - data[signal_line]
data['MACD_Diff_Prev'] = data['MACD_Diff'].shift(1)
data.loc[(data['MACD_Diff'] > 0) & (data['MACD_Diff_Prev'] < 0), 'MACD_Signal'] = 1  # Buy
data.loc[(data['MACD_Diff'] < 0) & (data['MACD_Diff_Prev'] > 0), 'MACD_Signal'] = -1  # Sell

# Generate Bollinger Bands trading signals
data['BB_Signal'] = 0
data.loc[(data['Close'] > data[bb_lower]) & (data['Close'].shift(1) <= data[bb_lower].shift(1)), 'BB_Signal'] = 1  # Buy
data.loc[(data['Close'] < data[bb_upper]) & (data['Close'].shift(1) >= data[bb_upper].shift(1)), 'BB_Signal'] = -1  # Sell

# Generate Stochastic trading signals
data['Stoch_Signal'] = 0
data['Stoch_Diff'] = data[stoch_k] - data[stoch_d]
data['Stoch_Diff_Prev'] = data['Stoch_Diff'].shift(1)
data.loc[(data['Stoch_Diff'] > 0) & (data['Stoch_Diff_Prev'] < 0) & (data[stoch_k] < 20), 'Stoch_Signal'] = 1  # Buy
data.loc[(data['Stoch_Diff'] < 0) & (data['Stoch_Diff_Prev'] > 0) & (data[stoch_k] > 80), 'Stoch_Signal'] = -1  # Sell

# Create visualization
plt.figure(figsize=(15, 20))  # Increased height for 4 panels

# Price plot (top)
ax1 = plt.subplot(4, 1, 1)
plt.plot(dates, data['Close'], label='Close Price', color='blue')
macd_buy = data[data['MACD_Signal'] == 1]['Close']
macd_sell = data[data['MACD_Signal'] == -1]['Close']
plt.scatter(macd_buy.index, macd_buy, color='green', label='MACD Buy', marker='^', s=100)
plt.scatter(macd_sell.index, macd_sell, color='red', label='MACD Sell', marker='v', s=100)
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
ax1.set_xlim(start_date_exact, buffer_end_date)
ax1.set_title(f'{stock_symbol} Stock Price with MACD Signals')  # Keep title, remove xlabel

# MACD plot (second)
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
plt.plot(dates, data[macd_line], label='MACD', color='blue')
plt.plot(dates, data[signal_line], label='Signal Line', color='orange')
plt.bar(dates, data[histogram], label='Histogram', color='grey', alpha=0.3)
plt.scatter(macd_buy.index, data.loc[macd_buy.index, macd_line], 
           color='green', label='MACD Buy', marker='^', s=100)
plt.scatter(macd_sell.index, data.loc[macd_sell.index, macd_line], 
           color='red', label='MACD Sell', marker='v', s=100)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.ylabel('MACD Value')
plt.legend()
plt.grid(True)
ax2.set_xlim(start_date_exact, buffer_end_date)
ax2.set_title('MACD Indicator')  # Keep title, remove xlabel

# Bollinger Bands plot (third)
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(dates, data['Close'], label='Close Price', color='blue')
plt.plot(dates, data[bb_lower], label='Lower Band', color='red', linestyle='--')
plt.plot(dates, data[bb_middle], label='Middle Band', color='black', linestyle='--')
plt.plot(dates, data[bb_upper], label='Upper Band', color='green', linestyle='--')
bb_buy = data[data['BB_Signal'] == 1]['Close']
bb_sell = data[data['BB_Signal'] == -1]['Close']
plt.scatter(bb_buy.index, bb_buy, color='green', label='Buy', marker='^', s=50)
plt.scatter(bb_sell.index, bb_sell, color='red', label='Sell', marker='v', s=50)
plt.ylabel('Price ($)')
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
ax3.set_xlim(start_date_exact, buffer_end_date)
ax3.set_title('Bollinger Bands')  # Keep title, remove xlabel

# Stochastic plot (bottom)
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
plt.plot(dates, data[stoch_k], label='%K', color='blue')
plt.plot(dates, data[stoch_d], label='%D', color='orange')
stoch_buy = data[data['Stoch_Signal'] == 1][stoch_k]
stoch_sell = data[data['Stoch_Signal'] == -1][stoch_k]
plt.scatter(stoch_buy.index, stoch_buy, color='green', label='Buy', marker='^', s=50)
plt.scatter(stoch_sell.index, stoch_sell, color='red', label='Sell', marker='v', s=50)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Overbought (80)')
plt.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Oversold (20)')
plt.ylabel('Value')
plt.ylim(0, 100)
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
ax4.set_xlim(start_date_exact, buffer_end_date)
ax4.set_title('Stochastic Oscillator')  # Keep title, remove xlabel

plt.tight_layout()
plt.show()