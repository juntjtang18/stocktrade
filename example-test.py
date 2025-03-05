import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# Example DataFrame
data = {
    'Close': [100, 102, 101, 99, 98, 97, 99, 101, 100, 98, 97, 96, 95, 97, 99, 101, 102, 100]
}
df = pd.DataFrame(data)

# Identify local extrema
def identify_local_extrema(df, order=5):
    df['Min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=order)[0]]['Close']
    df['Max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]]['Close']
    df['Trade_Label'] = 0
    df.loc[df['Min'].notna(), 'Trade_Label'] = 1  # Buy signal
    df.loc[df['Max'].notna(), 'Trade_Label'] = -1  # Sell signal
    return df

df = identify_local_extrema(df, order=2)
print(df)