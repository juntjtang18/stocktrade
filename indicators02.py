import tkinter as tk  # Corrected import
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mylib.download_data import download_data
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta

def generate_graph(symbol):
    try:
        data = download_data(symbol, source="polygon")
        if data is None or data.empty:
            return None

        data = data.sort_index(ascending=True).drop_duplicates()
        data = data[data['Volume'] > 0]
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        data = data.loc[start_date:end_date]
        dates = data.index
        start_date_exact = dates[0]
        buffer_end_date = datetime.now() + timedelta(days=5)

        macd = data.ta.macd(fast=12, slow=26, signal=9)
        data = pd.concat([data, macd], axis=1)
        bb = data.ta.bbands(length=20, std=2)
        data = pd.concat([data, bb], axis=1)
        stoch = data.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, smooth_k=3)
        data = pd.concat([data, stoch], axis=1)

        macd_line = 'MACD_12_26_9'
        signal_line = 'MACDs_12_26_9'
        histogram = 'MACDh_12_26_9'
        bb_lower = 'BBL_20_2.0'
        bb_middle = 'BBM_20_2.0'
        bb_upper = 'BBU_20_2.0'
        stoch_k = 'STOCHk_14_3_3'
        stoch_d = 'STOCHd_14_3_3'

        data['MACD_Signal'] = 0
        data['MACD_Diff'] = data[macd_line] - data[signal_line]
        data['MACD_Diff_Prev'] = data['MACD_Diff'].shift(1)
        data.loc[(data['MACD_Diff'] > 0) & (data['MACD_Diff_Prev'] < 0), 'MACD_Signal'] = 1
        data.loc[(data['MACD_Diff'] < 0) & (data['MACD_Diff_Prev'] > 0), 'MACD_Signal'] = -1

        data['BB_Signal'] = 0
        data.loc[(data['Close'] > data[bb_lower]) & (data['Close'].shift(1) <= data[bb_lower].shift(1)), 'BB_Signal'] = 1
        data.loc[(data['Close'] < data[bb_upper]) & (data['Close'].shift(1) >= data[bb_upper].shift(1)), 'BB_Signal'] = -1

        data['Stoch_Signal'] = 0
        data['Stoch_Diff'] = data[stoch_k] - data[stoch_d]
        data['Stoch_Diff_Prev'] = data['Stoch_Diff'].shift(1)
        data.loc[(data['Stoch_Diff'] > 0) & (data['Stoch_Diff_Prev'] < 0) & (data[stoch_k] < 20), 'Stoch_Signal'] = 1
        data.loc[(data['Stoch_Diff'] < 0) & (data['Stoch_Diff_Prev'] > 0) & (data[stoch_k] > 80), 'Stoch_Signal'] = -1

        fig = Figure(figsize=(12, 16))
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(dates, data['Close'], label='Close Price', color='blue')
        macd_buy = data[data['MACD_Signal'] == 1]['Close']
        macd_sell = data[data['MACD_Signal'] == -1]['Close']
        ax1.scatter(macd_buy.index, macd_buy, color='green', marker='^', s=100, label='MACD Buy')
        ax1.scatter(macd_sell.index, macd_sell, color='red', marker='v', s=100, label='MACD Sell')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{symbol} Stock Price with MACD Signals')
        ax1.legend(loc='upper left', fontsize='small')
        ax1.set_xticks([])

        ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
        ax2.plot(dates, data[macd_line], label='MACD', color='blue')
        ax2.plot(dates, data[signal_line], label='Signal Line', color='orange')
        ax2.bar(dates, data[histogram], label='Histogram', color='grey', alpha=0.3)
        ax2.scatter(macd_buy.index, data.loc[macd_buy.index, macd_line], color='green', marker='^', s=100, label='MACD Buy')
        ax2.scatter(macd_sell.index, data.loc[macd_sell.index, macd_line], color='red', marker='v', s=100, label='MACD Sell')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_ylabel('MACD Value')
        ax2.set_title('MACD Indicator')
        ax2.legend(loc='upper left', fontsize='small')
        ax2.set_xticks([])

        ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.plot(dates, data['Close'], label='Close Price', color='blue')
        ax3.plot(dates, data[bb_lower], label='Lower Band', color='red', linestyle='--')
        ax3.plot(dates, data[bb_middle], label='Middle Band', color='black', linestyle='--')
        ax3.plot(dates, data[bb_upper], label='Upper Band', color='green', linestyle='--')
        bb_buy = data[data['BB_Signal'] == 1]['Close']
        bb_sell = data[data['BB_Signal'] == -1]['Close']
        ax3.scatter(bb_buy.index, bb_buy, color='green', marker='^', s=50, label='Buy')
        ax3.scatter(bb_sell.index, bb_sell, color='red', marker='v', s=50, label='Sell')
        ax3.set_ylabel('Price ($)')
        ax3.set_title('Bollinger Bands')
        ax3.legend(loc='upper left', fontsize='small')
        ax3.set_xticks([])

        ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
        ax4.plot(dates, data[stoch_k], label='%K', color='blue')
        ax4.plot(dates, data[stoch_d], label='%D', color='orange')
        stoch_buy = data[data['Stoch_Signal'] == 1][stoch_k]
        stoch_sell = data[data['Stoch_Signal'] == -1][stoch_k]
        ax4.scatter(stoch_buy.index, stoch_buy, color='green', marker='^', s=50, label='Buy')
        ax4.scatter(stoch_sell.index, stoch_sell, color='red', marker='v', s=50, label='Sell')
        ax4.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Value')
        ax4.set_ylim(0, 100)
        ax4.set_title('Stochastic Oscillator')
        ax4.legend(loc='upper left', fontsize='small')
        ax4.set_xticks([])

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(start_date_exact, buffer_end_date)
            ax.grid(True)

        return fig
    except Exception as e:
        print(f"Error in generate_graph: {e}")
        return None

def main():
    try:
        root = tk.Tk()
        root.title("Stock Graph Generator")

        symbol_label = tk.Label(root, text="Enter Stock Symbol: ")
        symbol_label.grid(row=0, column=0, padx=5, pady=5)
        symbol_entry = tk.Entry(root)
        symbol_entry.grid(row=0, column=1, padx=5, pady=5)

        generate_button = tk.Button(root, text="Generate Graph", command=lambda: plot_graph(symbol_entry.get()))
        generate_button.grid(row=1, column=0, columnspan=2, pady=5)

        graph_frame = tk.Frame(root)
        graph_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(1, weight=1)

        error_label = tk.Label(root, text="", fg="red")
        error_label.grid(row=3, column=0, columnspan=2, pady=5)

        canvas = None

        def plot_graph(symbol):
            nonlocal canvas
            if canvas:
                canvas.get_tk_widget().destroy()
            fig = generate_graph(symbol.strip().upper())
            if fig:
                canvas = FigureCanvasTkAgg(fig, master=graph_frame)
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                canvas.draw()
                error_label.config(text="")
            else:
                error_label.config(text="Error: No data or invalid symbol.")

        root.mainloop()
    except Exception as e:
        print(f"Error initializing GUI: {e}")

if __name__ == "__main__":
    main()