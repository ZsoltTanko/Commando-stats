import os
import time
import calendar
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sn
from bs4 import BeautifulSoup


# Stores time and value of commando signals
class CommandoSymbol:
    def __init__(self, symbol_name):
        self.symbol_name = symbol_name
        self.times = []
        self.values = []
        self.num_entries = 0

    def add_call(self, time, value):
        self.times.append(time)
        self.values.append(value)
        self.num_entries += 1


# Draw a labeled heatmap with the option to save to file
def draw_heatmap(row_labels, col_labels, data, prob_pallet, title=None, save=False, display = True):
    # Create dataframe from row/column labels
    df = {}
    for i in range(0, len(col_labels)):
        df[col_labels[i]] = pd.Series(data[:, i], index=[str(x) for x in row_labels])
    df = pd.DataFrame(df)

    # Configure, draw, save heatmap
    sn.set(font_scale=0.8)
    palette = sn.diverging_palette(h_neg=10, h_pos=230, s=99, l=55, sep=3, as_cmap=True)
    ax = sn.heatmap(df, cmap=palette, annot=True, center=0.0, cbar=False)
    ax.figure.set_size_inches(5.5, 2.5)
    plt.title(title)
    plt.subplots_adjust(top=0.8)
    plt.subplots_adjust(left=0.25)
    plt.subplots_adjust(right=0.975)
    if save:
        plt.savefig(os.getcwd() + "/data/" + title + ".png")
    if display:
        plt.show()


# Load commando data
f = open(os.getcwd() + "/data/commando.txt", "r")
t = f.read()
f.close()
t = t.split('\n')

# Parse commando signals
signals = {}
signals_by_time = defaultdict(list)
for line in t:
    if line.startswith('2021'):
        # Skip signals where commando was updating
        if 'pending' in line:
            continue

        # Parse symbol, add to symbol list if needed
        split = line.split()
        symbol = split[2]
        if symbol not in signals.keys():
            signals[symbol] = CommandoSymbol(symbol)

        # Parse date
        timestamp = split[0] + ' ' + split[1]
        timestamp_parsed = calendar.timegm(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
        score = float(split[5])

        # Parse price delta since last commando update
        delta = score - float(re.search("T-4=[-]?\d+.\d+", line).group()[4:])

        signals_by_time[timestamp_parsed].append((symbol, score, delta))
        signals[symbol].add_call(timestamp_parsed, score, delta)

# Save btc signals separately
signals_btc = signals.pop('BTCUSDT')
signals_btc_by_time = {signals_btc.times[i]: signals_btc.values[i] for i in range(signals_btc.num_entries)}

# Compute commando signal averages
commando_avg_by_time = {}
min_avg = 4; max_avg = -2
for time, entries in signals_by_time.items():
    avg = np.mean([score for symbol, score, delta in entries])
    commando_avg_by_time[time] = avg
    commando_delta_avg_by_time[time] = np.mean([delta for symbol, score, delta in entries])

    min_avg = min(min_avg, avg)
    max_avg = max(max_avg, avg)

# Rescale averages to [-1,3]
for time in commando_avg_by_time.keys():
    commando_avg_by_time[time] -= min_avg
    commando_avg_by_time[time] /= max_avg
    commando_avg_by_time[time] *= 4
    commando_avg_by_time[time] -= 1

# Load tradingview ticker data for binance symbols
symbols = {'AAVEUSDT', 'ADAUSDT', 'AVAXUSDT', 'BALUSDT', 'BNBUSDT', 'DCRUSDT', 'DOGEUSDT',
           'DOTUSDT', 'EOSUSDT', 'ETHBTC', 'ETHUSDT', 'LINKUSDT', 'RUNEUSDT', 'RLCUSDT',
           'SOLUSDT', 'SUSHIUSDT', 'SXPUSDT', 'XMRUSDT', 'XRPUSDT', 'UNIUSDT', 'XHVUSDT',
           'YFIUSDT', 'BTCUSDT', 'LTCUSDT'}
dfs = {s:pd.read_csv(os.getcwd() + "/data//BINANCE_" + s + ", 15.csv") for s in symbols}


# Set parameters for backtest: SL and TP values, entry signal threshold, stats intervals (in h)
stop_loss = -0.05
tp = 0.2
score_entry_threshold = 2.5
btc_score_threshold = 1.5
commando_avg_threshold = 0.5
commando_delta_avg_threshold = 0.1
stat_intervals = [4, 8, 12, 24, 48, 24*3, 24*4, 24*5, 24*6, 24*7]
plotting_scatterplots = True


# Data to collect
num_entry_signals = 0
max_perc_up_after_entry = defaultdict(list)
max_perc_drawdown_after_entry = defaultdict(list)
max_perc_drawdown_before_high = defaultdict(list)
max_perc_drawdown_before_tp = defaultdict(list)
interval_end_perc_change = defaultdict(list)
not_tp_or_sl_perc_change = defaultdict(list)
avg_score_delta_on_entry = defaultdict(list)
avg_score_on_entry = defaultdict(list)
btc_score_on_entry = defaultdict(list)
score_on_entry = defaultdict(list)
num_times_stopped_out = {interval: 0 for interval in stat_intervals}
num_times_tp = {interval: 0 for interval in stat_intervals}
num_times_straight_down = {interval: 0 for interval in stat_intervals}

# Iterate over commando symbols
for symbol, commando in signals.items():
    df = dfs[symbol]

    for i in range(1, commando.num_entries):
        signal_time = commando.times[i]
        score = commando.values[i]
        prev_score = commando.values[i - 1]
        btc_score = signals_btc_by_time[signal_time]
        avg_score = commando_avg_by_time[signal_time]
        avg_score_delta = commando_delta_avg_by_time[signal_time]

        # Entry signal criteria
        if score >= score_entry_threshold and prev_score < score_entry_threshold and btc_score >= btc_score_threshold and avg_score >= commando_avg_threshold and avg_score_delta >= commando_delta_avg_threshold:

            # Skip commando entry signals for which ticker data is missing
            ind = df.loc[df['time'] == signal_time].index
            if len(ind) == 0:
                continue
            signal_ticker_index = ind[0]
            num_entry_signals += 1

            entry_close_price = df['close'].iloc[signal_ticker_index]

            # Find price changes over stats collection intervals after the entry signal
            for interval in stat_intervals:
                price_interval = df['close'].iloc[signal_ticker_index:signal_ticker_index + int(interval*60/15)]

                price_interval_end_perc_change = price_interval.iloc[-1]/entry_close_price - 1
                interval_end_perc_change[interval].append(price_interval_end_perc_change)

                # Compute TP/SL stats
                tp_price = entry_close_price*(1 + tp)
                if max(price_interval) >= tp_price:
                    tp_index = next(i for i, v in enumerate(price_interval) if v >= tp_price)

                    max_drawdown_before_tp = min(price_interval[:tp_index])
                    max_drawdown_before_tp = (max_drawdown_before_tp - entry_close_price)/entry_close_price
                    max_perc_drawdown_before_tp[interval].append(max_drawdown_before_tp)

                    if max_drawdown_before_tp <= stop_loss:
                        num_times_stopped_out[interval] += 1
                    else:
                        num_times_tp[interval] += 1
                else:
                    max_drawdown = min(price_interval)
                    max_drawdown = (max_drawdown - entry_close_price)/entry_close_price
                    if max_drawdown <= stop_loss:
                        num_times_stopped_out[interval] += 1
                    else:
                        not_tp_or_sl_perc_change[interval].append(price_interval_end_perc_change)
                        avg_score_delta_on_entry[interval].append(commando_delta_avg_by_time[signal_time])
                        avg_score_on_entry[interval].append(commando_avg_by_time[signal_time])
                        btc_score_on_entry[interval].append(signals_btc_by_time[signal_time])
                        score_on_entry[interval].append(commando.values[i])

                # Compute max gain/loss stats
                max_close = max(price_interval)
                max_perc_up = (max_close - entry_close_price)/entry_close_price
                max_perc_up_after_entry[interval].append(max_perc_up)

                max_close_index = list(price_interval).index(max_close)
                if max_close_index == 0: # Price went only down after entry signal
                    max_perc_drawdown_before_high[interval].append(0)
                    num_times_straight_down[interval] += 1
                else:
                    max_drawdown_before_high = min(price_interval[:max_close_index])
                    max_drawdown_before_high = (max_drawdown_before_high - entry_close_price)/entry_close_price
                    max_perc_drawdown_before_high[interval].append(max_drawdown_before_high)

                min_close = min(price_interval)
                max_perc_drawdown = (min_close - entry_close_price)/entry_close_price
                max_perc_drawdown_after_entry[interval].append(max_perc_drawdown)


# Print stats for every interval
print("Num entry signals: " + str(num_entry_signals))

for interval in stat_intervals:
    print("%d hours   (tp %.2f, sl %.2f)" % (interval, tp*100, stop_loss*100))
    print("Best gain percent: %.2f ± %.2f" % (np.mean(max_perc_up_after_entry[interval])*100, np.std(max_perc_up_after_entry[interval])*100))
    print("Worst drawdown percent: %.2f ± %.2f" % (np.mean(max_perc_drawdown_after_entry[interval])*100, np.std(max_perc_drawdown_after_entry[interval])*100))
    print("Avg drawdown before high: %.2f ± %.2f" % (np.mean(max_perc_drawdown_before_high[interval])*100, np.std(max_perc_drawdown_before_high[interval])*100))
    print("Avg drawdown before tp: %.2f ± %.2f" % (np.mean(max_perc_drawdown_before_tp[interval])*100, np.std(max_perc_drawdown_before_tp[interval])*100))
    print("Straight down percentage: %.2f" % ((num_times_straight_down[interval]/num_entry_signals)*100))
    print("Interval end change percent: %.2f ± %.2f" % (np.mean(interval_end_perc_change[interval])*100, np.std(interval_end_perc_change[interval])*100))

    percent_stopped = (num_times_stopped_out[interval]/num_entry_signals)*100
    percent_tp = (num_times_tp[interval]/num_entry_signals)*100
    print("SL percentage: %.2f" % (percent_stopped))
    print("TP percentage: %.2f" % (percent_tp))
    print("Not TP or SL end of interval change perc: %.2f" % (np.mean(not_tp_or_sl_perc_change[interval])*100))
    print("Expected return: %.2f" % (percent_stopped*stop_loss + percent_tp*tp + (100-(percent_stopped+percent_tp))*np.mean(not_tp_or_sl_perc_change[interval])))
    print('\n')


if plotting_scatterplots:
    # Set up summary stats data
    row_labels = ["max gain", "max drawdown", "drawdown before peak", "hodl return"]
    col_labels = []
    for x in stat_intervals:
        if x < 24:
            col_labels.append(str(x)+"h")
        else:
            col_labels.append(str(int(x/24))+"d")

    data = np.zeros((len(row_labels), len(col_labels)))
    for i in range(len(stat_intervals)):
        interval = stat_intervals[i]
        data[0, i] = np.mean(max_perc_up_after_entry[interval])*100
        data[1, i] = np.mean(max_perc_drawdown_after_entry[interval])*100
        data[2, i] = np.mean(max_perc_drawdown_before_high[interval])*100
        data[3, i] = np.mean(interval_end_perc_change[interval])*100

    df = {}
    for i in range(len(col_labels)):
        df[col_labels[i]] = pd.Series(data[:, i], index=[str(x) for x in row_labels])
    df = pd.DataFrame(df)

    # Plot summary stats and scatter plots
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    spec = gridspec.GridSpec(ncols=2, nrows=2)
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    ax3 = fig.add_subplot(spec[2])
    ax4 = fig.add_subplot(spec[3])

    sn.set(font_scale=0.8)
    palette = sn.diverging_palette(h_neg=10, h_pos=230, s=99, l=55, sep=3, as_cmap=True)
    sn.heatmap(df, ax=ax1, cmap=palette, annot=True, center=0.0, cbar=False)
    ax1.set_title("commando cross %.2f entry (average of %d entries)" %(entry_on_crossover_of, num_entry_signals))

    c = np.array(score_on_entry[0]) + 1
    ax2.scatter(avg_score_delta_on_entry[0], not_tp_or_sl_perc_change[0], c=c, s=10, alpha=0.5, cmap='cool')
    ax2.grid(True)
    ax2.set_title("PNL vs avg score delta")
    ax2.set_facecolor('black')

    ax3.scatter(avg_score_on_entry[0], not_tp_or_sl_perc_change[0], c=c, s=10, alpha=0.5, cmap='cool')
    ax3.grid(True)
    ax3.set_title("PNL vs avg score")
    ax3.set_facecolor('black')

    ax4.scatter(btc_score_on_entry[0], not_tp_or_sl_perc_change[0], c=c, s=10, alpha=0.5, cmap='cool')
    ax4.grid(True)
    ax4.set_title("PNL vs btc score")
    ax4.set_facecolor('black')

    plt.show()

else:
    # Set up and draw heatmap
    title = "commando cross %.2f entry (average of %d entries)\nbtc score between 0.5 and 2.0" %(entry_on_crossover_of, num_entry_signals)
    draw_heatmap(row_labels, col_labels, data, prob_pallet=False, title=title, save=False, display=True)

    row_labels = ["TP fired", "SL fired", "down only"]
    col_labels = []
    for x in stat_intervals:
        if x < 24:
            col_labels.append(str(x)+"h")
        else:
            col_labels.append(str(int(x/24))+"d")
    data = np.zeros((len(row_labels), len(col_labels)))

    for i in range(len(stat_intervals)):
        interval = stat_intervals[i]

        percent_stopped = (num_times_stopped_out[interval]/num_entry_signals)*100
        percent_tp = (num_times_tp[interval]/num_entry_signals)*100

        data[0, i] = percent_stopped
        data[1, i] = percent_tp
        data[2, i] = (num_times_straight_down[interval]/num_entry_signals)*100

    draw_heatmap(row_labels, col_labels, data, True)

