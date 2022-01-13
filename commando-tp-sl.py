import os
import time
import calendar
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
def draw_heatmap(row_labels, col_labels, data, prob_pallet, title=None, filename=None, save=False, display = True, annot=None, xlabel=None, ylabel=None):
    # Create dataframe from row/column labels
    df = {}
    for i in range(0, len(col_labels)):
        df[col_labels[i]] = pd.Series(data[:, i], index=[str(x) for x in row_labels])
    df = pd.DataFrame(df)

    # Configure, draw, save heatmap
    sn.set(font_scale=0.7)
    palette = sn.diverging_palette(h_neg=10, h_pos=230, s=99, l=55, sep=3, as_cmap=True)
    ax = sn.heatmap(df, annot=annot, cmap=palette, center=0.00, cbar=False, fmt='')
    ax.figure.set_size_inches(6, 5.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(os.getcwd() + "/data/" + filename + ".png")
    if display:
        plt.show()


# Load commando data
soup = BeautifulSoup(open(os.getcwd() + "/data/commando.html"), features="lxml")
t = soup.get_text()
t = t.split('\n')

# Parse commando signals
signals = {}
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

        # Parse date and signal value
        timestamp = split[0] + ' ' + split[1]
        timestamp_parsed = calendar.timegm(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
        signals[symbol].add_call(timestamp_parsed, float(split[5]))


# Load tradingview ticker data for binance symbols
symbols = {'AAVEUSDT', 'ADAUSDT', 'AVAXUSDT', 'BNBUSDT', 'BTCUSDT', 'DCRUSDT', 'DOGEUSDT',
           'DOTUSDT', 'EOSUSDT', 'ETHBTC', 'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'RUNEUSDT',
           'SOLUSDT', 'SUSHIUSDT', 'UNIUSDT', 'XHVUSDT', 'XRPUSDT', 'YFIUSDT'}
dfs = {s:pd.read_csv(os.getcwd() + "/data//BINANCE_" + s + ", 15.csv") for s in symbols}


# Set parameters for backtest: SL and TP values, entry signal threshold, stats intervals (in h)
stop_losses = [-1.0,-0.2,-0.15,-0.1,-0.07,-0.05,-0.04,-0.03,-0.02]
take_profits = [0.04,0.05,0.06,0.07,0.1,0.15,0.2,0.3,1.0]
entry_on_crossover_of = 3.0
stat_intervals = [4, 8, 12, 24, 48, 24*3, 24*4]
heatmap_stats_interval = 3

tp_fired_perc = np.zeros((len(stop_losses), len(take_profits)))
sl_fired_perc = np.zeros((len(stop_losses), len(take_profits)))
expected_returns = np.zeros((len(stop_losses), len(take_profits)))

# Iterate over TP/SL configurations
for k in range(len(stop_losses)):
    for j in range(0,len(take_profits)):
        stop_loss = stop_losses[k]
        tp = take_profits[j]

        # Data to collect for each TP/SL combination
        num_entry_signals = 0
        max_perc_up_after_entry = defaultdict(list)
        max_perc_drawdown_after_entry = defaultdict(list)
        max_perc_drawdown_before_high = defaultdict(list)
        max_perc_drawdown_before_tp = defaultdict(list)
        interval_end_perc_change = defaultdict(list)
        not_tp_or_sl_perc_change = defaultdict(list)
        num_times_stopped_out = {interval: 0 for interval in stat_intervals}
        num_times_tp = {interval: 0 for interval in stat_intervals}
        num_times_straight_down = {interval: 0 for interval in stat_intervals}

        # Iterate over commando symbols
        for symbol, commando in signals.items():
            df = dfs[symbol]

            for i in range(1, commando.num_entries):

                # Entry signal criteria
                if commando.values[i] >= entry_on_crossover_of and commando.values[i-1] < entry_on_crossover_of:

                    # Skip commando entry signals for which ticker data is missing
                    signal_time = commando.times[i]
                    ind = df.loc[df['time'] == signal_time].index
                    if len(ind) == 0:
                        continue
                    signal_ticker_index = ind[0]
                    num_entry_signals += 1

                    entry_close_price = df['close'].iloc[signal_ticker_index]

                    # Find price changes over stats collectio intervals after the entry signal
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

                        # Compute max gain/loss stats
                        max_close = max(price_interval)
                        max_perc_up = (max_close - entry_close_price)/entry_close_price
                        max_perc_up_after_entry[interval].append(max_perc_up)

                        max_close_index = list(price_interval).index(max_close)
                        if max_close_index == 0:
                            max_perc_drawdown_before_high[interval].append(0)
                            num_times_straight_down[interval] += 1
                        else:
                            max_drawdown_before_high = min(price_interval[:max_close_index])
                            max_drawdown_before_high = (max_drawdown_before_high - entry_close_price)/entry_close_price
                            max_perc_drawdown_before_high[interval].append(max_drawdown_before_high)

                        min_close = min(price_interval)
                        max_perc_drawdown = (min_close - entry_close_price)/entry_close_price
                        max_perc_drawdown_after_entry[interval].append(max_perc_drawdown)

        # Collect stats for heatmap
        percent_stopped = (num_times_stopped_out[heatmap_stats_interval] / num_entry_signals) * 100
        percent_tp = (num_times_tp[heatmap_stats_interval] / num_entry_signals) * 100
        tp_fired_perc[k,j] = percent_tp
        sl_fired_perc[k,j] = percent_stopped
        if len(not_tp_or_sl_perc_change[heatmap_stats_interval]) == 0:
            expected_returns[k,j] = percent_stopped*stop_loss + percent_tp*tp
        else:
            expected_returns[k,j] = percent_stopped*stop_loss + percent_tp*tp + (100-(percent_stopped+percent_tp))*np.mean(not_tp_or_sl_perc_change[heatmap_stats_interval])

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

            percent_stopped = (num_times_stopped_out[interval] / num_entry_signals) * 100
            percent_tp = (num_times_tp[interval] / num_entry_signals) * 100
            print("SL percentage: %.2f" % (percent_stopped))
            print("TP percentage: %.2f" % (percent_tp))
            print("Not TP or SL end of interval change perc: %.2f" % (np.mean(not_tp_or_sl_perc_change[interval])*100))
            print("Expected return: %.2f" % (percent_stopped*stop_loss + percent_tp*tp + (100-(percent_stopped+percent_tp))*np.mean(not_tp_or_sl_perc_change[interval])))
            print('\n')


# Set up and draw heatmap
row_labels = ["%.0f%%"%(x*100) for x in stop_losses]
col_labels = ["%.0f%%"%(x*100) for x in take_profits]

for i in range(len(row_labels)):
    if row_labels[i] == "-100%": row_labels[i] = "none"
for i in range(len(col_labels)):
    if col_labels[i] == "100%": col_labels[i] = "none"
data = expected_returns

strings = [["%.1f\n%.1f\n%.2f"%(tp_fired_perc[k,j], sl_fired_perc[k,j], expected_returns[k,j]) for j in range(len(take_profits))] for k in range(len(stop_losses))]
strings = np.asarray(strings).reshape(len(stop_losses), len(take_profits))

time_str = "%d hours"%(stat_intervals[heatmap_stats_interval])
if stat_intervals[heatmap_stats_interval] == 24:
    time_str = "%d day"%int(stat_intervals[heatmap_stats_interval]/24)
elif stat_intervals[heatmap_stats_interval] > 24:
    time_str = "%d days"%int(stat_intervals[heatmap_stats_interval]/24)

title = "commando cross %.2f with SL and TP within %s" %(entry_on_crossover_of, time_str)
filename = "commando cross %.2f with SL and TP within %d hours" %(entry_on_crossover_of, stat_intervals[heatmap_stats_interval])
ylabel = "stop loss"
xlabel = "take profit"
draw_heatmap(row_labels, col_labels, data, prob_pallet=True, title=title, filename=filename, save=True, display=True, annot=strings, xlabel=xlabel, ylabel=ylabel)

