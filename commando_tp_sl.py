import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import commando_utils


# ------------------------------------------------------------------------------------------
# Backtest config
# ------------------------------------------------------------------------------------------

base_tf = 15
symbols = {'AAVEUSDT', 'ADAUSDT', 'AVAXUSDT', 'BNBUSDT', 'BTCUSDT', 'DCRUSDT', 'DOGEUSDT',
           'DOTUSDT', 'EOSUSDT', 'ETHBTC', 'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'RUNEUSDT',
           'SOLUSDT', 'SUSHIUSDT', 'UNIUSDT', 'XHVUSDT', 'XRPUSDT', 'YFIUSDT'}

stop_losses = [-1.0,-0.2,-0.15,-0.1,-0.07,-0.05,-0.04,-0.03,-0.02]
take_profits = [0.04,0.05,0.06,0.07,0.1,0.15,0.2,0.3,1.0]
entry_on_crossover_of = 3.0
stat_intervals = [4, 8, 12, 24, 24*2, 24*3, 24*4]
heatmap_stats_interval = 3


# ------------------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------------------

# Load commando signals and tradingview ticker data
signals = commando_utils.parse_commando(os.getcwd() + "/data/commando.html")
dfs = {s:pd.read_csv(os.getcwd() + "/data/BINANCE_" + s + ", " + str(base_tf) + ".csv") for s in symbols}


# ------------------------------------------------------------------------------------------
# Collect backtest stats
# ------------------------------------------------------------------------------------------

# Data to collect
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
                if commando.values[i] >= entry_on_crossover_of and commando.values[i - 1] < entry_on_crossover_of:

                    # Skip commando entry signals for which ticker data is missing
                    signal_time = commando.times[i]
                    ind = df.loc[df['time'] == signal_time].index
                    if len(ind) == 0:
                        continue
                    signal_ticker_index = ind[0]
                    num_entry_signals += 1

                    entry_close_price = df['close'].iloc[signal_ticker_index]

                    # Find price changes over stats collection intervals after the entry signal
                    for interval in stat_intervals:
                        price_interval = df['close'].iloc[signal_ticker_index:signal_ticker_index + int(interval*60/base_tf)]

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


# ------------------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------------------

# Set up and draw heatmap
row_labels = ["%.0f%%"%(x*100) for x in stop_losses]
col_labels = ["%.0f%%"%(x*100) for x in take_profits]

annotations = [["%.1f\n%.1f\n%.2f"%(tp_fired_perc[k,j], sl_fired_perc[k,j], expected_returns[k,j]) for j in range(len(take_profits))] for k in range(len(stop_losses))]
annotations = np.asarray(strings).reshape(len(stop_losses), len(take_profits))

title = "commando cross %.2f with SL and TP within %s" %(entry_on_crossover_of, commando_utils.interval_to_str(stat_intervals[heatmap_stats_interval]))
filename = "commando cross %.2f with SL and TP within %d hours" %(entry_on_crossover_of, stat_intervals[heatmap_stats_interval])
ylabel = "stop loss"
xlabel = "take profit"

commando_utils.draw_heatmap(row_labels, col_labels, expected_returns, title=title, filename=filename, save=True, display=True, annot=annotations, xlabel=xlabel, ylabel=ylabel)

