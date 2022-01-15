import os
import re
import time
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from bs4 import BeautifulSoup


# ------------------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------------------

# Stores time and value of commando signals
class CommandoSymbol:
    def __init__(self, symbol_name):
        self.symbol_name = symbol_name
        self.times = []
        self.values = []
        self.deltas = []
        self.num_entries = 0

def add_call(self, time, value, delta=None):
        self.times.append(time)
        self.values.append(value)
        if delta is not None:
            self.deltas.append(delta)
        self.num_entries += 1


# ------------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------------

# Get a descriptor for a time interval (in hours): X hours, day, or days
def interval_to_str(interval):
    if interval < 24:
        return "%d hours" % (interval)
    elif interval == 24:
        return "%d day" % (int(interval/24))
    return "%d days" % (int(interval/24))


# Load commando data, parse signals into a dict {str symbol name: CommandoSymbol signals}
def parse_commando(path, save_score_deltas=False):
    # Load commando data
    t = None
    with open(path) as file:
        soup = BeautifulSoup(file, features="lxml")
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

            # Parse date
            timestamp = split[0] + ' ' + split[1]
            timestamp_parsed = calendar.timegm(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
            score = float(split[5])

            # Parse price delta since last commando update
            if save_score_deltas:
                delta = score - float(re.search("T-4=[-]?\d+.\d+", line).group()[4:])

                signals_by_time[timestamp_parsed].append((symbol, score, delta))
                signals[symbol].add_call(timestamp_parsed, score, delta)
            else:
                signals_by_time[timestamp_parsed].append((symbol, score))
                signals[symbol].add_call(timestamp_parsed, score)

    return signals


# Draw a labeled heatmap with the option to save to file
def draw_heatmap(row_labels, col_labels, data, title=None, filename=None, save=False, display = True, annot=None, xlabel=None, ylabel=None):
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
        plt.savefig(os.getcwd() + "/" + filename + ".png")
    if display:
        plt.show()
