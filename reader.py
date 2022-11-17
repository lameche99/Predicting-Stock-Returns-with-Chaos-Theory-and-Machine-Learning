import pandas as pd
import yfinance as yf
import gc
import os
import sys

gc.enable()
sys.path.append(os.getcwd())

class Reader:

    # constructor to read YF data
    def __init__(
        self,
        tickers: list,
        start: str = '',
        end: str = '',
        frequency: str = '1d',
        yahoo: bool = True
        ):

        self.tickers = tickers
        self.start = start
        self.end = end
        self.frequency = frequency
        self.yahoo = yahoo
    
    # read data from Yahoo Finance from list of tickers
    def read_yahoo(self):
        dfs = dict()
        for t in self.tickers:
            data = yf.download(t, start=self.start, end=self.end, interval=self.frequency)
            dfs[t] = data
        return dfs
    
    # read given data
    def read_files(self):
        dfs = dict()
        for name in self.tickers:
            df = pd.read_csv('./stock_dfs/{}'.format(name))
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            gc.collect()
            dfs[name] = df
        return dfs
    
    # read data
    def read(self):
        if self.yahoo:
            return self.read_yahoo()
        else:
            return self.read_files()