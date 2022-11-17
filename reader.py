import pandas as pd
import yfinance as yf
import os
import sys

sys.path.append(os.getcwd())

class Reader:

    # constructor to read YF data
    def __init__(self, start: str, end: str, tickers: list, freq: str = '1d'):
        self.start = start
        self.end = end
        self.tickers = tickers
        self.frequency = freq
    
    # constructor to read given data 
    def __init__(self, files: list) -> None:
        self.files = files

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
        for name in self.files:
            df = pd.read_csv('./stock_dfs/{}'.format(name))
            df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            dfs[name] = df
        return dfs