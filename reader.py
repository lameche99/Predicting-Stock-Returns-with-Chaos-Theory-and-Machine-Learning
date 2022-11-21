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
        frequency: str = '1d'
        ):

        self.tickers = tickers
        self.start = start
        self.end = end
        self.frequency = frequency
    
    # read data from Yahoo Finance from list of tickers
    def read_yahoo(self):
        dfs = dict()
        for t in self.tickers:
            data = yf.download(t, start=self.start, end=self.end, interval=self.frequency)
            dfs[t] = data
        return dfs