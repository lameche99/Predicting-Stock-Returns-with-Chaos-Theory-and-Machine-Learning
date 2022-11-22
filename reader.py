import yfinance as yf

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
        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            interval=self.frequency,
            progress=False,
            group_by='ticker',
            show_errors=False)
        return data