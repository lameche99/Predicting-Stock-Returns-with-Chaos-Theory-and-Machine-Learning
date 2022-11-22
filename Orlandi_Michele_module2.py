import pandas as pd
from ta.trend import ema_indicator
import numpy as np

class Engineer:

    # no-args constructor
    def __init__(self):
        pass
    

    def get_macd(self, prices: pd.Series, fast: int, slow: int, signal: int):
        """
        This function returns the postive and negative macd of a price series
        :param prices: pd.Series - time series of stock prices
        :param fast: int - fast EMA window
        :param slow: int - slow EMA window
        :param signal: int - signal EMA window
        :return: tuple(pd.Series, pd.Series) - returns two time series
        macd: fast EMA - slow EMA
        signal: EMA of macd
        """

        f = ema_indicator(prices, window=fast)
        s = ema_indicator(prices, window=slow)
        macd = f - s
        sig = ema_indicator(macd, window=signal)

        return macd, sig
    
    def get_alligator(self, prices: pd.Series, tide: int, wave: int, ripple: int):
        """
        This function calculates the Alligator indicator (TWR) for a price series
        :param prices: pd.Series -- time series of stock prices
        :param tide: int -- tide moving average period
        :param wave: int -- wave moving average period
        :param ripple: int -- ripple moving average period
        :return: tuple(pd.Series, pd.Series) - returns three time series
        t: fast MA
        w: medium MA
        r: slow MA
        """
        t = prices.rolling(tide).mean()
        w = prices.rolling(wave).mean()
        r = prices.rolling(ripple).mean()

        return t, w, r
    

    def get_fractals(self, highs: pd.Series, lows: pd.Series, period: int):
        """
        This function calculates the top and bottom fractals on a certain number of time intervals
        :param period: int -- number of time intervals to look for a fractal pattern
        :param prices: pd.Series -- time series of stock prices
        :return: tuple(pd.Series, pd.Series) -- returns two time series
        tops: max high of n_neighbors left and right
        bottoms: min low of n_neighbors left and right
        """

        tops = highs.rolling(period, center=True).max()
        bottoms = lows.rolling(period, center=True).min()

        return tops, bottoms
    
    def get_squat(self, highs: pd.Series, lows: pd.Series, volume: pd.Series):
        """
        This function calculates which bars are squats
        :param highs: pd.Series -- time series of stock highs
        :param lows: pd.Series -- time series of stock lows
        :param volume: pd.Series -- time series of volume
        :return: pd.Series -- two time series
        mfi_change: percent change in MFI
        where MFI = (High - Low) / Volume
        vol_change: percent change in volume
        """

        mfi = (highs - lows) / volume
        mfi_change = mfi.pct_change().astype(np.float64)
        vol_change = volume.pct_change().astype(np.float64)
        
        return mfi_change, vol_change

    def get_prediction(self, prices: pd.Series, holding_period: int):
        """
        This function returns the prediction variable or whether the
        HPR is positive or negative
        :param prices: pd.Series -- time series of stock prices
        :param holding_period: int -- holding period length
        :return: pd.Series -- series of binary indicators: 1=positive returns, 0=else
        """
        rets = prices.pct_change(holding_period)
        prediction = np.where(rets > 0, 1, 0)
        return prediction
    

    def engineer_features(
        self,
        df: pd.DataFrame,
        period: int = 5,
        fast: int = 5,
        slow: int = 34,
        signal: int = 5,
        tide: int = 5,
        wave: int = 13,
        ripple: int = 34,
        holding_period: int = 5
        ):
        """
        This function calculates predictor variables with One-Hot-Encoding
        :param df: pd.DataFrame -- data frame with new predictor variables
        :param period: int -- number of time intervals to look for a fractal pattern
        :param fast: int - fast EMA window
        :param slow: int - slow EMA window
        :param signal: int - signal EMA window
        :param tide: int -- tide moving average period
        :param wave: int -- wave moving average period
        :param ripple: int -- ripple moving average period
        :return: pd.DataFrame -- original dataframe with new features
        """

        prices = df['Close']
        highs = df['High']
        lows = df['Low']
        volume = df['Volume']

        df['macd'], df['signal'] = self.get_macd(prices, fast, slow, signal)
        df['tide'], df['wave'], df['ripple'] = self.get_alligator(prices, tide, wave, ripple)
        df['tops'], df['bottoms'] = self.get_fractals(highs, lows, period)
        df['mfi_change'], df['volume_change'] = self.get_squat(highs, lows, volume)
        df['prediction'] = self.get_prediction(prices, holding_period)

        return df