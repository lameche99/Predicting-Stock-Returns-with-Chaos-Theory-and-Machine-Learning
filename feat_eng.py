import pandas as pd
import pandas_ta as ta
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
        :return: tuple(pd.Series, pd.Series) - returns two series of binary indicators
        pos_macd: 1=(fast - slow)>signal, 0=else
        neg_macd: 1=(fast - slow)<signal, 0=else
        """

        f = ta.ema(prices, length=fast)
        s = ta.ema(prices, length=slow)
        sig = ta.ema(prices, length=signal)
        macd = f - s

        pos_macd = np.where(macd > sig, 1, 0)
        neg_macd = np.where(macd < sig, 1, 0)

        return pos_macd, neg_macd
    
    def get_alligator(self, prices: pd.Series, tide: int, wave: int, ripple: int):
        """
        This function calculates the Alligator indicator (TWR) for a price series
        :param prices: pd.Series -- time series of stock prices
        :param tide: int -- tide moving average period
        :param wave: int -- wave moving average period
        :param ripple: int -- ripple moving average period
        :return: tuple(pd.Series, pd.Series) - returns two series of binary indicators
        pos_twr: 1=(tide > wave & wave > ripple), 0=else
        neg_twr: 1=(tide < wave & wave < ripple), 0=else
        """

        t = prices.rolling(tide).mean()
        w = prices.rolling(wave).mean()
        r = prices.rolling(ripple).mean()

        pos_twr = np.where((t > w) & (w > r), 1, 0)
        neg_twr = np.where((t < w) & (w < r), 1, 0)

        return pos_twr, neg_twr
    

    def get_fractals(self, highs: pd.Series, lows: pd.Series, period: int):
        """
        This function calculates the top and bottom fractals on a certain number of time intervals
        :param period: int -- number of time intervals to look for a fractal pattern
        :param prices: pd.Series -- time series of stock prices
        :return: tuple(pd.Series, pd.Series) -- returns two series of binary indicators
        tops: 1=max high of n_period neighbors, 0=else
        bottoms: 1=min low of n_period neighbors, 0=else
        """

        tops = np.where(
            highs == highs.rolling(period, center=True).max(), 1, 0
        )
        bottoms = np.where(
            lows == lows.rolling(period, center=True).min(), 1, 0
        )

        return tops, bottoms
    
    def get_squat(self, highs: pd.Series, lows: pd.Series, volume: pd.Series):
        """
        This function calculates which bars are squats
        :param highs: pd.Series -- time series of stock highs
        :param lows: pd.Series -- time series of stock lows
        :param volume: pd.Series -- time series of volume
        :return: pd.Series -- series of binary indicators
        squats: 1=(volume change > 0 & MFI change < 0), 0=else
        where MFI = (High - Low) / Volume
        """

        mfi = (highs - lows) / volume
        squats = np.where(
            (volume.pct_change() > 0) & (mfi.pct_change() < 0), 1, 0
        )

        return squats

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

        df['macd_pos'], df['macd_neg'] = self.get_macd(prices, fast, slow, signal)
        df['alligator_pos'], df['alligator_neg'] = self.get_alligator(prices, tide, wave, ripple)
        df['tops'], df['bottoms'] = self.get_fractals(highs, lows, period)
        df['squat'] = self.get_squat(highs, lows, volume)
        df['prediction'] = self.get_prediction(prices, holding_period)

        return df