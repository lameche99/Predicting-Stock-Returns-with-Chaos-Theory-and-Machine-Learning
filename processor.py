import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Processor:

    def __init__(self) -> None:
        pass

    def clean_data(self, df: pd.DataFrame):
        """
        This function forward fills missing values in the columns
        and standardizes the data
        """
        scaler = StandardScaler()
        for col in df.columns:
            df[col] = scaler.fit_transform(df[col].ffill())

    def split_data(self, X, y, test_s: float = 0.4, rand_s: int = 903436):
        """
        This function splits predictor and prediction into train and test sets
        """
        return train_test_split(X, y, test_size=test_s, random_state=rand_s)
            