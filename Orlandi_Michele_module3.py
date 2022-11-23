import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class Processor:

    def __init__(self) -> None:
        pass

    def clean_data(self, df: pd.DataFrame):
        """
        This function forward fills missing values in the columns 
        """
        for col in df.columns:
            df[col] = df[col].ffill()

    def scale_data(self, df: pd.DataFrame):
        """
        This function standardizes the data
        """
        scaler = StandardScaler()

        momentum = np.asarray(df.loc[:, ['macd', 'signal']])
        twr = np.asarray(df.loc[:, ['tide', 'wave', 'ripple']])
        fractals = np.asarray(df.loc[:, ['tops', 'bottoms']])
        squats = np.asarray(df.loc[:, ['mfi_change', 'volume_change']])

        scaled_mom = scaler.fit_transform(momentum)
        scaled_twr = scaler.fit_transform(twr)
        scaled_frac = scaler.fit_transform(fractals)
        scaled_squat = scaler.fit_transform(squats)

        df['macd'], df['signal'] = scaled_mom[:,0], scaled_mom[:,1]
        df['tide'], df['wave'], df['ripple'] = scaled_twr[:,0], scaled_twr[:,1], scaled_twr[:,2]
        df['tops'], df['bottoms'] = scaled_frac[:,0], scaled_frac[:,1]
        df['mfi_change'], df['volume_change'] = scaled_squat[:,0], scaled_squat[:,1]

    def split_data(self, X, y, test_s: float = 0.4, rand_s: int = 903436):
        """
        This function splits predictor and prediction into train and test sets
        """
        return train_test_split(X, y, test_size=test_s, random_state=rand_s, train_size=1-test_s)
    
    @ignore_warnings(category=ConvergenceWarning)
    def fit_and_score(
        self,
        X_train: pd.DataFrame,
        X_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
        model: str,
        params: dict = {}
        ):

        if model == 'mlp':
            mlp = MLPClassifier(
                hidden_layer_sizes=(50,50,50),
                activation='tanh',
                solver='lbfgs',
                alpha=0.0001,
                learning_rate='adaptive'
            )
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred)

            return accuracy, precision
        else:
            svm = SVC()
            grid = GridSearchCV(svm, param_grid=params, refit=True)
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred)
            return accuracy, precision