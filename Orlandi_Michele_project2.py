# # **Predicting Stock Returns with Elliot Waves**
# 
# ## *Neural Network and Random Forest Classifiers*
# 
# ### *Michele Orlandi ISYE6767 Fall 2022*

# 1. **Setup**

# 1.1 **Packages and Classes**

import os
import sys
sys.path.append(os.getcwd())
import gc
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Orlandi_Michele_module2 import Engineer
from Orlandi_Michele_module1 import Reader
from Orlandi_Michele_module3 import Processor

gc.enable()
warnings.filterwarnings('ignore')

# 1.2 **Import Stocks Data**

def fix_mcap(x):
    if type(x) == str:
        if 'B' in x:
            end = x.find('B')
            val = float(x[1:end])
            return val * 10e9
        elif 'M' in x:
            end = x.find('M')
            val = float(x[1:end])
            return val * 10e6
    else:
        return float(x)

# input parameters
start_date = '2000-01-01'
end_date = '2021-11-12'
exchange = 'nasd'

# read tickers from given file
tickers = pd.read_csv('./data/tickers.csv')['Ticker'].to_list()
large = pd.read_csv('./data/tickers_{}.csv'.format(exchange))
large['MarketCap'] = large['MarketCap'].apply(fix_mcap)
large = large.sort_values(by=['MarketCap'], ascending=False)
large_universe = large['Symbol'][:100].to_list()

# create reader object
reader_1 = Reader(tickers=tickers, start=start_date, end=end_date)
reader_2 = Reader(tickers=large_universe, start=start_date, end=end_date)

# retrieve data frames
dfs = reader_1.read_yahoo()

# retireve large universe data frames
large_dfs = reader_2.read_yahoo()

# 1.4 **Preprocess Data**

# create processor and engineer objects
processor = Processor()
engineer = Engineer()

# input parameters for technical indicators
fractal_period, holding_period = 9, 24
fast, slow, signal = 5, 34, 5
tide, wave, ripple = 5, 13, 34
feats = ['macd', 'signal', 'tide', 'wave',\
    'ripple', 'tops', 'bottoms', 'mfi_change',\
        'volume_change', 'prediction']

# 1.4.1 **Small Universe Stocks**

# clean data and engineer predictor variables
# also get prediction variable and shift the values by HP -> future n_day return is positive or negative
feature_dict = dict()
for t in tickers:
    df = dfs.loc[:, t].copy()
    # clean data
    processor.clean_data(df)
    # create predictor variables
    new_df = engineer.engineer_features(
        df,
        period=fractal_period,
        fast=fast,
        slow=slow,
        signal=signal,
        tide=tide,
        wave=wave,
        ripple=ripple,
        holding_period=holding_period
    )
    new_df['prediction'] = new_df['prediction'].shift(-holding_period)
    new_df = new_df[np.isfinite(new_df).all(1)].copy()
    # standardize data
    processor.scale_data(new_df)
    feature_dict[t] = new_df.loc[:, feats].copy()
    gc.collect()

# 1.4.2 **Large Universe Stocks**

# clean ata and engineer predictor variables
# also get prediction variable and shift the values by HP -> future n_day return is positive or negative
large_feature_dict = dict()
for t in large_dfs.columns.levels[0].to_list():
    large_df = large_dfs.loc[:, t].copy()
    # check if data frame is full of nan values
    if large_df.isna().values.all():
        continue
    # check if any of the columns is full of nan values
    for col in large_df.columns:
        if large_df[col].isna().values.all():
            break
    else:
        # clean data
        processor.clean_data(large_df)
        # add predictor variables
        new_df = engineer.engineer_features(
            large_df,
            period=fractal_period,
            fast=fast,
            slow=slow,
            signal=signal,
            tide=tide,
            wave=wave,
            ripple=ripple,
            holding_period=holding_period
        )
        new_df['prediction'] = new_df['prediction'].shift(-holding_period)
        new_df = new_df[np.isfinite(new_df).all(1)].copy()
        # check if the new dataframe is empty
        if new_df.empty:
            continue
        # standardize data
        processor.scale_data(new_df)
        large_feature_dict[t] = new_df.loc[:, feats].copy()
        gc.collect()
    continue

del dfs, new_df, large_dfs, large_df, df

# 2. **Neural Network**
# 2.1 **Small Universe Stocks**

# create a dictionary to store results
results = {
    'ticker': tickers,
    'accuracy': [],
    'precision': []
}

# # apply NN to each stock and store results
for t in feature_dict.keys():
    # split data
    x_train, x_test, y_train, y_test = processor.split_data(
        feature_dict[t].loc[:, feature_dict[t].columns[:-1]].copy(),
        feature_dict[t].loc[:, 'prediction'].copy()
        )
    # fit model and get scores
    accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'mlp')
    results['accuracy'].append(round(accuracy, 5))
    results['precision'].append(round(precision, 5))

results = pd.DataFrame.from_dict(results, orient='columns').set_index('ticker').sort_values(by=['accuracy'], ascending=False)
gc.collect()

# 2.2 **Large Universe Stocks**

# create a dictionary to store results
large_results = {
    'ticker': [],
    'accuracy': [],
    'precision': []
}

# # apply NN to each stock and store results
for t in large_feature_dict.keys():
    # check if there is enough data for model
    if (len(large_feature_dict[t]) * 0.4) < (holding_period * 5):
        continue
    else:
        # split data
        x_train, x_test, y_train, y_test = processor.split_data(
            large_feature_dict[t].loc[:, large_feature_dict[t].columns[:-1]].copy(),
            large_feature_dict[t].loc[:, 'prediction'].copy()
            )
        # fit model and get results
        accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'mlp')
        large_results['ticker'].append(t)
        large_results['accuracy'].append(round(accuracy, 5))
        large_results['precision'].append(round(precision, 5))

large_results = pd.DataFrame.from_dict(large_results, orient='columns').set_index('ticker').sort_values(
    by=['accuracy'],
    ascending=False)
gc.collect()

# 2.3 **Plot Results**

del x_train, x_test, y_train, y_test

# 3. **Support Vector Machine**
# 3.1 **Small Universe Stocks**

# set up dictionary to store results and paramters to be tunes
params = {
    'C': [0.1, 1, 10, 100, 1000], 
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
svm_results = {
    'ticker': tickers,
    'accuracy': [],
    'precision': []
}

# split data, fit SVC and get store results into dictionary
for t in feature_dict.keys():
    # split data
    x_train, x_test, y_train, y_test = processor.split_data(
        feature_dict[t].loc[:, feature_dict[t].columns[:-1]].copy(),
        feature_dict[t].loc[:, 'prediction'].copy()
        )
    # fit model and get scores
    accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'svm', params)
    svm_results['accuracy'].append(round(accuracy, 5))
    svm_results['precision'].append(round(precision, 5))

svm_results = pd.DataFrame.from_dict(svm_results, orient='columns').set_index('ticker').sort_values(
    by=['accuracy'],
    ascending=False)
gc.collect()

# 3.2 **Large Universe Stocks**

# set up dictionary to store results
large_svm_results = {
    'ticker': [],
    'accuracy': [],
    'precision': []
}

# split data, fit SVC and get store results into dictionary
for t in large_feature_dict.keys():
    if (len(large_feature_dict[t]) * 0.4) < (holding_period * 5):
        continue
    else:
        # split data
        x_train, x_test, y_train, y_test = processor.split_data(
            large_feature_dict[t].loc[:, large_feature_dict[t].columns[:-1]].copy(),
            large_feature_dict[t].loc[:, 'prediction'].copy()
            )
        # fit model and get scores
        accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'svm', params)
        large_svm_results['ticker'].append(t)
        large_svm_results['accuracy'].append(round(accuracy, 5))
        large_svm_results['precision'].append(round(precision, 5))

large_svm_results = pd.DataFrame.from_dict(large_svm_results, orient='columns').set_index('ticker').sort_values(
    by=['accuracy'],
    ascending=False)
gc.collect()

# export data to a csv file
results.to_csv('./small_NN_results.csv')
large_results.to_csv('./large_NN_results.csv')
svm_results.to_csv('./small_SVM_results.csv')
large_svm_results.to_csv('./large_SVM_results.csv')