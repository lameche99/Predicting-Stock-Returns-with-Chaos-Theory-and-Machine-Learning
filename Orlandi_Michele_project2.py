# %% [markdown]
# # **Predicting Stock Returns with Elliot Waves**
# 
# ## *Neural Network and Random Forest Classifiers*
# 
# ### *Michele Orlandi ISYE6767 Fall 2022*

# %% [markdown]
# # 1. **Setup**

# %% [markdown]
# ## 1.1 **Packages and Classes**

# %%
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

# %%
gc.enable()
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1.2 **Import Stocks Data**

# %%
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

# %%
# input parameters
start_date = '2000-01-01'
end_date = '2021-11-12'
exchange = 'nasd'

# %%
# read tickers from given file
tickers = pd.read_csv('./data/tickers.csv')['Ticker'].to_list()
large = pd.read_csv('./data/tickers_{}.csv'.format(exchange))
large['MarketCap'] = large['MarketCap'].apply(fix_mcap)
large = large.sort_values(by=['MarketCap'], ascending=False)
large_universe = large['Symbol'][:100].to_list()

# %%
# create reader object
reader_1 = Reader(tickers=tickers, start=start_date, end=end_date)
reader_2 = Reader(tickers=large_universe, start=start_date, end=end_date)

# %%
# retrieve data frames
dfs = reader_1.read_yahoo()

# %%
# retireve large universe data frames
large_dfs = reader_2.read_yahoo()

# %% [markdown]
# ## 1.3 **Helper Functions**

# %%
def get_scatter(xval: pd.Series, yval: pd.Series, yname: str, mode: str = 'markers'):
    fig = go.Scatter(
        mode=mode,
        x=xval,
        y=yval,
        name=yname
    )
    return fig

# %%
def plot_data(df: pd.DataFrame, title: str, acc_bench: float, prec_bench: float):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            get_scatter(
                df.index,
                df[col],
                col
            )
        )
    
    fig.add_hline(y=acc_bench, annotation_text='Accuracy Benchmark = {}'.format(acc_bench))
    fig.add_hline(y=prec_bench, annotation_text='Precision Benchmark = {}'.format(prec_bench))

    fig.update_layout(
        title=title,
        xaxis_title='Tickers',
        yaxis_title='Value'
    )
    return fig

# %% [markdown]
# ## 1.4 **Preprocess Data**

# %%
# preprocess data
processor = Processor()
engineer = Engineer()

# %%
# input parameters for technical indicators
fractal_period, holding_period = 9, 24
fast, slow, signal = 5, 34, 5
tide, wave, ripple = 5, 13, 34
feats = ['macd', 'signal', 'tide', 'wave',\
    'ripple', 'tops', 'bottoms', 'mfi_change',\
        'volume_change', 'prediction']

# %% [markdown]
# ### 1.4.1 **Small Universe Stocks**

# %%
# clean data and engineer predictor variables
# also get prediction variable and shift the values by HP -> future n_day return is positive or negative
feature_dict = dict()
for t in tickers:
    df = dfs.loc[:, t].copy()
    processor.clean_data(df)
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
    processor.scale_data(new_df)
    feature_dict[t] = new_df.loc[:, feats].copy()
    gc.collect()

# %% [markdown]
# ### 1.4.2 **Large Universe Stocks**

# %%
# clean ata and engineer predictor variables
# also get prediction variable and shift the values by HP -> future n_day return is positive or negative
large_feature_dict = dict()
for t in large_dfs.columns.levels[0].to_list():
    large_df = large_dfs.loc[:, t].copy()
    if large_df.isna().values.all():
        continue
    for col in large_df.columns:
        if large_df[col].isna().values.all():
            break
    else:
        # print(large_df.describe())
        processor.clean_data(large_df)
        # print(large_df)
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
        if new_df.empty:
            continue
        processor.scale_data(new_df)
        large_feature_dict[t] = new_df.loc[:, feats].copy()
        gc.collect()
    continue

# %%
del dfs, new_df, large_dfs, large_df, df

# %% [markdown]
# # 2. **Neural Network**

# %% [markdown]
# ## 2.1 **Small Universe Stocks**

# %%
# create a dictionary to store results
results = {
    'ticker': tickers,
    'accuracy': [],
    'precision': []
}

# %%
# # apply NN to each stock and store results
for t in feature_dict.keys():
    x_train, x_test, y_train, y_test = processor.split_data(
        feature_dict[t].loc[:, feature_dict[t].columns[:-1]].copy(),
        feature_dict[t].loc[:, 'prediction'].copy()
        )
    accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'mlp')
    results['accuracy'].append(round(accuracy, 5))
    results['precision'].append(round(precision, 5))

# %%
results = pd.DataFrame.from_dict(results, orient='columns').set_index('ticker').sort_values(by=['accuracy'], ascending=False)
gc.collect()

# %% [markdown]
# ## 2.2 **Large Universe Stocks**

# %%
# create a dictionary to store results
large_results = {
    'ticker': [],
    'accuracy': [],
    'precision': []
}

# %%
# # apply NN to each stock and store results
for t in large_feature_dict.keys():
    if (len(large_feature_dict[t]) * 0.4) < (holding_period * 5):
        continue
    else:
        x_train, x_test, y_train, y_test = processor.split_data(
            large_feature_dict[t].loc[:, large_feature_dict[t].columns[:-1]].copy(),
            large_feature_dict[t].loc[:, 'prediction'].copy()
            )
        accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'mlp')
        large_results['ticker'].append(t)
        large_results['accuracy'].append(round(accuracy, 5))
        large_results['precision'].append(round(precision, 5))

# %%
large_results = pd.DataFrame.from_dict(large_results, orient='columns').set_index('ticker').sort_values(
    by=['accuracy'],
    ascending=False).iloc[:20]
gc.collect()

# %% [markdown]
# ## 2.3 **Plot Results**

# %%
# fig = plot_data(results, 'Small Universe Neural Network Results', acc_bench=0.5014, prec_bench=0.5141)
# fig.show(renderer='png')

# %%
# large_fig = plot_data(large_results, 'Large Universe Neural Network Results', acc_bench=0.5014, prec_bench=0.5141)
# large_fig.show(renderer='png')

# %%
del x_train, x_test, y_train, y_test

# %% [markdown]
# # 3. **Support Vector Machine**

# %% [markdown]
# ## 3.1 **Small Universe Stocks**

# %%
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

# %%
for t in feature_dict.keys():
    x_train, x_test, y_train, y_test = processor.split_data(
        feature_dict[t].loc[:, feature_dict[t].columns[:-1]].copy(),
        feature_dict[t].loc[:, 'prediction'].copy()
        )
    accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'svm', params)
    svm_results['accuracy'].append(round(accuracy, 5))
    svm_results['precision'].append(round(precision, 5))

# %%
svm_results = pd.DataFrame.from_dict(svm_results, orient='columns').set_index('ticker').sort_values(by=['accuracy'], ascending=False)
gc.collect()

# %% [markdown]
# ## 3.2 **Large Universe Stocks**

# %%
# set up dictionary to store results
large_svm_results = {
    'ticker': [],
    'accuracy': [],
    'precision': []
}

# %%
for t in large_feature_dict.keys():
    if (len(large_feature_dict[t]) * 0.4) < (holding_period * 5):
        continue
    else:
        x_train, x_test, y_train, y_test = processor.split_data(
            large_feature_dict[t].loc[:, large_feature_dict[t].columns[:-1]].copy(),
            large_feature_dict[t].loc[:, 'prediction'].copy()
            )
        accuracy, precision = processor.fit_and_score(x_train, x_test, y_train, y_test, 'svm', params)
        large_svm_results['ticker'].append(t)
        large_svm_results['accuracy'].append(round(accuracy, 5))
        large_svm_results['precision'].append(round(precision, 5))

# %%
large_svm_results = pd.DataFrame.from_dict(large_svm_results, orient='columns').set_index('ticker').sort_values(
    by=['accuracy'],
    ascending=False).iloc[:20]
gc.collect()

# %% [markdown]
# ## 3.3 **Plot Results**

# %%
# svm_fig = plot_data(svm_results, 'Small Universe Support Vector Machine Results', acc_bench=0.5014, prec_bench=0.5141)
# svm_fig.show(renderer='png')

# %%
# large_svm_fig = plot_data(large_svm_results, 'Large Universe Support Vector Machine Results', acc_bench=0.5014, prec_bench=0.5141)
# large_svm_fig.show(renderer='png')
# %%
# export data to a csv file
results.to_csv('./small_NN_results.csv')
large_results.to_csv('./large_NN_results.csv')
svm_results.to_csv('./small_SVM_results.csv')
large_svm_results.to_csv('./large_SVM_results.csv')