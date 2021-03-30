import traceback
from collections import Counter
from itertools import repeat
from math import sqrt
from multiprocessing import Manager, Pool, Process, cpu_count
from time import ctime, time
from warnings import catch_warnings, filterwarnings

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm


# n-step sarima forecast
def arimax_forecast(history, config, history_exog, forecast_exog, n_step):

    # define model
    model = ARIMA(history, exog=history_exog, order=config[0])

    # fit model
    model_fit = model.fit()
    # make n step forecast
    yhat = model_fit.forecast(exog=forecast_exog, steps=n_step)
    return yhat[n_step-1]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# create exogenous data for forecasting
def create_forecast_exog(data, index, step=1):
    return data[index:index+step].to_numpy().tolist()

# walk-forward validation for univariate data
def walk_forward_validation(data, cfg, n_step, cols_to_shift):
    train, test = create_dataset(data, cols_to_shift)
    exog = list(cols_to_shift.keys())

    # seed history with training dataset
    history = train['Cases'].to_numpy().tolist()
    history_exog = train[exog].to_numpy().tolist()
    predictions = list()

    # get max index
    if cols_to_shift:
        min_lag = cols_to_shift[min(cols_to_shift, key=cols_to_shift.get)]
    else:
        min_lag = 0
        
    if n_step < min_lag:
        max_index = len(test['Cases'].dropna())
    else:
        max_index = len(test) - n_step

    # step over each time-step in the test set
    for i in range(max_index):
        forecast_exog = create_forecast_exog(test[exog], i, n_step)
        # fit model and make forecast for history
        yhat = arimax_forecast(history, cfg, history_exog, forecast_exog, n_step)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test['Cases'].values[i].tolist())
        history_exog.append(test[exog].values[i].tolist())
    if n_step <= min_lag:
        predicted = pd.DataFrame(predictions, index=test[n_step:].dropna(subset=['Cases']).index.union(pd.date_range(start='1/1/2018', periods=n_step, freq='W')))
        # estimate prediction error only using indices with actual values
        error = measure_rmse(test[n_step:max_index]['Cases'].values, predictions[:-n_step])
    else:
        predicted = pd.DataFrame(predictions, index=test[n_step:max_index+n_step].index)
        error = measure_rmse(test[n_step:max_index]['Cases'].values, predictions[:-n_step])
    return error, predicted

# score a model, return None on failure
def score_model(data, scores, cfg, n_step, cols_to_shift, debug=False):
    result = None
    predicted = None
    # convert config to a key
    key = str(cols_to_shift)
    try:
        # never show warnings when grid searching, too noisy
        with catch_warnings():
            filterwarnings("ignore")
            result, predicted = walk_forward_validation(data, cfg, n_step, cols_to_shift)
    except Exception as e:
        raise e
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    scores.append([key, result, predicted])

def create_dataset(data, cols_to_shift=None, split=0.66):
    if cols_to_shift:
        for x in list(cols_to_shift):
            if cols_to_shift.get(x) == None:
                cols_to_shift.pop(x)
                
    if cols_to_shift:
        # add new date index based on max lag to dframe
        max_lag = cols_to_shift[max(cols_to_shift, key=cols_to_shift.get)]
        df_temp = pd.DataFrame(data, index=data[max_lag:].index.union(pd.date_range(start='1/1/2018', periods=max_lag, freq='W')))
        
        
        # shift columns by value
        for col in cols_to_shift:
            col_lag = cols_to_shift[col]
            df_temp[col] = df_temp[col].shift(col_lag)
        
        # new dframe with cases + features
        cols_to_return = ['Cases'] + list(cols_to_shift.keys())
        data = df_temp[cols_to_return].dropna(subset=list(cols_to_shift.keys()))

    size = int(len(data) * split)
    train, test = data[0:size], data[size:len(data)]
    if cols_to_shift:
        train, test = scale_data(train, test)

    return train, test


def scale_data(train, test):
    cols_to_scale = list((Counter(train.columns.to_list()) - Counter(['Cases'])).elements())
#     scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
    test[cols_to_scale] = scaler.transform(test[cols_to_scale])

    return train, test
