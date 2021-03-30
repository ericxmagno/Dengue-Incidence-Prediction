from math import sqrt
from multiprocessing import cpu_count
from itertools import repeat
from multiprocessing import Process, Manager, Pool
from warnings import catch_warnings
from warnings import filterwarnings
from tqdm import tqdm
import traceback
from time import time, ctime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd


# n-step sarima forecast
def arima_forecast(history, config, n_step):
    order = config
    # define model
    model = ARIMA(history, order=order[0])
    # fit model
    model_fit = model.fit()
    # make n step forecast
    yhat = model_fit.forecast(steps=n_step)
    return yhat[n_step-1]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# walk-forward validation for univariate data
def walk_forward_validation(data, cfg, n_step):
    train = data[0]
    test = data[1]
    predictions = list()
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = arima_forecast(history, cfg, n_step)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    predicted = pd.DataFrame(predictions, index=test[n_step:].index.union(pd.date_range(start='1/1/2018', periods=n_step, freq='W')))
    # estimate prediction error only using indices with actual values
    error = measure_rmse(test[n_step:], predictions[:-n_step])
    return error, predicted

# score a model, return None on failure
def score_model(data, scores, cfg, n_step, debug=False):
    result = None
    predicted = None
    # convert config to a key
    key = str(cfg)
    try:
        # never show warnings when grid searching, too noisy
        with catch_warnings():
            filterwarnings("ignore")
            result, predicted = walk_forward_validation(data, cfg, n_step)
    except Exception as e:
        raise e
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    scores.append([key, result, predicted])