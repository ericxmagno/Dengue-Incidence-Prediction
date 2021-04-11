import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
from streamlit import caching
import streamlit.components.v1 as components

def objectives():
    st.write('')
    st.header('Objective')
    st.write('-----------------------------------------------------------------------') 
    st.subheader('- Prediction of dengue outbreaks to allow  for early response.')

    

def cleaned_data():
    caching.clear_cache()
    st.write('')
    st.header('Cleaned Data')
    st.write('-----------------------------------------------------------------------') 

    st.subheader('Weekly Dengue Cases')
    df_dengue_weekly = pd.read_csv('data/clean/dengue_cases.csv', index_col=0)
    st.write(df_dengue_weekly)
    st.write('')
    
    st.subheader('Weekly Climate Data')
    df_weather = pd.read_csv('data/clean/weather_data.csv', header=[0,1], index_col=[0])
    st.write(df_weather)
    st.write('')
    
    st.subheader('Dengue Cases by District')
    df_dengue_district = pd.read_csv('data/clean/dengue_cases_by_district.csv')
    st.write(df_dengue_district)
    st.write('')
        

def district_eda():
    caching.clear_cache()
    st.write('')
    st.header('EDA - Dengue Cases by District')
    st.write('-----------------------------------------------------------------------') 

    HtmlFile = open("notebooks/TimeSliderChoropleth.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 600)

def eda_features():
    st.write('')
    st.header('EDA - Features')
    st.write('-----------------------------------------------------------------------') 

    st.subheader('Comprehensive Plot of Available Data')
    image = Image.open('figures/all_plots.png').convert('RGB')
    st.image(image, caption='')
    st.write(
        '''
Observations:
- There seems to be a seasonal component to dengue cases.
- A spike/outbreak of dengue cases every 2 years can be seen.
        '''
    )
    st.write('')

    st.subheader('Seasonal Decomposition of Dengue Cases')
    image = Image.open('figures/seasonal_cases.png').convert('RGB')
    st.image(image, caption='')
    st.write(
        '''
Observations:
- A seasonal component for dengue cases can be verified.
        '''
    )
    st.write('')

    st.subheader('Correlation Matrix of Features')
    image = Image.open('figures/corr.png').convert('RGB')
    st.image(image, caption='')
    st.write(
        '''
Observations:
- RAINFALL mean and max seem to be highly correlated.
- TMAX and TMEAN seem to be highly correlated.
- All RH values seem to be highly correlated.

For simplicity, use 5 features:
- Rainfall mean
- TMIN
- TMEAN
- RH mean
- Windspeed mean
        '''
    )
    st.write('')


def arima():
    caching.clear_cache()
    st.write('')
    st.header('Modeling - ARIMAX')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Check Stationary of Cases using AD-Fuller Test')
    if st.checkbox('Show code', value=False, key="1"):
        st.code("""
from statsmodels.tsa.stattools import coint, adfuller

def check_for_stationarity(X, cutoff=0.01):
    pvalue = adfuller(X)[1]
    print(adfuller(X)[0])
    print(adfuller(X)[4])
    if pvalue < cutoff:
        print(f'p-value = {str(pvalue)} The series {X.name} is likely stationary.')
    else:
        print(f'p-value = {str(pvalue)} The series {X.name} is likely not stationary.')

check_for_stationarity(df_merged['Cases'])
        """, language="python")
    st.text('''
{'1%': -3.44455286264131, '5%': -2.8678027030003483, '10%': -2.5701057817594894}
p-value = 8.589960084412457e-06 The series Cases is likely stationary.
    ''')
    st.write('Although AD-Fuller is not without its caveats, for simplicity let us assume that time series data of dengue cases is indeed stationary.')
    st.write('')

    st.subheader('Grid Search of ARIMA Hyperparameters')
    if st.checkbox('Show code', value=False, key="2"):
        st.code("""
# create a set of arima configs to try
def arima_configs():
    models = list()
    # define config lists
    p_params = [0, 1, 2, 3]
    d_params = [0, 1]
    q_params = [0, 1, 2, 3]
#     t_params = ['n','c']
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                    cfg = [(p,d,q)]
                    models.append(cfg)
    return models
        """, language="python")
    st.write('')

    st.subheader('Top 3 Configurations for 1 Week Ahead and Graph of Best Configuration')
    st.text('''
[(1, 0, 3)] 31.036296623339823
[(1, 0, 0)] 31.377830675335346
[(3, 0, 0)] 31.40841720572334
    ''')
    image = Image.open('figures/arima_1.png').convert('RGB')
    st.image(image, caption='')
    st.write('')

    st.subheader('Top 3 Configurations for 4 Weeks Ahead and Graph of Best Configuration')
    st.text('''
[(1, 0, 3)] 56.02651805468894
[(3, 0, 0)] 56.508972646446274
[(1, 0, 2)] 56.722229305021195
    ''')
    image = Image.open('figures/arima_2.png').convert('RGB')
    st.image(image, caption='')
    st.write('')

    st.subheader('Top 3 Configurations for 12 Weeks Ahead and Graph of Best Configuration')
    st.text('''
[(1, 0, 0)] 72.68145886383068
[(1, 0, 3)] 72.70490814211776
[(3, 0, 0)] 72.94062963556097
    ''')
    image = Image.open('figures/arima_3.png').convert('RGB')
    st.image(image, caption='')
    st.write('')


def arimax():
    caching.clear_cache()
    st.write('')
    st.header('Modeling - ARIMAX')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Grid Search of Best Exogenous Variable Combinations Based on Top 2 Lags')
    if st.checkbox('Show code', value=False, key="2"):
        st.code("""
# create a set of sarima configs to try
def exog_configs(week):
    features = list()
    # define config lists
    if week == 1:
        rainfall = [0, 1, None]
        tmin = [0, 2, None]
        tmean = [0, 1, None]
        rhmean = [0, 3, None]
        windspeed = [0, 1, None]
    elif week == 4:
        rainfall = [0, 6, None]
        tmin = [0, 4, None]
        tmean = [0, 6, None]
        rhmean = [4, 6, None]
        windspeed = [6, 9, None]
    else:
        rainfall = [5, 6, None]
        tmin = [4, 8, None]
        tmean = [5, 6, None]
        rhmean = [2, 4, None]
        windspeed = [11, 12, None]
        
    rainfall = create_exog_dict_from_list('RAINFALL mean', rainfall)
    tmin = create_exog_dict_from_list('TMIN', tmin)
    tmean = create_exog_dict_from_list('TMEAN', tmean)
    rhmean = create_exog_dict_from_list('RH mean', rhmean)
    windspeed = create_exog_dict_from_list('WINDSPEED mean', windspeed)

    # create config instances
    for r in rainfall:
        for t1 in tmin:
            for t2 in tmean:
                for rh in rhmean:
                    for w in windspeed:
                        cfg = {**r, **t1, **t2, **rh, **w}
                        features.append(cfg)
    return features
        """, language="python")
    st.write('')

    st.subheader('Top 3 Exog Variable Combinations for 1 Week Ahead and Graph of Best Configuration')
    st.text('''
{'RAINFALL mean': None, 'TMIN': 2, 'TMEAN': 0, 'RH mean': 3, 'WINDSPEED mean': 0} 30.909814314717288
{'RAINFALL mean': None, 'TMIN': 0, 'TMEAN': 0, 'RH mean': None, 'WINDSPEED mean': None} 30.911667496640728
{'RAINFALL mean': 0, 'TMIN': None, 'TMEAN': None, 'RH mean': 0, 'WINDSPEED mean': None} 30.938076823232958
    ''')
    image = Image.open('figures/arimax_1.png').convert('RGB')
    st.image(image, caption='')
    st.write('')

    st.subheader('Top 3 Exog Variable Combinations for 4 Weeks Ahead and Graph of Best Configuration')
    st.text('''
{'RAINFALL mean': 6, 'TMIN': None, 'TMEAN': 6, 'RH mean': 6, 'WINDSPEED mean': 6} 55.0306756892499
{'RAINFALL mean': 0, 'TMIN': None, 'TMEAN': 0, 'RH mean': None, 'WINDSPEED mean': None} 55.46235205200629
{'RAINFALL mean': 6, 'TMIN': 4, 'TMEAN': None, 'RH mean': 4, 'WINDSPEED mean': 6} 55.49240658571921
    ''')
    image = Image.open('figures/arimax_2.png').convert('RGB')
    st.image(image, caption='')
    st.write('')

    st.subheader('Top 3 Exog Variable Combinations for 12 Weeks Ahead and Graph of Best Configuration')
    st.text('''
{'RAINFALL mean': None, 'TMIN': 8, 'TMEAN': None, 'RH mean': None, 'WINDSPEED mean': None} 69.35622680328314
{'RAINFALL mean': None, 'TMIN': 4, 'TMEAN': None, 'RH mean': None, 'WINDSPEED mean': None} 70.96584904196253
{'RAINFALL mean': 6, 'TMIN': None, 'TMEAN': 5, 'RH mean': None, 'WINDSPEED mean': 12} 70.97917019983734
    ''')
    image = Image.open('figures/arimax_3.png').convert('RGB')
    st.image(image, caption='')
    st.write('')

def prophet():
    caching.clear_cache()
    st.write('')
    st.header('Modeling - Prophet')
    st.write('-----------------------------------------------------------------------') 
    st.write('')
    
    st.subheader('Intuitive Model for Dengue Cases Predicted 1 Week Ahead')
    if st.checkbox('Show code', value=False, key="1"):
        st.code("""
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

n_weeks = 1

history = train.copy()
predictions = list()

for t in tqdm(range(len(test))):
    m = Prophet(growth='logistic',
        seasonality_mode='multiplicative', daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
    ).add_seasonality(name='biyearly', period=730.5, fourier_order=15, prior_scale=25
    ).add_seasonality(name='yearly', period=365.25, fourier_order=5)
    m.fit(history)
    future = m.make_future_dataframe(periods=n_weeks, freq='W')
    future['floor'] = 10
    future['cap'] = 500
    forecast = m.predict(future)
    yhat = forecast[-1:]['yhat'].values[0]
    predictions.append(yhat)
    history = history.append(test.loc[len(train)+t])
pred = pd.DataFrame(predictions, index=pd.DatetimeIndex(test[n_weeks-1:]['ds']).union(pd.date_range(start='1/1/2018', periods=n_weeks-1, freq='W')))
rmse = np.sqrt(mean_squared_error(test['y'], predictions))
print(f'rmse: {rmse}')
plt.figure(figsize=(12, 6), dpi=80)
plt.title('Dengue Cases Predicted 1 Week Ahead with Prophet')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.plot(df_merged['Cases'], color='C0')
plt.plot(pred, color='red')
plt.show()
        """, language="python")
    st.text('''
rmse: 33.25570726619088
    ''')
    image = Image.open('figures/prophet_1.png').convert('RGB')
    st.image(image, caption='')
    st.write('')

    st.subheader('Intuitive Model for Dengue Cases Predicted 4 Weeks Ahead')
    if st.checkbox('Show code', value=False, key="2"):
        st.code("""
n_weeks = 4

history = train.copy()
predictions = list()

for t in tqdm(range(len(test))):
    m = Prophet(growth='logistic',
        seasonality_mode='multiplicative', daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
    ).add_seasonality(name='biyearly', period=730.5, fourier_order=15, prior_scale=25
    ).add_seasonality(name='yearly', period=365.25, fourier_order=5)
    m.fit(history)
    future = m.make_future_dataframe(periods=n_weeks, freq='W')
    future['floor'] = 10
    future['cap'] = 500
    forecast = m.predict(future)
    yhat = forecast[-1:]['yhat'].values[0]
    predictions.append(yhat)
    history = history.append(test.loc[len(train)+t])
pred = pd.DataFrame(predictions, index=pd.DatetimeIndex(test[n_weeks-1:]['ds']).union(pd.date_range(start='1/1/2018', periods=n_weeks-1, freq='W')))
rmse = np.sqrt(mean_squared_error(test['y'], predictions))
print(f'rmse: {rmse}')
plt.figure(figsize=(12, 6), dpi=80)
plt.title('Dengue Cases Predicted 4 Weeks Ahead with Prophet')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.plot(df_merged['Cases'], color='C0')
plt.plot(pred, color='red')
plt.show()
        """, language="python")
    st.text('''
rmse: 48.14180502101209
    ''')
    image = Image.open('figures/prophet_2.png').convert('RGB')
    st.image(image, caption='')
    st.write('')

    st.subheader('Intuitive Model for Dengue Cases Predicted 12 Weeks Ahead')
    if st.checkbox('Show code', value=False, key="3"):
        st.code("""
n_weeks = 12

history = train.copy()
predictions = list()

for t in tqdm(range(len(test))):
    m = Prophet(growth='logistic',
        seasonality_mode='multiplicative', daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
    ).add_seasonality(name='biyearly', period=730.5, fourier_order=10, prior_scale=25
    ).add_seasonality(name='yearly', period=365.25, fourier_order=5)
    m.fit(history)
    future = m.make_future_dataframe(periods=n_weeks, freq='W')
    future['floor'] = 10
    future['cap'] = 500
    forecast = m.predict(future)
    yhat = forecast[-1:]['yhat'].values[0]
    predictions.append(yhat)
    history = history.append(test.loc[len(train)+t])
pred = pd.DataFrame(predictions, index=pd.DatetimeIndex(test[n_weeks-1:]['ds']).union(pd.date_range(start='1/1/2018', periods=n_weeks-1, freq='W')))
rmse = np.sqrt(mean_squared_error(test['y'], predictions))
print(f'rmse: {rmse}')
plt.figure(figsize=(12, 6), dpi=80)
plt.title('Dengue Cases Predicted 12 Weeks Ahead with Prophet')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.plot(df_merged['Cases'], color='C0')
plt.plot(pred, color='red')
plt.show()
        """, language="python")
    st.text('''
rmse: 77.81631422355609
    ''')
    image = Image.open('figures/prophet_3.png').convert('RGB')
    st.image(image, caption='')
    st.write('')
    
  
def results():
    caching.clear_cache()
    st.write('')
    st.header('Results')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Table Comparing the RMSE of Models')
    image = Image.open('figures/results.png').convert('RGB')
    st.image(image, caption='')
    st.write(
        '''
Observations:
- ARIMAX performed best for prediction intervals of 1 week and 12 weeks ahead.
- Prophet performed best for prediction interval of 4 weeks ahead.
        '''
    )
    st.write('')

    st.subheader('Table Comparing Prediction Graphs of ARIMAX and Prophet')
    image = Image.open('figures/results2.png').convert('RGB')
    st.image(image, caption='')
    st.write(
        '''
Observations:
- ARIMAX tends to predict more closely to the granularity of actual cases.
- ARIMAX tends to predict outbreaks quite late, increasing in inaccuracy as the prediction interval increases.
- Prophet is able to closely capture the outbreak spikes, across all prediction intervals.
        '''
    )
    st.write('')


def candr():
    caching.clear_cache()
    st.write('')
    st.header('Conclusions and Recommendations')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Conclusions:')
    st.markdown('- ARIMAX is a more accurate time series model in terms of rmse as a metric (1 week and 12 weeks ahead).')
    st.markdown('- Despite this, Prophet predicts more accurately 4 weeks ahead.')
    st.markdown('- Prophet is also able to more accurately predict a large outbreak for longer forecasting intervals.')

    st.write('')

    st.subheader('Recommendations:')
    st.markdown('- Explore more meta models for dengue incidence prediction (LSTM - RNN).')
    st.markdown('- Explore using climate data as regressors for Facebook Prophet.')
    st.markdown('- Explore using more current data for both dengue cases and climate.')


def bonus():
    caching.clear_cache()
    st.write('')
    st.header('Neural Networks!!?!!')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Table Comparing the RMSE of Models + Neural Networks')
    image = Image.open('figures/bonus.png').convert('RGB')
    st.image(image, caption='')
    st.write(
        '''
Observations:
- Neural networks dominates in performance across all prediction intervals.
- Maybe neural networks are better suited for dengue incidence prediction tasks???

Note:
- These results were taken from my thesis project as an undergrad.

Credit to my undergrad thesis partner for these results and allowing me to use our data for this project - [Farshana Datukon](https://www.linkedin.com/in/fdat/)!!
        '''
    )
    st.write('')

def contributors():
    caching.clear_cache()
    st.write('')
    st.header('Contributors')
    st.write('-----------------------------------------------------------------------') 
    st.write('')

    st.subheader('Eric Vincent Magno')
    st.markdown('- Email: [ericvincentmagno@gmail.com](mailto:ericvincentmagno@gmail.com)')
    st.markdown('- LinkedIn: [https://www.linkedin.com/in/ericxmagno/](https://www.linkedin.com/in/ericxmagno/)')

    st.subheader('JC Albert Peralta - Mentor')
    st.markdown('- Email: jcacperalta@gmail.com')
    st.markdown('- LinkedIn: [https://www.linkedin.com/in/jcacperalta/](https://www.linkedin.com/in/jcacperalta/)')