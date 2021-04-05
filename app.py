import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from streamlit import caching
from streamlit_folium import folium_static

import app_body as body

warnings.filterwarnings('ignore')

st.set_page_config(page_title='Forecasting Dengue Incidence in Davao City')

st.title('Forecasting Dengue Incidence in Davao City using Climate Data and Time Series Analysis')

# ## Side Bar Information
# image = Image.open('eskwelabs.png')
# st.sidebar.image(image, caption='', use_column_width=True)
# st.sidebar.markdown("<h1 style='text-align: center;margin-bottom:50px'>DS Cohort VI</h1>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center;margin-bottom:50px'>Categories</h1>", unsafe_allow_html=True)
## Create Select Box and options
add_selectbox = st.sidebar.radio(
    "",
    ("Objective", "Cleaned Data", "EDA - Dengue Cases by District", "EDA - Features", "Modeling - ARIMA", "Modeling - ARIMAX", "Modeling - Prophet", "Results", "Conclusions and Recommendations", "~BONUS~", "Contributors")
)


if add_selectbox == 'Objective':
    body.objectives()

elif add_selectbox == 'Cleaned Data':
    body.cleaned_data()

elif add_selectbox == 'EDA - Dengue Cases by District':
    body.district_eda()

elif add_selectbox == 'EDA - Features':
    body.eda_features()

elif add_selectbox == 'Modeling - ARIMA':
    body.arima()

elif add_selectbox == 'Modeling - ARIMAX':
    body.arimax()

elif add_selectbox == 'Modeling - Prophet':
    body.prophet()

elif add_selectbox == 'Results':
    body.results()

elif add_selectbox == 'Conclusions and Recommendations':
    body.candr()

elif add_selectbox == '~BONUS~':
    body.bonus()

elif add_selectbox == 'Contributors':
    body.contributors()