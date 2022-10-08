import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from prophet import Prophet
import base64
import requests

header = st.container()
dataset = st.container()


# Functions
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


def get_api_records(api_name):
    # extracting Line Data directly from API
    url = api_name

    # Access credentials
    app_id = 'EXPLORE_TUBE_TWIN_PROJECT'
    app_key = 'f8f6ab8258464b1ca5ddd33c0ff2ee12'
    api_token = {"app_id": app_id, "app_key": app_key}

    # Stating file format to use
    headers = {"Accept": "application/json"}

    # Initializing get request
    r = requests.get(url, api_token, headers=headers)

    # Assigning file to Json() format
    file_name = r.json()

    # Displaying fetched records
    return file_name


# Lists
page_options = [
    "Landing Page",
    "Live Feed",
    "Modelling",
    "Passenger Forecast",
    "Explorative Data Analysis [EDA]",
    "Train Simulation",
    "About Team"
]
api_options = [
    "Air Quality",
    "Line"
]

# Dictionaries
api_dict = {
    "AccidentStats": "https://api.tfl.gov.uk/AccidentStats/{year}",
    "AirQuality": "https://api.tfl.gov.uk/AirQuality/",
    "BikePoint": "https://api.tfl.gov.uk/BikePoint/",
    "Journey": "https://api.tfl.gov.uk/Journey",
    "Line": "https://api.tfl.gov.uk/Line/Route?serviceTypes=Regular",
    "Mode": "https://api.tfl.gov.uk/Mode/{mode}/Arrivals[?count]",
    "Occupancy": "https://api.tfl.gov.uk/Occupancy",
    "Road": "https://api.tfl.gov.uk/Road/{ids}/Disruption[?stripContent][&severities][&categories][&closures]",
    "Search": "https://api.tfl.gov.uk/Search/Meta/Categories",
    "Station": "https://api.tfl.gov.uk/stationdata/tfl-stationdata-detailed.zip",
    "StopPoint": "https://api.tfl.gov.uk/StopPoint"
}


def main():
    st.sidebar.image('resources/images/LU_logo.jpg', use_column_width=True)
    page_selection = st.sidebar.selectbox("PAGE SELECTION", page_options)

    if page_selection == "Landing Page":
        add_bg_from_local('resources/images/Landing_Page_Background.jpg')

    if page_selection == "Live Feed":
        st.title('LIVE [TFL - API] FEED')
        api_selection = st.selectbox("API SELECTION", api_options)

        if api_selection == 'Air Quality':
            if st.button('Update'):
                st.subheader('URL: ' + api_dict['AirQuality'])
                df = get_api_records(api_dict['AirQuality'])
                st.dataframe(df)

        if api_selection == 'Line':
            if st.button('Update'):
                st.subheader('URL: ' + api_dict['Line'])
                df1 = get_api_records(api_dict['Line'])
                st.dataframe(df1)

    if page_selection == "Passenger Forecast":
        with header:
            st.title('Explore The Tube Passenger Forecast')

        df = pd.read_csv('resources/data/df_in_2021_mtt.csv')
        with dataset:
            st.header('Passenger Forecst By Station')
            st.write(df.head())
            st.header('Visualize Historic Data for Selected Station and Day')
            if st.button('Visualize'):
                st.line_chart(df.rename(columns={'entry_date_time':'index'}).set_index('index'))
        
        with dataset: 
            st.header('Passenger Forecasting for Selected Day')
            df.columns = ['ds', 'y']
            m = Prophet(interval_width=0.95, daily_seasonality=True)
            model = m.fit(df)
            future = m.make_future_dataframe(periods=96,freq='15T')
            forecast = m.predict(future)
            
            if st.button('Make Forecast'):
                plot1 = m.plot(forecast)
                st.write(plot1)
            
            if st.button('Explore Componets of The Forecast'): 
                plot2 = m.plot_components(forecast)
                st.write(plot2)



    if page_selection == 'Modelling':
        st.title('PREDICTIVE MODELLING')

    if page_selection == 'Explorative Data Analysis [EDA]':
        st.title('EXPLORATIVE DATA ANALSIS [EDA]')

    if page_selection == 'Train Simulation':
        st.title('TRAIN SIMULATION')

    if page_selection == 'About Team':
        st.title('ABOUT TEAM')


if __name__ == '__main__':
    main()
