import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from prophet import Prophet
import base64
import requests
from matplotlib import pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly.offline import plot
import seaborn as sns
import plotly.express as px
import matplotlib.ticker as ticker
from plotly.offline import plot

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
    "Explorative Data Analysis [EDA]",
    "Passenger Forecast",
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

    if page_selection == 'Explorative Data Analysis [EDA]':

        with header:
            st.title('EXPLORATIVE DATA ANALSIS [EDA]')

            st.markdown('**The Project**')

            st.write("The London Underground Railway Network (The Tube) is a rapid transit system serving "
             + "Greater London and some parts of the adjacent counties of Buckinghamshire, " +
                "Essex and Hertfordshire in England. The Underground has its origins in the Metropolitan Railway, " +
                "the world's first underground passenger railway")

            st.write('The data for the London Underground is a time series data collected over a period of 4 years ' +
                "that shows the passenger counts (Inflow and Outflows) for all stations in the network in different days "
                + "of the week")

            st.write("The days as captured in the data with their notations are:")
            st.markdown("- Mondays to Thursdays (MTT)")
            st.markdown("- Fridays (FRI)")
            st.markdown("- Saturdays (SAT)")
            st.markdown("- Sundays (SUN)")

            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
            </style>
            ''', unsafe_allow_html=True)

            st.write("This Exploratory Data Analysis (EDA) brings key insights from the London Underground data " +
                "by using visual plots to explore the historic data and tell few stories before we try to make " + 
                "some forecast into the future passengercounts")

            data = pd.read_csv('resources/data/tube_time_interval_data_sorted.csv')

            st.markdown('******Exploring the data******')
            st.write('Clicking the ***view top*** button bellow will display the first 5 rows of The Tube dataset, ' +
                'and also the ***view bottom*** shows the last 5 rows')
            st.write('Preview of all station dataset (first 5 records)')
            if st.button('view top'):
                st.write(data.sort_values('entry_date_time').head())

            st.write('Preview of all station data (bottom 5)')
            if st.button('view bottom'):
                st.write(data.sort_values('entry_date_time').tail())

        with dataset:
            st.write("Our main analysis targets the London Underground stations with the highest number "+
                "of passenger counts, so the table below shows those station queried from the general dataset")
            
            df = data[['entry_date_time', 'time', 'station', 'year_of_entry', 'day', 'dir', 'counts' ]]
            df_top = df.loc[df.counts > 3000]          
            df_top = df_top.sort_values('counts', ascending=False)
            st.write(df_top)

            st.write('From the result shown in the table above, we can see that the ***Bank and Monument*** '+
                'and ***Waterloo LU*** stations are 2 most busiest ,and the trend is spread across different years'+
                ' and interestingly they both lie in the same ***zone 1*** '+ "of The Tube map. This shows a "+ 
                "potential heavy weight on the associated lines. We will explore these stations.")

            st.write('The following list of stations are those captured in the dataset above with passenger counts ' +
                'greater than 3000 for the corresponding years in the daily 15 minutes time interval.' + 
                ' These stations will be our main focus of interest in this exploration and also in the passenger counts forecasting '+
                'As understanding the factors contributing to the busy passenger flow (IN and OUT) of the station will '+
                'help us make good recommendations to TFL on improving the London Underground (The Tube) network lines')
            top_station = df_top['station'].unique()
            st.write(top_station)

            st.write("Using an interactive ***Bar Chart***, we can visualize these stations and look at the ***Time*** associated "+
                "with the high traffic per stations. With the station legends ***(Names)*** on the right of the chart, we can "+
                "select the stations we want to view by dehighlighting other station. This presents a bolder non-clustered chart")

            df_bar = df_top.head(500)
            # df_bar = df_bar.set_index('time')
            fig = px.bar(df_bar, x='time', y = 'counts', color='station')
            st.plotly_chart(fig)

            st.write("The Bar Chart above provided the right insight on the ***Time*** and ***Stations*** with highest passenger "+
                "traffic. But to view the year that produced the top 30 recorded passenger counts within the period of 4 years, "+
                "we will use a static Bar Chart this time. The plot below presents that data with complete date and time on X-axis. "+
                "and Passenger counts on Y-axis")
            from plotly.offline import plot
            ax = sns.barplot(x = 'entry_date_time', y = 'counts', data = df_top.sort_values('counts', ascending=False).head(30),  
                hue='station', palette='twilight_shifted', lw=3)
            plt.xticks(rotation=45)
            st.write(ax.get_figure())

            with dataset:
                st.write('***TIME SERIES VISUALIZATION***')
                st.markdown('<p style="font-family:Courier; color:Blue; font-size: 20px;">TIME SERIES VISUALIZATION</p>', unsafe_allow_html=True)
                # Building the function to help slice the required data for visualization
                def get_data_top_stations(df, year, day, dire):
                
                    df = df
                    day = day
                    dire = dire
                    year = year
                    
                    data_in_sorted = df.sort_values(['asc', 'time'], ascending=[True, True])
                    df_in_new = data_in_sorted.loc[data_in_sorted.day==day]
                    df_in_new = df_in_new.loc[df_in_new.dir==dire]
                    df_in_new_year = df_in_new.loc[df_in_new.year_of_entry==year]
                    
                    df_in_new_year = df_in_new_year[['entry_date_time', 'station', 'time', 'counts']]
                
                    return df_in_new_year

                st.write('Lets visualize the entire dataset to ')

                data_top = pd.read_csv('resources/data/tube_time_interval_data_sorted.csv')
                data_top['counts'] = data['counts'].round(decimals=0)

                list_station = ['Bank and Monument', 'Waterloo LU', 'Oxford Circus',
                'Canary Wharf LU', 'Liverpool Street LU', 'Moorgate','London Bridge LU', 
                'Farringdon', 'Victoria LU', 'Green Park']
                df_top_station = data_top.loc[data_top['station'].isin(list_station)]


                col1, col2, col3 = st.columns(3)
                col1 = st.selectbox('year',(2018, 2019, 2020, 2021))
                year = col1
                col2 = st.selectbox('select day',('MTT', 'FRI', 'SAT', 'SUN'))
                day = col2
                col3 = st.selectbox('direction',('IN', 'OUT'))
                dire=col3

                df = get_data_top_stations(df_top_station, year, day, dire)
    
                sns.set_style('darkgrid')
                sns.set(rc={'figure.figsize':(14,8)})

                fig2 = px.line(df.sort_values('entry_date_time', ascending=True), x='time', y = 'counts', color='station')
                st.plotly_chart(fig2)    

    if page_selection == "Passenger Forecast":
        with header:
            st.title('Forecasting The Tube Passenger Count')

        def get_data_year(df, station, day, dire, year):
        
            df = df
            station = station
            day = day
            dire = dire
            year = year
            
            data_in_sorted = df.sort_values(['asc', 'time'], ascending=[True, True])
            df_in_new = data_in_sorted.loc[data_in_sorted.station==station]
            df_in_new = df_in_new.loc[df_in_new.day==day]
            df_in_new = df_in_new.loc[df_in_new.dir==dire]
            df_in_new_year = df_in_new.loc[df_in_new.year_of_entry==year]
            
            df_in_new_year = df_in_new_year[['entry_date_time', 'counts']]
        
            return df_in_new_year

        
        with dataset:
            station_option = st.selectbox('select station',
                ('Bank and Monument', 'Waterloo LU', 'Oxford Circus','Canary Wharf LU', 
                    'Liverpool Street LU', 'Moorgate','London Bridge LU', 'Farringdon', 
                    'Victoria LU', 'Green Park',"King's Cross St. Pancras", 'Holborn', 
                    'Brixton LU', 'Stratford','Finsbury Park'))
            station=station_option

            day_option = st.selectbox('select day',
                ('MTT', 'FRI', 'SAT', 'SUN'))
            day=day_option

            dir_option = st.selectbox('direction',
                ('IN', 'OUT'))
            dire=dir_option

            year_option = st.selectbox('year',
                (2018, 2019, 2020, 2021))
            year=year_option        

            data = pd.read_csv('resources/data/tube_time_interval_data_sorted.csv')
            data['counts'] = data['counts'].round(decimals=0)

            df = get_data_year(data, station, day, dire, year)
            st.header('Passenger Forecast By Station')
            st.write(station + ' ' + 'station'+ ' '+ str(year) + ' ' + 'data view')
            st.write(df.head())
            st.header('Visualize Historic Data for Selected Station and Day')
            if st.button('Visualize'):
                st.line_chart(df.rename(columns={'entry_date_time':'index'}).set_index('index'))
        
            st.header('Passenger Forecasting for Selected Day')
            df.columns = ['ds', 'y']
            m = Prophet(interval_width=0.95, daily_seasonality=True)
            model = m.fit(df)
            future = m.make_future_dataframe(periods=96,freq='15T')
            forecast = m.predict(future)
            
            if st.button('Make Forecast'):
                fig = plot_plotly(model, forecast)
                fig
                # plot1 = m.plot(forecast)
                # st.write(plot1)
            
            if st.button('Explore Componets of The Forecast'): 
                plot2 = m.plot_components(forecast)
                st.write(plot2)
 

    if page_selection == 'Train Simulation':
        st.title('TRAIN SIMULATION')

    if page_selection == 'About Team':
        st.title('ABOUT TEAM')


if __name__ == '__main__':
    main()
