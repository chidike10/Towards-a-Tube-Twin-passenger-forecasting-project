# Importing libraries
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from prophet import Prophet
import base64
import requests
from sklearn import preprocessing
from matplotlib import pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly.offline import plot
import seaborn as sns
import plotly.express as px
import matplotlib.ticker as ticker
from plotly.offline import plot
import networkx as nx
import colorsys
from bokeh.plotting import figure
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge, ElasticNet

header = st.container()
dataset = st.container()


# Load pickled models
lr = pickle._load(open('resources/models/lr_model.sav', 'rb'))
dtr = pickle._load(open('resources/models/dtr_model.sav', 'rb'))
knn = pickle._load(open('resources/models/knn_model.sav', 'rb'))
rfr = pickle._load(open('resources/models/rfr_model.sav', 'rb'))
w_br = pickle._load(open('resources/models/w_br_model.sav', 'rb'))
enr = pickle._load(open('resources/models/enr_model.sav', 'rb'))



# Global Constants
columns3A = ['nlc', 'Station Name', 'Time Period', 'Total', 'Male', 'Female', 'Sex: Not Stated', 'Age: Not Stated', 'Under 16', '16-19',
           '20-24', '25-34', '35-44', '45-59', '60-64', '65-70', 'Over 70', 'MIP Not Given', 'Mobility Impariment', 'Hearing Impairment',
           'Mental Health Condition', 'None', 'Visual Impairment', 'Learning Dissability', 'Serious Long Term Illness', 'Other']
columns4 = ['nlc', 'Station Name', 'Time Period', 'All Modes', 'Home to Work', 'Work to Home', 'Origin Purpose: Home', 'Work',
           'Shop', 'Education', 'Tourist', 'Hotel', 'Other', 'Unknown/Not Given', 'Destination Purpose: Home', 'Work', 'Shop',
           'Education', 'Tourist', 'Hotel', 'Other', 'Unkown/Not Given']
columns3B = ['nlc', 'Station Name', 'Time Period', 'Total', 'Male', 'Female', 'Sex: Not Stated', 'Age: Not Stated', 'Under 16',
            '16-19', '20-24','25-34','35-44','45-59','60-64','65-70','Over 70']
st.set_option('deprecation.showPyplotGlobalUse', False)



# Reading Files
EntryExit_2021 = pd.read_csv('resources/data/EntryExit_2021.csv')

# Graph Network Files
lu_stations = pd.read_csv('resources/data/Stations_Coodinates.csv', index_col=0)
connections = pd.read_csv('resources/data/LU_Loading_Data.csv')
lu_lines = pd.read_csv('resources/data/Lines.csv', index_col=0)

# 2017 NUMBAT RODS Data
# Entry Data
NBT_entry_agm = pd.read_excel('resources/data/Age, gender, mobility category by entry station-zone-time of day 2017.xlsx', names = columns3A, sheet_name = 'agesex', header= 4)
NBT_entry_ajt = pd.read_excel('resources/data/Average journey time by entry station-zone-time of day 2017.xlsx', sheet_name = 'station', header= 4)
NBT_entry_dt = pd.read_excel('resources/data/Distance travelled by entry station-zone-line-purpose-time of day-ticket type 2017.xlsx', sheet_name = 'By station', header= 4)
NBT_entry_jp = pd.read_excel('resources/data/Journey purpose by entry station-zone-time of day-ticiket type 2017.xlsx', names = columns4, sheet_name = 'journey purpose', header= 5)
NBT_entry_tt = pd.read_excel('resources/data/Ticket type by entry station-zone-time of day 2017.xlsx', sheet_name = 'ticket_type', header= 4)
# Exit Data
NBT_exit_agm = pd.read_excel('resources/data/Age, gender, mobility category by exit station-zone-time of day 2017.xlsx', names = columns3B, sheet_name = 'age&sex', header= 4)
NBT_exit_ajt = pd.read_excel('resources/data/Average journey time by exit station-zone-time of day 2017.xlsx', sheet_name = 'station', header= 4)
NBT_exit_dt = pd.read_excel('resources/data/Distance travelled by exit station-zone-line-purpose-time of day-ticket type 2017.xlsx', sheet_name = 'by ticket type', header= 4)
NBT_exit_egress = pd.read_excel('resources/data/Egress mode by exit station-zone-time of day 2017.xlsx', sheet_name = 'Egress', header= 4)
NBT_exit_jp = pd.read_excel('resources/data/Journey purpose by exit station-zone-time of day-ticket type 2017.xlsx', names = columns4,  sheet_name = 'journey purpose', header= 5)
NBT_exit_tt = pd.read_excel('resources/data/Ticket type by exit station-zone-time of day 2017.xlsx', sheet_name = 'ticket_type', header= 4)



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


# This function performs the train-test split for the pre-processed Annualized Entry Exit [2017-20121] datasets
def Xy_split(df):
    x_df = df.drop(columns = list(df)[11:107])
    x_df = x_df.drop(columns = ['Station', 'Mode'])
    y_df = df[list(df)[11:107]]
    return x_df, y_df

X,y = Xy_split(EntryExit_2021)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Lists
page_options = [
    "Landing Page",
    "Explorative Data Analysis [EDA]",
    "Passenger Forecast Modeling",
    "Graphical Network Representation",
    "About The Team"
]


api_options = [
    "Air Quality",
    "Line"
]


eda_options = [
    "General EDA",
    "Time-Series Based EDA"
]


eda_ds_options = [
'Passenger Station Entry Counts: [Age, Gender & Mobility] Dataset',
'Passenger Station Entry Counts: [Average Journey Time] Dataset',
'Passenger Station Entry Counts: [Distance Travelled] Dataset',
'Passenger Station Entry Counts: [Journey Purpose] Dataset',
'Passenger Station Entry Counts: [Ticket Type] Dataset',
'Passenger Station Exit Counts: [Age, Gender & Mobility] Dataset',
'Passenger Station Exit Counts: [Average Journey Time] Dataset',
'Passenger Station Exit Counts: [Distance Travelled] Dataset',
'Passenger Station Exit Counts: [Egress] Dataset',
'Passenger Station Exit Counts: [Journey Purpose] Dataset',
'Passenger Station Exit Counts: [Ticket Type] Dataset'
]


model_options = [
    "Raw Dataset Overview",
    "Bayesian Ridge Regressor",
    "Elastic Net Regressor",
    "Random Forest Regressor",
    "K-Nearest Neighbors Regressor",
    "Linear Regressor",
    "Decision Tree Regressor",
    "Time-Series Model"
]


dataset_options = [
    "Passenger Station Entry/Exit Counts Year: 2021",
]


col_list = ['0500-0515',
            '0515-0530',
            '0530-0545',
            '0545-0600',
            '0600-0615',
            '0615-0630',
            '0630-0645',
            '0645-0700',
            '0700-0715',
            '0715-0730',
            '0730-0745',
            '0745-0800',
            '0800-0815',
            '0815-0830',
            '0830-0845',
            '0845-0900',
            '0900-0915',
            '0915-0930',
            '0930-0945',
            '0945-1000',
            '1000-1015',
            '1015-1030',
            '1030-1045',
            '1045-1100',
            '1100-1115',
            '1115-1130',
            '1130-1145',
            '1145-1200',
            '1200-1215',
            '1215-1230',
            '1230-1245',
            '1245-1300',
            '1300-1315',
            '1315-1330',
            '1330-1345',
            '1345-1400',
            '1400-1415',
            '1415-1430',
            '1430-1445',
            '1445-1500',
            '1500-1515',
            '1515-1530',
            '1530-1545',
            '1545-1600',
            '1600-1615',
            '1615-1630',
            '1630-1645',
            '1645-1700',
            '1700-1715',
            '1715-1730',
            '1730-1745',
            '1745-1800',
            '1800-1815',
            '1815-1830',
            '1830-1845',
            '1845-1900',
            '1900-1915',
            '1915-1930',
            '1930-1945',
            '1945-2000',
            '2000-2015',
            '2015-2030',
            '2030-2045',
            '2045-2100',
            '2100-2115',
            '2115-2130',
            '2130-2145',
            '2145-2200',
            '2200-2215',
            '2215-2230',
            '2230-2245',
            '2245-2300',
            '2300-2315',
            '2315-2330',
            '2330-2345',
            '2345-0000',
            '0000-0015',
            '0015-0030',
            '0030-0045',
            '0045-0100',
            '0100-0115',
            '0115-0130',
            '0130-0145',
            '0145-0200',
            '0200-0215',
            '0215-0230',
            '0230-0245',
            '0245-0300',
            '0300-0315',
            '0315-0330',
            '0330-0345',
            '0345-0400',
            '0400-0415',
            '0415-0430',
            '0430-0445',
            '0445-0500']



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
        if st.button('About Project'):
            add_bg_from_local('resources/images/bkg dark.jpg')
            with header:
                st.title('The Tube Twin Project')
                st.image('resources/images/the tube.jpeg', use_column_width='always', output_format='jpeg')
                st.subheader('Summary')
                st.write("With around 400km of rails, over 267 stations, and more than 1.3 billion " +
                         "passenger journeys each year (according to the project data), quickly andsafely " +
                         "moving passengers through stations and onto trains is an ongoing priority for " +
                         "the London Underground. Gaining deeper insights into passengercounts and traffic "
                         "at the different stations will help Transport for London (TfL)make better "
                         "decisions and plan for future operations.")
                st.write("This project's objective is to provide data solutions that provides decision "
                         "making support to the management of the Tube in delivering a more efficient and "
                         "reliable underground rail transport system. This system will provide a graphical "
                         "network, a forecasting model for passenger count, and tube analyses for "
                         "recommendations on operations and traffic flow.")
                st.write("The analysis is based on historical data collected from TfL's open data API, "
                         "which shows records of the number of passengers inflow and outflow per station "
                         "in daily 15 minutes time interval. This project produced a web-based data application"
                         " developed with ***Streamlit*** and deployed on ***AWS Cloud*** to integrated the "
                         " the digital solutions to the problems discussed below.")

                st.subheader('Introduction & Problem Statement')
                st.write(" Transport for London (TfL) is a local government body responsible for most of "
                         "London's transport network, including the London Underground (LU, aka The Tube). "
                         "The tube is a public railway transit system that serves Greater London and its nearby"
                         "counties.")
                st.write("London has a population of approximately 9 million people, on top of "
                         "hundreds and thousands of tourists visit it every day. This presents some challenges "
                         "to TfL concerning operations of the tube railway network which includes the following:")
                st.markdown("- Network operations and surveillance.")
                st.markdown("- Traffic flow, analyses, and control.")
                st.markdown("- Infrastructure management and maintenance.")

                st.subheader("Solution")
                st.write("As part of this project and with the aim of tackling the problems mentioned above the "
                         "***Explore AI Tube Twin team 6*** designed and created a digital system to address each of "
                         "them. These include the following:")
                st.markdown("1. **A Graphical Network of the Tube**")
                st.write("From the data gotten from Tfl, we generated a graph representation of the tube "
                         "network using NetworkX. From this graph, we were able to make a few analyses like "
                         "determining the important and busiest stations within the network. The image below "
                         "shows the network stations connectivity with the most important stations shown in "
                         "larger circles. This importance is determined by some station (called the node) features "
                         "like the number of lines connecting on stations (edges).")
                st.image('resources/images/bokeh_plot_new.png', width=600)
                st.info('***fig1 - NetworkX Image representation Tube Twin***')

                st.markdown("2. **A Simulation of the Tube Network**")
                st.write("We created a simulation of the tube network using SUMO. For this, we selected a "
                         "few stations based on their importance and usage ranking.")

                st.markdown("3. **A Model to Forecast Passenger Count**")
                st.write("We created a time-series model to forecast passenger count based on stations, "
                         "year, and day. We achieved this by making use of the Python library, FBProphet model in"
                         "in a process called ***Transfer Learning***. This will allow Tfl to predict future "
                         "traffic and plan ahead of time.")
                st.write("The image below shows the the pictorial representation of the ***Moorgate Statio*** in "
                         "an outgoing direction on saturdays. ")
                st.image('resources/images/forecast.jpg', width=600)
                st.info('***fig2 - Moorgate station passengers forecast***')
                st.write("In the forecast plot above, the dots in the graph represents the actual data points, while "
                         "the line represents the model predictions. The shaded region which the line runs through "
                         "is the ***boundary*** of the predictions given as lower and upper bounds. The entire region "
                         "from where the thick dots ends represent the forecasted region which is same as the peroid"
                         "of the station data (passenger counts in all 15 minutes time interval on saturdays)"
                         "")

                st.subheader("Conclusion and Recommendation")
                st.write("The Tube Twin analysis and and passenger forecasting conducted in this project "
                         "provides the solution to the traffic issues in the heavily-weighted Tube station lines. "
                         "The deployed web application adequately covered the top 14 stations for passenger"
                         "forecasting, while the general time series data analysis and and the graphical network"
                         "representation provided an overall analysis. The project outcome gives a reliable "
                         "prototype system that, if deployed and interatively developed, can support TFL in "
                         " achieving many of it's Tube management objectives which includes traffic control.")

                st.markdown("- " "The Tube Twin project by Explore AI (2022) team 6 provides a good ground for "
                            "further research and improvement of passenger flow and traffic analyses on the London Tube. "
                            "We recommend expansion in the scope of passenger forecasting (i.e more stations) "
                            "taking into account certain social activiies and enviromental factors "
                            "(e.g social events and weather).")
                st.markdown("- " "The next research on the project should consider a live streaming data collection "
                            "of the tube for a minimum period of one year as this will allow for collection of "
                            "comprehensive realtime data for higher forcasting accuracy. We also recommend ")

                st.markdown("- " "This project and report are limited in scope based on the limited time to deliver"
                            " a more robust system, as such, this report does not provide a conclusive and thorough "
                            "explanation of passenger flow on the Tube, neither did it provide all intended solutions. "
                            "This is because available data of the Tube accessible by the team for this research is "
                            "not sufficient enough for more robust Tube forecasting solution or problem that may arise. ")
                st.markdown("- " "Lastly, to be able to achieve interactive passenger forecasting system that not "
                            "only serve the management of the Tube but also the passengers, The Transport for London (TFL) "
                            "can consider integrating ***Live Weather Conditions*** at the point of data generation "
                            "to enable the analysis and forecating of passenger counts given a certain weather condition ")

                st.subheader("System Design and Development")
                st.write("All applications were developed using python programming language and several open software "
                         "packages like NetworkX, FBProphet, SUMO, and Streamlit. The diagram below depicts the flow of "
                         "processes of the project applications from start to finish. These include the use of AWS "
                         "resources (EC2 and S3) for computing and storage resources.")
                st.image('resources/images/TubeTwin App Process Flow.jpg', width=600)
                st.write("Looking at the diagram above, we can affirm that every data project after underground study "
                         "kickstarts with actual data collection from the data source. This ***Tube Twin*** project started "
                         "technically with the background study and data collection and engineering. After achieving "
                         "shaped into the required formats, they are then saved into the bucket on AWS where our applications "
                         "***clean datasets*** runs. All application codes are also stored in the S3 bucket attached to "
                         "the EC2 instance responsible for run the app")

                st.markdown('<p style="font-family:Calibri; color:Black; font-size: 36px;">References</p>',
                            unsafe_allow_html=True)

                st.markdown("- Li-Yang (Edward) Chiang, Robert Crockett, Ian Johnson, Aidan O’Keefe. "
                            "(2017). Passenger Flow in the Tube.")
                st.markdown("- Taylor SJ, Letham B. 2017. Forecasting at scale. PeerJ Preprints 5:e3190v2. "
                            "https://doi.org/10.7287/peerj.preprints.3190v2")
                st.markdown("- Transport for London. (n.d.). Open data policy. Retrieved April 09, 2017, "
                            "from https://tfl.gov.uk/info-for/open-data-users/open-data-policy")

                link = '[GitHub](http://github.com)'
                st.markdown(link, unsafe_allow_html=True)



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
        st.title('EXPLORATIVE DATA ANALSIS [EDA]')
        eda_selection = st.selectbox("EDA Selection", eda_options)
        if eda_selection == 'General EDA':
            eda_ds_selection = st.selectbox('EDA Dataset Selection', eda_ds_options)
            if eda_ds_selection == 'Passenger Station Entry Counts: [Age, Gender & Mobility] Dataset':
                df = NBT_entry_agm.groupby('Station Name')['Total', 'Male', 'Female', 'Sex: Not Stated', 'Age: Not Stated',
                                                           'Under 16', '16-19', '20-24', '25-34', '35-44', '45-59', '60-64',
                                                           '65-70', 'Over 70', 'MIP Not Given', 'Mobility Impariment',
                                                           'Hearing Impairment', 'Mental Health Condition',
                                                           'None', 'Visual Impairment', 'Learning Dissability',
                                                           'Serious Long Term Illness', 'Other'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by=['Total'], ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                agm_columns = {

                    'col_gender': ['Male',
                                   'Female',
                                   'Sex: Not Stated',
                                   'Age: Not Stated'],

                    'col_age': ['Under 16',
                                '16-19',
                                '20-24',
                                '25-34',
                                '35-44',
                                '45-59',
                                '60-64',
                                '65-70',
                                'Over 70'],

                    'col_impairment': ['MIP Not Given',
                                       'Mobility Impariment',
                                       'Hearing Impairment',
                                       'Mental Health Condition',
                                       'None',
                                       'Visual Impairment',
                                       'Learning Dissability',
                                       'Serious Long Term Illness',
                                       'Other']
                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y=df.index, x=agm_columns['col_gender'], title='GENDER ANALYSIS GRAPH: [GENDER SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y=df.index, x=agm_columns['col_age'], title='AGE ANALYSIS GRAPH: [AGE SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig3 = px.bar(df, y=df.index, x=agm_columns['col_impairment'],
                                 title='MOBILITY ANALYSIS GRAPH: [IMPAIRMENT SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.plotly_chart(fig3, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Entry Counts: [Average Journey Time] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_entry_ajt.groupby('Station Name')['Time period', 'Total time',
                                                           'Average', '< 15 mins', '15 - 30', '30 - 45',
                                                           '45 - 60', '60 - 90', 'over 90'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by=['Total time'], ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                ajt_columns = {

                    'col_summary_time': [
                        'Total time',
                        'Average'
                    ],

                    'col_time': ['< 15 mins',
                                 '15 - 30',
                                 '30 - 45',
                                 '45 - 60',
                                 '60 - 90',
                                 'over 90'],

                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y=df.index, x=ajt_columns['col_summary_time'],
                                 title='AVERAGE JOURNEY TIME ANALYSIS GRAPH: [SUMMARY SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = ajt_columns['col_time'], title = 'AVERAGE JOURNEY TIME ANALYSIS GRAPH: [TIME SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Entry Counts: [Distance Travelled] Dataset':
                df = NBT_entry_dt.groupby('Station')[
                    'Total', 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, '30+ kms'
                ].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by=['Total'], ascending=False)

                df = df.drop(['Total LUL'], axis=0)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                dt_columns = {

                    'col_near': [2, 3, 4, 5, 6, 7, 8, 9, 10],

                    'col_intermediate': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],

                    'col_far': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, '30+ kms']

                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index , x = dt_columns['col_near'], title = 'DISTANCE TRAVELLED ANALYSIS GRAPH: [DISTANCE-NEAR SUBSET ]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index , x = dt_columns['col_intermediate'], title = 'DISTANCE TRAVELLED ANALYSIS GRAPH: [DISTANCE-INTERMEDIATE SUBSET ]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig3 = px.bar(df, y = df.index , x = dt_columns['col_far'], title = 'DISTANCE TRAVELLED ANALYSIS GRAPH: [DISTANCE-FAR SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.plotly_chart(fig3, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Entry Counts: [Journey Purpose] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_entry_jp.groupby('Station Name')['All Modes',
                                                          'Home to Work',
                                                          'Work to Home',
                                                          'Origin Purpose: Home',
                                                          'Work',
                                                          'Shop',
                                                          'Education',
                                                          'Tourist',
                                                          'Hotel',
                                                          'Other',
                                                          'Unknown/Not Given',
                                                          'Destination Purpose: Home',
                                                          'Work.1',
                                                          'Shop.1',
                                                          'Education.1',
                                                          'Tourist.1',
                                                          'Hotel.1',
                                                          'Other.1',
                                                          'Unkown/Not Given'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by='All Modes', ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                jp_columns = {
                    'col_origin_home': ['Origin Purpose: Home',
                                        'Work',
                                        'Shop',
                                        'Education',
                                        'Tourist',
                                        'Hotel',
                                        'Other',
                                        'Unknown/Not Given'
                                        ],

                    'col_destination_home': ['Destination Purpose: Home',
                                             'Work.1',
                                             'Shop.1',
                                             'Education.1',
                                             'Tourist.1',
                                             'Hotel.1',
                                             'Other.1',
                                             'Unkown/Not Given'
                                             ],

                    'col_general': ['Home to Work',
                                    'Work to Home'
                                    ]
                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = jp_columns['col_origin_home'], title = 'JOURNEY PURPOSE ANALYSIS GRAPH: [ORIGIN PURPOSE SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = jp_columns['col_destination_home'], title = 'JOURNEY PURPOSE ANALYSIS GRAPH: [DESTINATION PURPOSE SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig3 = px.bar(df, y = df.index, x = jp_columns['col_general'], title = 'JOURNEY PURPOSE ANALYSIS GRAPH: [GENERAL PURPOSE SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.plotly_chart(fig3, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Entry Counts: [Ticket Type] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_entry_tt.groupby('Station name')['Daily (inc T/Card)  ',
                                                          'Weekly              ',
                                                          'Periods             ',
                                                          'All Permits         ',
                                                          'Not Stated /Other    ',
                                                          'Total all types'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by='Total all types', ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                tt_columns = {
                    'col_time': ['Daily (inc T/Card)  ',
                                 'Weekly              ',
                                 'Periods             ',
                                 ],

                    'col_permits': [
                        'All Permits         ',
                        'Not Stated /Other    ',
                        'Total all types'
                    ]
                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = tt_columns['col_time'], title = 'TICKET TYPE ANALYSIS GRAPH: [TIME SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = tt_columns['col_permits'], title = 'TICKET TYPE ANALYSIS GRAPH: [PERMITS SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Exit Counts: [Age, Gender & Mobility] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_exit_agm.groupby('Station Name')['Total', 'Male', 'Female', 'Sex: Not Stated', 'Age: Not Stated',
                                                          'Under 16', '16-19', '20-24', '25-34', '35-44', '45-59', '60-64',
                                                          '65-70', 'Over 70'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by=['Total'], ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                agm_columns = {

                    'col_gender': ['Male',
                                   'Female',
                                   'Sex: Not Stated',
                                   'Age: Not Stated'],

                    'col_age': ['Under 16',
                                '16-19',
                                '20-24',
                                '25-34',
                                '35-44',
                                '45-59',
                                '60-64',
                                '65-70',
                                'Over 70']

                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = agm_columns['col_gender'], title = 'GENDER ANALYSIS GRAPH: [GENDER SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = agm_columns['col_age'], title = 'AGE ANALYSIS GRAPH: [AGE SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Exit Counts: [Average Journey Time] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_exit_ajt.groupby('Station Name')['Time period', 'Total time',
                                                          'Average', '< 15 mins', '15 - 30', '30 - 45',
                                                          '45 - 60', '60 - 90', 'over 90'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by=['Total time'], ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                ajt_columns = {

                    'col_summary_time': [
                        'Total time',
                        'Average'
                    ],

                    'col_time': ['< 15 mins',
                                 '15 - 30',
                                 '30 - 45',
                                 '45 - 60',
                                 '60 - 90',
                                 'over 90'],

                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = ajt_columns['col_summary_time'], title = 'AVERAGE JOURNEY TIME ANALYSIS GRAPH: [TIME SUMMARY SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = ajt_columns['col_time'], title = 'AVERAGE JOURNEY TIME ANALYSIS GRAPH: [TIME SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)



            if eda_ds_selection == 'Passenger Station Exit Counts: [Distance Travelled] Dataset':
                NBT_exit_dt['Total passenger kilometres'] = pd.to_numeric(NBT_exit_dt['Total passenger kilometres'],
                                                                          errors='coerce').notnull()

                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_exit_dt.groupby('Station name')['Total passengers', 'Total passenger kilometres',
                                                         'Average journey length (kms)'].sum()

                # Group the dataset by a categorical feature and get the sum of ordinal features
                df_a = NBT_exit_dt.groupby('Ticket Type')['Total passengers', 'Total passenger kilometres',
                                                          'Average journey length (kms)'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df, df_a = df.sort_values(by=['Total passengers'], ascending=False), df_a.sort_values(
                    by=['Total passengers'], ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                dt_columns = {

                    'col_total_passenger': ['Total passengers'],
                    'col_total_passenger_km': ['Total passenger kilometres'],
                    'col_averaeg_journey_length': ['Average journey length (kms)']

                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = dt_columns['col_total_passenger'], title = 'DISTANCE TRAVELLED ANALYSIS GRAPH: [PASSENGER COUNT SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = dt_columns['col_total_passenger_km'], title = 'DISTANCE TRAVELLED ANALYSIS GRAPH: [PASSENGER DISTANCE SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig3 = px.bar(df, y = df.index, x = dt_columns['col_averaeg_journey_length'], title = 'DISTANCE TRAVELLED ANALYSIS GRAPH: [AVERAGE LENGTH SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.plotly_chart(fig3, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Exit Counts: [Egress] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_exit_egress.groupby(' station name')['NR/DLR/ Tram         ',
                                                              'Bus/ Coach           ',
                                                              'Bicycle             ',
                                                              'Motorcycle          ',
                                                              'Car/Van Parked      ',
                                                              'Car/Van driven away ',
                                                              'Walked              ',
                                                              'Taxi/ Minicab        ',
                                                              'RiverBus/ Ferry      ',
                                                              'Other               ',
                                                              'Not Stated          ',
                                                              '     Total all modes'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by='     Total all modes', ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                eg_columns = {
                    'col_vehicle': ['NR/DLR/ Tram         ',
                                    'Bus/ Coach           ',
                                    'Bicycle             ',
                                    'Motorcycle          ',
                                    'Car/Van Parked      ',
                                    'Car/Van driven away ',
                                    'Walked              ',
                                    'Taxi/ Minicab        ',
                                    'RiverBus/ Ferry      ',
                                    'Other               ',
                                    'Not Stated          '
                                    ],

                    'col_sum': [
                        '     Total all modes'
                    ]
                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = eg_columns['col_vehicle'], title = 'EGRESS ANALYSIS GRAPH: [VEHICLE SUBSET]')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = eg_columns['col_sum'], title = 'EGRESS ANALYSIS GRAPH: [TOTAL MODE SUBSET]')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Exit Counts: [Journey Purpose] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_exit_jp.groupby('Station Name')['All Modes',
                                                         'Home to Work',
                                                         'Work to Home',
                                                         'Origin Purpose: Home',
                                                         'Work',
                                                         'Shop',
                                                         'Education',
                                                         'Tourist',
                                                         'Hotel',
                                                         'Other',
                                                         'Unknown/Not Given',
                                                         'Destination Purpose: Home',
                                                         'Work.1',
                                                         'Shop.1',
                                                         'Education.1',
                                                         'Tourist.1',
                                                         'Hotel.1',
                                                         'Other.1',
                                                         'Unkown/Not Given'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by='All Modes', ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                jp_columns = {
                    'col_origin_home': ['Origin Purpose: Home',
                                        'Work',
                                        'Shop',
                                        'Education',
                                        'Tourist',
                                        'Hotel',
                                        'Other',
                                        'Unknown/Not Given'
                                        ],

                    'col_destination_home': ['Destination Purpose: Home',
                                             'Work.1',
                                             'Shop.1',
                                             'Education.1',
                                             'Tourist.1',
                                             'Hotel.1',
                                             'Other.1',
                                             'Unkown/Not Given'
                                             ],

                    'col_general': ['Home to Work',
                                    'Work to Home'
                                    ]
                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = jp_columns['col_origin_home'], title = 'JOURNEY PURPOSE ANALYSIS GRAPH')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = jp_columns['col_destination_home'],  title = 'JOURNEY PURPOSE ANALYSIS GRAPH')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig3 = px.bar(df, y = df.index, x = jp_columns['col_general'],  title = 'JOURNEY PURPOSE ANALYSIS GRAPH')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.plotly_chart(fig3, use_container_width=True)


            if eda_ds_selection == 'Passenger Station Exit Counts: [Ticket Type] Dataset':
                # Group the dataset by a categorical feature and get the sum of ordinal features
                df = NBT_exit_tt.groupby('Station name')['Daily (inc T/Card)  ',
                                                         'Weekly              ',
                                                         'Periods             ',
                                                         'All Permits         ',
                                                         'Not Stated /Other    ',
                                                         'Total all types'].sum()

                # Sort grouped dataframe values by an ordinal feature and display the top ten results
                df = df.sort_values(by='Total all types', ascending=False)

                df = df.head(10)

                # Divide dataset into subsets by creating a dictionary of features
                tt_columns = {
                    'col_time': ['Daily (inc T/Card)  ',
                                 'Weekly              ',
                                 'Periods             ',
                                 ],

                    'col_permits': [
                        'All Permits         ',
                        'Not Stated /Other    ',
                        'Total all types'
                    ]
                }
                if st.button("Generate EDA Graph"):
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig1 = px.bar(df, y = df.index, x = tt_columns['col_time'],  title = 'TICKET TYPE ANALYSIS GRAPH')
                    # Plots a plotly graph using the above created function df and feature dictionary
                    fig2 = px.bar(df, y = df.index, x = tt_columns['col_permits'], title = 'TICKET TYPE ANALYSIS GRAPH')
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)


        if eda_selection == 'Time-Series Based EDA':
            with header:
                st.title('EXPLORATIVE DATA ANALSIS [EDA]')

                st.markdown('**The Project**')

                st.write("The London Underground Railway Network (The Tube) is a rapid transit system serving "
                         + "Greater London and some parts of the adjacent counties of Buckinghamshire, " +
                         "Essex and Hertfordshire in England. The Underground has its origins in the Metropolitan Railway, " +
                         "the world's first underground passenger railway")

                st.write(
                    'The data for the London Underground is a time series data collected over a period of 4 years ' +
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
                import pandas as pd

                data = pd.read_csv('resources/data/tube_time_interval_data_sorted.csv')

                st.markdown('******Exploring the data******')
                st.write(
                    'Clicking the ***view top*** button bellow will display the first 5 rows of The Tube dataset, ' +
                    'and also the ***view bottom*** shows the last 5 rows')
                st.write('Preview of all station dataset (first 5 records)')
                if st.button('view top'):
                    st.write(data.sort_values('entry_date_time').head())

                st.write('Preview of all station data (bottom 5)')
                if st.button('view bottom'):
                    st.write(data.sort_values('entry_date_time').tail())

            with dataset:
                st.write("Our main analysis targets the London Underground stations with the highest number " +
                         "of passenger counts, so the table below shows those station queried from the general dataset")

                df = data[['entry_date_time', 'time', 'station', 'year_of_entry', 'day', 'dir', 'counts']]
                df_top = df.loc[df.counts > 3000]
                df_top = df_top.sort_values('counts', ascending=False)
                st.write(df_top)

                st.write('From the result shown in the table above, we can see that the ***Bank and Monument*** ' +
                         'and ***Waterloo LU*** stations are 2 most busiest ,and the trend is spread across different years' +
                         ' and interestingly they both lie in the same ***zone 1*** ' + "of The Tube map. This shows a " +
                         "potential heavy weight on the associated lines. We will explore these stations.")

                st.write(
                    'The following list of stations are those captured in the dataset above with passenger counts ' +
                    'greater than 3000 for the corresponding years in the daily 15 minutes time interval.' +
                    ' These stations will be our main focus of interest in this exploration and also in the passenger counts forecasting ' +
                    'As understanding the factors contributing to the busy passenger flow (IN and OUT) of the station will ' +
                    'help us make good recommendations to TFL on improving the London Underground (The Tube) network lines')
                top_station = df_top['station'].unique()
                st.write(top_station)

                st.write(
                    "Using an interactive ***Bar Chart***, we can visualize these stations and look at the ***Time*** associated " +
                    "with the high traffic per stations. With the station legends ***(Names)*** on the right of the chart, we can " +
                    "select the stations we want to view by dehighlighting other station. This presents a bolder non-clustered chart")

                df_bar = df_top.head(500)
                # df_bar = df_bar.set_index('time')
                fig = px.bar(df_bar, x='time', y='counts', color='station')
                st.plotly_chart(fig)

                st.write(
                    "The Bar Chart above provided the right insight on the ***Time*** and ***Stations*** with highest passenger " +
                    "traffic. But to view the year that produced the top 30 recorded passenger counts within the period of 4 years, " +
                    "we will use a static Bar Chart this time. The plot below presents that data with complete date and time on X-axis. " +
                    "and Passenger counts on Y-axis")
                from plotly.offline import plot
                ax = sns.barplot(x='entry_date_time', y='counts',
                                 data=df_top.sort_values('counts', ascending=False).head(30),
                                 hue='station', palette='twilight_shifted', lw=3)
                plt.xticks(rotation=45)
                st.write(ax.get_figure())

                with dataset:
                    st.markdown('''
                    <style>
                    [data-testid="stMarkdownContainer"] ul{
                        list-style-position: inside;
                    }
                    </style>
                    ''', unsafe_allow_html=True)
                    st.markdown(
                        '<p style="font-family:Courier; color:Blue; font-size: 20px;">TIME SERIES VISUALIZATION</p>',
                        unsafe_allow_html=True)

                    # Building the function to help slice the required data for visualization
                    def get_data_top_stations(df, year, day, dire):
                        df = df
                        day = day
                        dire = dire
                        year = year

                        data_in_sorted = df.sort_values(['asc', 'time'], ascending=[True, True])
                        df_in_new = data_in_sorted.loc[data_in_sorted.day == day]
                        df_in_new = df_in_new.loc[df_in_new.dir == dire]
                        df_in_new_year = df_in_new.loc[df_in_new.year_of_entry == year]

                        df_in_new_year = df_in_new_year[['entry_date_time', 'station', 'time', 'counts']]

                        return df_in_new_year

                    st.write("Lets visualize the entire dataset to see passenger counts relations in the to stations")

                    data_top = pd.read_csv('resources/data/tube_time_interval_data_sorted.csv')
                    data_top['counts'] = data['counts'].round(decimals=0)

                    list_station = ['Bank and Monument', 'Waterloo LU', 'Oxford Circus',
                                    'Canary Wharf LU', 'Liverpool Street LU', 'Moorgate', 'London Bridge LU',
                                    'Farringdon', 'Victoria LU', 'Green Park']
                    df_top_station = data_top.loc[data_top['station'].isin(list_station)]

                    col1, col2, col3 = st.columns(3)
                    col1 = st.selectbox('year', (2018, 2019, 2020, 2021))
                    year = col1
                    col2 = st.selectbox('select day', ('MTT', 'FRI', 'SAT', 'SUN'))
                    day = col2
                    col3 = st.selectbox('direction', ('IN', 'OUT'))
                    dire = col3

                    df = get_data_top_stations(df_top_station, year, day, dire)

                    sns.set_style('darkgrid')
                    sns.set(rc={'figure.figsize': (14, 8)})

                    fig2 = px.line(df.sort_values('entry_date_time', ascending=True), x='time', y='counts',
                                   color='station')
                    st.plotly_chart(fig2)



    if page_selection == "Passenger Forecast Modeling":
        model_selection = st.selectbox("MODEL SELECTION", model_options)
        if model_selection == "Raw Dataset Overview":
            st.write("#### Below is an overview of the unprocessed dataset and its features")
            st.dataframe(EntryExit_2021)


        if model_selection == "Linear Regressor":
            dataset_selection = st.selectbox("Select Dataset to Predict", dataset_options)
            if dataset_selection == "Passenger Station Entry/Exit Counts Year: 2021":
                if st.button("Make Prediction"):
                    lr_pred_train = lr.predict(X_train)
                    lr_pred_test = lr.predict(X_test)
                    import pandas as pd
                    st.write('##### 15 Min Interval Model Predictions for Passenger Station Counts across 24hr Period')
                    df = pd.DataFrame(lr_pred_test, columns=col_list)
                    st.dataframe(df)
                    st.write('##### Model Performance Metrics')
                    col1, col2 = st.columns([1, 0.215])
                    col1.metric(label="RMSE Value", value = round(np.sqrt(mean_squared_error(y_test, lr_pred_test)), 2) ,
                                   help= "Displays the RMSE Value of the Trained Model on the current dataset")
                    col2.metric(label="R-Squared Value", value = round(r2_score(y_test, lr_pred_test),4) ,
                                   help= "Displays the R-Squared Value of the Trained Model on the current dataset")


        if model_selection == "Decision Tree Regressor":
            dataset_selection = st.selectbox("Select Dataset to Predict", dataset_options)
            if dataset_selection == "Passenger Station Entry/Exit Counts Year: 2021":
                if st.button("Make Prediction"):
                    dtr_pred_train = dtr.predict(X_train)
                    dtr_pred_test = dtr.predict(X_test)
                    import pandas as pd
                    st.write('##### 15 Min Interval Model Predictions for Passenger Station Counts across 24hr Period')
                    df = pd.DataFrame(dtr_pred_test, columns=col_list)
                    st.dataframe(df)
                    st.write('##### Model Performance Metrics')
                    col1, col2 = st.columns([1, 0.215])
                    col1.metric(label="RMSE Value", value = round(np.sqrt(mean_squared_error(y_test, dtr_pred_test)), 2) ,
                                   help= "Displays the RMSE Value of the Trained Model on the current dataset")
                    col2.metric(label="R-Squared Value", value = round(r2_score(y_test, dtr_pred_test),4) ,
                                   help= "Displays the R-Squared Value of the Trained Model on the current dataset")


        if model_selection == "Random Forest Regressor":
            dataset_selection = st.selectbox("Select Dataset to Predict", dataset_options)
            if dataset_selection == "Passenger Station Entry/Exit Counts Year: 2021":
                if st.button("Make Prediction"):
                    rfr_pred_train = rfr.predict(X_train)
                    rfr_pred_test = rfr.predict(X_test)
                    import pandas as pd
                    st.write('##### 15 Min Interval Model Predictions for Passenger Station Counts across 24hr Period')
                    df = pd.DataFrame(rfr_pred_test, columns=col_list)
                    st.dataframe(df)
                    st.write('##### Model Performance Metrics')
                    col1, col2 = st.columns([1, 0.215])
                    col1.metric(label="RMSE Value", value = round(np.sqrt(mean_squared_error(y_test, rfr_pred_test)), 2) ,
                                   help= "Displays the RMSE Value of the Trained Model on the current dataset")
                    col2.metric(label="R-Squared Value", value = round(r2_score(y_test, rfr_pred_test),4) ,
                                   help= "Displays the R-Squared Value of the Trained Model on the current dataset")


        if model_selection == "K-Nearest Neighbors Regressor":
            dataset_selection = st.selectbox("Select Dataset to Predict", dataset_options)
            if dataset_selection == "Passenger Station Entry/Exit Counts Year: 2021":
                if st.button("Make Prediction"):
                    knn_pred_train = knn.predict(X_train)
                    knn_pred_test = knn.predict(X_test)
                    import pandas as pd
                    st.write('##### 15 Min Interval Model Predictions for Passenger Station Counts across 24hr Period')
                    df = pd.DataFrame(knn_pred_test, columns=col_list)
                    st.dataframe(df)
                    st.write('##### Model Performance Metrics')
                    col1, col2 = st.columns([1, 0.215])
                    col1.metric(label="RMSE Value", value = round(np.sqrt(mean_squared_error(y_test, knn_pred_test)), 2) ,
                                   help= "Displays the RMSE Value of the Trained Model on the current dataset")
                    col2.metric(label="R-Squared Value", value = round(r2_score(y_test, knn_pred_test),4) ,
                                   help= "Displays the R-Squared Value of the Trained Model on the current dataset")


        if model_selection == "Elastic Net Regressor":
            dataset_selection = st.selectbox("Select Dataset to Predict", dataset_options)
            if dataset_selection == "Passenger Station Entry/Exit Counts Year: 2021":
                if st.button("Make Prediction"):
                    enr_pred_train = enr.predict(X_train)
                    enr_pred_test = enr.predict(X_test)
                    import pandas as pd
                    st.write('##### 15 Min Interval Model Predictions for Passenger Station Counts across 24hr Period')
                    df = pd.DataFrame(enr_pred_test, columns=col_list)
                    st.dataframe(df)
                    st.write('##### Model Performance Metrics')
                    col1, col2 = st.columns([1, 0.215])
                    col1.metric(label="RMSE Value", value = round(np.sqrt(mean_squared_error(y_test, enr_pred_test)), 2) ,
                                   help= "Displays the RMSE Value of the Trained Model on the current dataset")
                    col2.metric(label="R-Squared Value", value = round(r2_score(y_test, enr_pred_test),4) ,
                                   help= "Displays the R-Squared Value of the Trained Model on the current dataset")


        if model_selection == "Bayesian Ridge Regressor":
            dataset_selection = st.selectbox("Select Dataset to Predict", dataset_options)
            if dataset_selection == "Passenger Station Entry/Exit Counts Year: 2021":
                if st.button("Make Prediction"):
                    w_br_pred_train = w_br.predict(X_train)
                    w_br_pred_test = w_br.predict(X_test)
                    import pandas as pd
                    st.write('##### 15 Min Interval Model Predictions for Passenger Station Counts across 24hr Period')
                    df = pd.DataFrame(w_br_pred_test, columns=col_list)
                    st.dataframe(df)
                    st.write('##### Model Performance Metrics')
                    col1, col2 = st.columns([1, 0.215])
                    col1.metric(label="RMSE Value", value = round(np.sqrt(mean_squared_error(y_test, w_br_pred_test)), 2) ,
                                   help= "Displays the RMSE Value of the Trained Model on the current dataset")
                    col2.metric(label="R-Squared Value", value = round(r2_score(y_test, w_br_pred_test),4) ,
                                   help= "Displays the R-Squared Value of the Trained Model on the current dataset")


        if model_selection == "Time-Series Model":
            with header:
                st.title('Forecasting The Tube Passenger Count')

            def get_data_year(df, station, day, dire, year):

                df = df
                station = station
                day = day
                dire = dire
                year = year

                data_in_sorted = df.sort_values(['asc', 'time'], ascending=[True, True])
                df_in_new = data_in_sorted.loc[data_in_sorted.station == station]
                df_in_new = df_in_new.loc[df_in_new.day == day]
                df_in_new = df_in_new.loc[df_in_new.dir == dire]
                df_in_new_year = df_in_new.loc[df_in_new.year_of_entry == year]

                df_in_new_year = df_in_new_year[['entry_date_time', 'counts']]

                return df_in_new_year

            with dataset:
                st.write("Provide the program some input details depending on the station to forecast passenger count")
                st.write("The ***year*** column will select the historical tube data of the year we want to use "
                         "for the forcasting. The ***select day*** will select the day of week you want to forecast "
                         "while ***direction*** column will determine passenger movement direction from the station (IN or OUT).")
                station_option = st.selectbox('select station',
                                              ('Bank and Monument', 'Waterloo LU', 'Oxford Circus', 'Canary Wharf LU',
                                               'Liverpool Street LU', 'Moorgate', 'London Bridge LU', 'Farringdon',
                                               'Victoria LU', 'Green Park', "King's Cross St. Pancras", 'Holborn',
                                               'Brixton LU', 'Stratford', 'Finsbury Park'))
                station = station_option

                day_option = st.selectbox('select day',
                                          ('MTT', 'FRI', 'SAT', 'SUN'))
                day = day_option

                dir_option = st.selectbox('direction',
                                          ('IN', 'OUT'))
                dire = dir_option

                year_option = st.selectbox('year',
                                           (2018, 2019, 2020, 2021))
                year = year_option

                data = pd.read_csv('resources/data/tube_time_interval_data_sorted.csv')
                data['counts'] = data['counts'].round(decimals=0)

                df = get_data_year(data, station, day, dire, year)
                if day == 'MTT':
                    day = "Mondays to Thursdays"
                elif day == 'FRI':
                    day = "Fridays"
                elif day == 'SAT':
                    day = "Saturdays"
                elif day == 'SUN':
                    day = "Sundays"
                st.subheader(
                    'Passenger Forecast for ' + station + " station on " + day + " according to 15-minutes time interval")
                st.write(
                    'Preview of the ' + station + ' ' + 'station' + ' ' + str(year) + ' ' + 'data (first 5 records)')
                st.write(df.head())
                st.write("Click the **Visualize** button below to view the **Time Series** "
                         "graph of " + station + " station Historical Data on " + day + " for " + str(year))
                if st.button('Visualize'):
                    st.line_chart(df.rename(columns={'entry_date_time': 'index'}).set_index('index'))

                st.write("Click the **Make Forecast** button to see the passenger movement "
                         "predition for " + station + ' on ' + day)
                df.columns = ['ds', 'y']
                m = Prophet(interval_width=0.95, daily_seasonality=True)
                model = m.fit(df)
                future = m.make_future_dataframe(periods=96, freq='15T')
                forecast = m.predict(future)

                if st.button('Make Forecast'):
                    fig = plot_plotly(model, forecast)
                    fig
                    # plot1 = m.plot(forecast)
                    # st.write(plot1)
                    st.info(
                        "The generated interactive chart above shows the passenger counts forecast graph for all 15-minutes intervals next " + day + "for " + station +
                        ". You hover around the chart to see the result for each 15-minutes time frame of interes, and drag along to see to the end of the day.")

                if st.button('Explore Componets of The Forecast'):
                    plot2 = m.plot_components(forecast)
                    st.write(plot2)



    if page_selection == "Graphical Network Representation":
        with header:
            st.title('The London Undergroung Graphical Network')

            st.markdown('## Graphical Representation and Analyses of the Tube Network')
            st.markdown(
                'In this section of the project, we will be representing the London Tube rail network as a graph. And we think utilizing NetworkX will be the esiest way to achieve this. Nodes will be the stations and edges are the connections between them. We will make some analyses our graphs such as pageranking, calculating Hits, degree of centralities and inbetweeness etc.')

            st.write("### Let's start by loading all needed dataframes")

            st.write('Showing top entries of the London Underground stations (aka Nodes).')
            st.write(lu_stations.head(3))

            st.write(
                'Showing top entries of the London Underground loading data, which will serve as the Connections between the stations (aka Edges).')
            st.write(connections.head(3))

            st.write('Below are the London Underground Lines (aka Edge Labels).')
            st.write(lu_lines)

            st.markdown('### A simplified graph')
            st.markdown('Now that we have our dataframes, we can create a simple graph of the network.')

            simple_graph = nx.Graph()
            simple_graph.add_nodes_from(lu_stations['name'])
            simple_graph.add_edges_from(list(zip(connections['from_station'], connections['to_station'])))

            plt.figure(figsize=(8, 5))
            st.pyplot(nx.draw(simple_graph, node_size=5))

            st.markdown('We can see from the above graph what the London Tube connections look like. Nodes which are ditached from the network are stations from the Stations_Coordinates.csv file that has no loading record in the LU_Loading_Data.csv file. \
                            Although this is not a realistic representation of the stations location compared to what they would look like on a geographical map.')
            st.markdown(
                'Already we can even do some analysing on the graph, like getting a reasonable (shortest) path between the stations `Oxford Circus` and `Canary Wharf`')

            st.write(nx.shortest_path(simple_graph, 'Oxford Circus', 'Canary Wharf'))

            st.markdown(
                'Also we can run the PageRank and Hits algorithm on the network to messure the connections between the LU stations. Both of these compares the nodes (LU stations) using the numbers of connections found between them.')
            st.markdown("This time though, we'll focus on the stations that has connections between them as edges.")

            graph = nx.Graph()
            graph.add_edges_from(list(zip(connections['from_station'], connections['to_station'])))

            pagerank = nx.pagerank_numpy(graph)
            import pandas as pd
            pagerank = pd.DataFrame(pagerank.items(), columns=['name', 'pagerank'])
            stations = pd.merge(lu_stations, pagerank, on='name')

            st.write(stations.sort_values('pagerank', ascending=False).head(10))

            hits = nx.hits_scipy(graph, max_iter=1000)[0]
            hits = pd.DataFrame(hits.items(), columns=['name', 'hits'])
            stations = pd.merge(stations, hits, on='name')

            st.write(stations.sort_values('hits', ascending=False).head(10))

            st.markdown(
                'We show the top 10 station rank for both PageRank and Hits comparison above. Where PageRank finds the most important stations, the HITS algorithm seems to be pretty good at finding the busiest stations. To fully understand this, we can say the network relies on the important stations to function and without them, operations will be affected significantly. But the busiest stations does not impact the network operation in such significant way, they only tell us which stations has the highest traffic.')
            st.markdown("Lets visualise the importance of stations as defined by PageRank. Less important stations will be colored green, and more important stations will be colored red. \
                            At the same time, we'll make use of the coordinates from our `stations` dataframe to allign the nodes in order to make our graph a more realistic plot of the London Underground stations.")

            def pseudocolor(val):
                h = (1.0 - val) * 120 / 360
                r, g, b = colorsys.hsv_to_rgb(h, 1., 1.)
                return r * 255, g * 255, b * 255

            normed = stations[['longitude', 'latitude', 'pagerank']]
            normed = normed - normed.min()
            normed = normed / normed.max()
            locations = dict(zip(stations['name'], normed[['longitude', 'latitude']].values))
            pageranks = dict(zip(stations['name'], normed['pagerank'].values))

            p = figure(
                title='The London Underground Network',
                x_range=(.4, .7),
                y_range=(.2, .5),
                height=700,
                width=1100,
                toolbar_location='above'
            )

            # p.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

            for edge in graph.edges():
                try:
                    p.line(
                        x=[locations[pt][0] for pt in edge],
                        y=[locations[pt][1] for pt in edge],
                    )
                except KeyError:
                    pass

            for node in graph.nodes():
                try:
                    x = [locations[node][0]]
                    y = [locations[node][1]]
                    p.circle(
                        x, y,
                        radius=.01 * pageranks[node],
                        fill_color=pseudocolor(pageranks[node]),
                        line_alpha=0)
                    p.text(
                        x, y,
                        text={'value': node},
                        text_font_size=str(min(pageranks[node] * 12, 10)) + "pt",
                        text_alpha=pageranks[node],
                        text_align='center',
                        text_font_style='bold')
                except KeyError:
                    pass

            st.bokeh_chart(p)  # Optional argument (use_container_width=True)
            # show(p)

            st.markdown('### Further analyses that can be done on this graph include the following:')
            st.markdown('* Degree Centrality')
            st.markdown('* Edge labeling and ranking')
            st.markdown('* And more...')

            st.markdown(' ### End of Analysis')


    if page_selection == 'About The Team':
        st.title("TEAM PROFILE")
        st.info("**Welcome to the Explore AI Team 6 The Tube Project Team**")

        # Display Images side by side
        from PIL import Image
        col1, col2 = st.columns(2)
        with col1:
            st.image('resources/images/emmanuel.jpeg', width=243)
        with col2:
            st.subheader("Fielami Emmanuel David")
            st.markdown('**Data Scientist**')

            st.markdown("<p><b>LinkedIn</b>: <a href='www.linkedin.com/in/fielami'>Emmanuel Fielami</a>"
                "<br> <b>Email</b>: emmzytamara2@gmail.com<br> <b>Contact</b>: +2347067805884<br>"
                "<b>About</b>: A top - level Data Scientist and web developer, dedicated to providing cutting edge solutions "
                " to real world problems. I have spent the past 3 + years doing data analytics, number crunching and database"
                " management in the Public sector, 2 + years developing web applications on a part - time basis, and close to a year "
                " studying and working with Big Data as a Data Scientist with an Artificial Intelligence company"
                " in South Africa. </p>", unsafe_allow_html=True)


        col3, col4 = st.columns(2)
        with col3:
            st.image('resources/images/michael.jpeg', width=243)
        with col4:
            st.subheader("Michael Chidike Mamah")
            st.write("**Data Scientist**")
            st.markdown(
                "<p><b>LinkedIn</b>: <a href='https://ng.linkedin.com/in/michael-mamah-b88b5315b'>Michael Mamah</a>"
                "<br><b>Email</b>: mamachidike@mail.com<br><b>Contact</b>: +2348123234582<br>"
                "<b>About</b>: Trained Data Scientist leveraging data, "
                "machine learning, and cloud technologies to provide data solutions with "
                "focus on efficient resource use, economic growth, and sustainable "
                "developments. </p>", unsafe_allow_html=True)

        col5, col6 = st.columns(2)
        with col5:
            st.image('resources/images/kelvin.png', width=243)
        with col6:
            st.subheader("Kelvin Mwaniki")
            st.write("**Data Sciencist**")
            st.markdown("<p>LinkedIn</b>: <a href='www.linkedin.com/in/kelvin-mwaniki-89ab65171'>Kelvin Mwaniki</a>"
                        "<br>Email</b>: mwaniki.kelvin918@gmail.com"
                        "<br>Contact</b>: +254711427305<br>"
                        "<b>About</b>: An enthusiastic data scientist with 1+ years experience coding in python, "
                        "SQL, dashboard design on Power BI, app design and deployment on streamlit & cloud hosting "
                        "on Amazon Web Services. With a miriad of wolrd class projects under my belt under the tutelage of Explore A.I. "
                         , unsafe_allow_html = True)


        col7, col8 = st.columns(2)
        with col7:
            st.image('resources/images/hakim.jpeg', width=243)
        with col8:
            st.subheader("Hakim Balogun")
            st.write("**Data Engineer**")
            st.markdown("<p>LinkedIn</b>: <a href='https://www.linkedin.com/in/hakimbalogun'>Hakim Balogun</a>"
                        "<br>Email</b>: hakim.obalogun@gmail.com"
                        "<br>Contact</b>: +2348039100897<br>"
                        "<b>About</b>: A competent professional with an excellent work ethic and "
                        "increasing technical skills in data engineering and cloud computing fields. "
                        "When I'm not working I take walks, listen to music, and see the world around me. </p>", unsafe_allow_html = True)



        col9, col10 = st.columns(2)
        with col9:
            st.image('resources/images/harmony.jpeg', width=243)
        with col10:
            st.subheader("Harmony Odumuko")
            st.write("**Data Scientist**")
            st.markdown("<p>LinkedIn</b>: <a href='www.linkedin.com/in/hodumuko'>Harmony Odumuko</a>"
                        "<br>Email</b>: nibotics@gmail.com"
                        "<br>Contact:</b>: +2348024541916", unsafe_allow_html=True)

        col11, col12 = st.columns(2)
        with col11:
            st.image('resources/images/endurance.jpeg', width=243)
        with col12:
            st.subheader("Endurance Arienkhe")
            st.write("**Data Scientist**")
            st.markdown("<p>LinkedIn</b>: <a href='www.linkedin.com/in/endurance-arienkhe'>Endurance Arienkhe</a>"
                        "<br>Email</b>: endurance.arienkhe@gmail.com"
                        "<br>Contact:</b>: +2347065516048", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
