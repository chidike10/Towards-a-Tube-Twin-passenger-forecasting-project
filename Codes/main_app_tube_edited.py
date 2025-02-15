import streamlit as st
import pandas as pd
import numpy as np
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
import plotly.figure_factory as ff
import streamlit.components.v1 as plot_components
import networkx as nx 
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_pydot import write_dot
import colorsys 
import os
#from bokeh.plotting import figure, show
#from bokeh.io import output_file
#from bokeh.plotting import figure 

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
    "About The Project",
    "Explorative Data Analysis [EDA]",
    "Passenger Forecast",
    #"Tube Network Graph",
    "About The Team"
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


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Base directory of the project
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def get_file_path(*path_parts):
        """Constructs a file path relative to the base directory."""
        return os.path.join(BASE_DIR, *path_parts)

    # Example usage:
    DATA_FILE = get_file_path("resources", "data", "tube_time_interval_data_sorted.csv")
    landing_image = get_file_path("resources", "images", "Landing_Page_Background.jpg")
    sidebar_image = get_file_path("resources", "images", "LU_logo.jpg")
    tube_image = get_file_path("resources", "images", "the tube.jpeg")
    bokeh_graph_image = get_file_path("resources", "images", "bokeh_plot_new.png")
    forecast_image = get_file_path("resources", "images", "forecast.jpg")
    app_process_image = get_file_path("resources", "images", "TubeTwin App Process Flow.jpg")
    michael_image = get_file_path("resources", "images", "michael.jpeg")
    emmanuel_image = get_file_path("resources", "images", "emmanuel.jpeg")
    hakim_image = get_file_path("resources", "images", "hakim.jpeg")
    harmony_image = get_file_path("resources", "images", "harmony.jpeg")
    endurance_image = get_file_path("resources", "images", "endurance.jpeg")

    st.sidebar.image(sidebar_image, use_column_width=True)
    page_selection = st.sidebar.selectbox("PAGE SELECTION", page_options)

    if page_selection == "Landing Page":
        add_bg_from_local(landing_image)

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

    if page_selection == 'About The Project':
    
        with header:
            st.title('The Tube Twin Project')
            st.image(tube_image, use_column_width='always', output_format='jpeg')
            st.subheader('Summary')  
            st.write("With around 400km of rails, over 267 stations, and more than 1.3 billion "+ 
                "passenger journeys each year (according to the project data), quickly andsafely "+ 
                "moving passengers through stations and onto trains is an ongoing priority for "+ 
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
                "like the number of lines connecting on stations (edges)." )
            st.image(bokeh_graph_image, width=600 )
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
            st.image(forecast_image, width=600 )
            st.info('***fig2 - Moorgate station passengers forecast***')
            st.write("In the forecast plot above, the dots in the graph represents the actual data points, while "
                "the line represents the model predictions. The shaded region which the line runs through "
                "is the ***boundary*** of the predictions given as lower and upper bounds. The entire region "
                "from where the thick dots ends represent the forecasted region which is same as the peroid"
                "of the station data (passenger counts in all 15 minutes time interval on saturdays)"
                "") 

            st.subheader("System Design and Development")
            st.write("All applications were developed using python programming language and several open software " 
                "packages like NetworkX, FBProphet, SUMO, and Streamlit. The diagram below depicts the flow of "
                "processes of the project applications from start to finish. These include the use of AWS "
                "resources (EC2 and S3) for computing and storage resources.")
            st.image(app_process_image, width=600)
            st.write("Looking at the diagram above, we can affirm that every data project after underground study "
                "kickstarts with actual data collection from the data source. This ***Tube Twin*** project started "
                "technically with the background study and data collection and engineering. After achieving "
                "shaped into the required formats, they are then saved into the bucket on AWS where our applications "
                "***clean datasets*** runs. All application codes are also stored in the S3 bucket attached to "
                "the EC2 instance responsible for run the app")

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
                + "of the week. This was collected from the Traffic for London (TfL) website and engineered " + 
                "to fit our project expected outcome")

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
                "by using visual plots to explore the historical data and tell few stories before we try to make " + 
                "some forecast into the future passenger counts")
            import pandas as pd

            data = pd.read_csv(DATA_FILE)

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
                'and ***Waterloo LU*** stations are 2 most busiest, and the trend is spread across different years;'+
                ' interestingly they both lie in the same ***zone 1*** '+ "of The Tube map. This shows a "+ 
                "potential heavy weight on the associated lines. We will explore these stations.")

            st.write('The following list of stations are those captured in the dataset above with passenger counts ' +
                'greater than 3000 for their corresponding years in the daily 15 minutes time interval.' + 
                ' These stations will be our main focus of interest in this exploration, and also in the passenger counts forecasting '+
                'as understanding the factors contributing to the busy passenger flow (IN and OUT) of the station will '+
                'help us make good recommendations to TFL on improving the London Underground (The Tube) network lines')
            top_station = df_top['station'].unique()
            st.write(top_station)

            st.write("Using an interactive ***Bar Chart***, we can visualize these stations and look at the ***Time*** associated "+
                "with the high traffic per stations. With the station legends ***(Names)*** on the right of the chart, we can "+
                "select the stations we want to view by dehighlighting other stations. This presents a bolder non-clustered chart")

            df_bar = df_top.head(500)
            # df_bar = df_bar.set_index('time')
            fig = px.bar(df_bar, x='time', y = 'counts', color='station')
            st.plotly_chart(fig)

            st.write("The Bar Chart above provided the right insight on the ***Time*** and ***Stations*** with highest passenger "+
                "traffic. But to view the time in all the entire data (all years inclusive) that produced the top 30 recorded passenger counts within the period of 4 years, "+
                "we will use a static Bar Chart this time. The plot below presents the data with complete date and time on X-axis. "+
                "and Passenger counts on Y-axis")
            from plotly.offline import plot
            ax = sns.barplot(x = 'entry_date_time', y = 'counts', data = df_top.sort_values('counts', ascending=False).head(30),  
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
                st.markdown('<p style="font-family:Courier; color:Blue; font-size: 20px;">TIME SERIES VISUALIZATION</p>', 
                    unsafe_allow_html=True)
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

                st.write("Lets visualize the entire dataset to see passenger counts relations in the to stations")

                data_top = pd.read_csv(DATA_FILE)
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
            st.write("Provide the program some input details depending on the station to forecast passenger count")
            st.write("The ***year*** column will select the historical tube data of the year we want to use "
            "for the forcasting. The ***select day*** will select the day of week you want to forecast "
            "while ***direction*** column will determine passenger movement direction from the station (IN or OUT)."
            " Note the data dynamically changing as you select the station, day, direct, and year inputs.")
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

            data = pd.read_csv(DATA_FILE)
            data['counts'] = data['counts'].round(decimals=0)

            df = get_data_year(data, station, day, dire, year)
            if day=='MTT':
                day = "Mondays to Thursdays" 
            elif day=='FRI':
                day = "Fridays"
            elif day=='SAT':
                day = "Saturdays"
            elif day=='SUN':
                day = "Sundays" 
            st.subheader('Passenger Forecast for '+ station+ " station on "+day+ " according to 15-minutes time interval")
            st.write('Preview of the '+station + ' ' + 'station'+ ' '+ str(year) + ' ' + 'data (first 5 records)')
            st.write(df.head())
            st.write("Click the **Visualize** button below to view the **Time Series** "
                "graph of "+ station+ " station Historical Data on "+day + " for " +str(year))
            if st.button('Visualize'):
                st.line_chart(df.rename(columns={'entry_date_time':'index'}).set_index('index'))
        
            st.write("Click the **Make Forecast** button to see the passenger movement "
                "predition for "+station+ ' on '+ day)
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
                st.info("The generated interactive chart above shows the passenger counts forecast graph for all 15-minutes intervals next "+ day +"for "+station+
                    ". You hover around the chart to see the result for each 15-minutes time frame of interes, and drag along to see to the end of the day.")
            
            if st.button('Explore Componets of The Forecast'): 
                plot2 = m.plot_components(forecast)
                st.write(plot2)

    
    # if page_selection == "Tube Network Graph": 
    #     lu_stations = pd.read_csv('../Data/TfL-Station-Data-detailed/Transformed/Stations_Coodinates.csv', index_col=0) 
    #     connections = pd.read_csv('../Data/TfL-Station-Data-detailed/Transformed/LU_Loading_Data.csv') 
    #     lu_lines = pd.read_csv('../Data/TfL-Station-Data-detailed/Transformed/Wiki/Lines.csv', index_col=0) 
        
    #     with dataset:

    #         st.title('The London Undergroung Graphical Network') 
                
    #         st.markdown('## Graphical Representation and Analyses of the Tube Network') 
    #         st.markdown('In this section of the project, we will be representing the London Tube rail network as a graph. And we think utilizing NetworkX will be the esiest way to achieve this. Nodes will be the stations and edges are the connections between them. We will make some analyses our graphs such as pageranking, calculating Hits, degree of centralities and inbetweeness etc.') 
                                        
    #         st.write("### Let's start by loading all needed dataframes") 
                                
    #         st.write('Showing top entries of the London Underground stations (aka Nodes).') 
    #         st.write(lu_stations.head(3)) 

    #         st.write('Showing top entries of the London Underground loading data, which will serve as the Connections between the stations (aka Edges).') 
    #         st.write(connections.head(3)) 
                
    #         st.write('Below are the London Underground Lines (aka Edge Labels).') 
    #         st.write(lu_lines) 

    #         st.markdown('### A simplified graph') 
    #         st.markdown('Now that we have our dataframes, we can create a simple graph of the network.')
                
    #         simple_graph = nx.Graph() 
    #         simple_graph.add_nodes_from(lu_stations['name']) 
    #         simple_graph.add_edges_from(list(zip(connections['from_station'], connections['to_station']))) 
            
    #         plt.figure(figsize =(8, 5)) 
    #         st.pyplot(nx.draw(simple_graph, node_size = 5)) 
                
    #         st.markdown('We can see from the above graph what the London Tube connections look like. Nodes which are ditached from the network are stations from the Stations_Coordinates.csv file that has no loading record in the LU_Loading_Data.csv file. \
    #                     Although this is not a realistic representation of the stations location compared to what they would look like on a geographical map.') 
    #         st.markdown('Already we can even do some analysing on the graph, like getting a reasonable (shortest) path between the stations `Oxford Circus` and `Canary Wharf`') 

    #         st.write(nx.shortest_path(simple_graph, 'Oxford Circus', 'Canary Wharf')) 
    #         st.markdown('Also we can run the PageRank and Hits algorithm on the network to messure the connections between the LU stations. Both of these compares the nodes (LU stations) using the numbers of connections found between them.')
    #         st.markdown("This time though, we'll focus on the stations that has connections between them as edges.") 

    #         graph = nx.Graph() 
    #         graph.add_edges_from(list(zip(connections['from_station'], connections['to_station']))) 

    #         pagerank = nx.pagerank(graph) 
    #         pagerank = pd.DataFrame(pagerank.items(), columns=['name', 'pagerank'])
    #         stations = pd.merge(lu_stations, pagerank, on='name') 

    #         st.write(stations.sort_values('pagerank', ascending=False).head(10)) 
    #         hits = nx.hits(graph, max_iter=1000)[0]
    #         hits = pd.DataFrame(hits.items(), columns=['name', 'hits'])
    #         stations = pd.merge(stations, hits, on='name') 

    #         st.write(stations.sort_values('hits', ascending=False).head(10)) 

    #         st.markdown('We show the top 10 station rank for both PageRank and Hits comparison above. Where PageRank finds the most important stations, the HITS algorithm seems to be pretty good at finding the busiest stations. To fully understand this, we can say the network relies on the important stations to function and without them, operations will be affected significantly. But the busiest stations does not impact the network operation in such significant way, they only tell us which stations has the highest traffic.') 
    #         st.markdown("Lets visualise the importance of stations as defined by PageRank. Less important stations will be colored green, and more important stations will be colored red. \
    #                     At the same time, we'll make use of the coordinates from our `stations` dataframe to allign the nodes in order to make our graph a more realistic plot of the London Underground stations.") 

    #         def pseudocolor(val):
    #             h = (1.0 - val) * 120 / 360
    #             r, g, b = colorsys.hsv_to_rgb(h, 1., 1.)
    #             return r * 255, g * 255, b * 255 

    #         normed = stations[['longitude', 'latitude', 'pagerank']] 
    #         normed = normed - normed.min() 
    #         normed = normed / normed.max() 
    #         locations = dict(zip(stations['name'], normed[['longitude', 'latitude']].values)) 
    #         pageranks = dict(zip(stations['name'], normed['pagerank'].values)) 

    #         p = figure( 
    #             #title='The London Underground Network', 
    #             x_range = (.4,.7), 
    #             y_range = (.2,.5), 
    #             height= 700, 
    #             width= 1100,
    #             toolbar_location='above'  
    #         ) 

    #             # p.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool()) 

    #         for edge in graph.edges(): 
    #             try: 
    #                 p.line( 
    #                     x= [locations[pt][0] for pt in edge],
    #                     y= [locations[pt][1] for pt in edge],
    #                 )
    #             except KeyError:
    #                 pass 

    #         for node in graph.nodes():
    #             try: 
    #                 x = [locations[node][0]]
    #                 y = [locations[node][1]]
    #                 p.circle( 
    #                     x, y, 
    #                     radius = .01 * pageranks[node], 
    #                     fill_color = pseudocolor(pageranks[node]), 
    #                     line_alpha=0) 
    #                 p.text(
    #                     x, y,                              
    #                     text = {'value':node}, 
    #                     text_font_size = str(min(pageranks[node] * 12, 10)) + "pt", 
    #                     text_alpha = pageranks[node],
    #                     text_align='center',
    #                     text_font_style='bold') 
    #             except KeyError:
    #                 pass                 

    #         st.bokeh_chart(p) # Optional argument (use_container_width=True) 
    #             # show(p) 

    #         st.markdown('### Further analyses that can be done on this graph include the following:') 
    #         st.markdown('* Degree Centrality') 
    #         st.markdown('* Edge labeling and ranking') 
    #         st.markdown('* And more...') 

    #         st.markdown(' ### End of Analyses')  
    
    #     st.title('TRAIN SIMULATION')

    #     st.caption('here')
    #     st.text_area('here')

    if page_selection == 'About Team':
        st.title('ABOUT TEAM')

        # Building our company's profile page
    if page_selection == "About The Team":
        
        st.title("TEAM PROFILE") 

        st.info("**Welcome to the Explore AI Team 6 The Tube Project Team**")
               
        #Display Images side by side        
        from PIL import Image     

        col1, col2 = st.columns(2)        
        with col1:            
            st.image(michael_image, width=243)    
        with col2:
            st.subheader("Michael Chidike Mamah")
            st.write("**Data Scientist**")
            st.markdown("<p><b>LinkedIn</b>: <a href='www.linkedin.com/in/michaelmamah'>Michael Mamah</a>"
                "<br><b>Email</b>: mamachidike@mail.com<br><b>Contact</b>: +234 704 244 6244<br>"
                "<b>About</b>: Trained Data Scientist leveraging data, "
                "machine learning, and cloud technologies to provide data solutions with "
                "focus on efficient resource use, economic growth, and sustainable "
                "developments. </p>", unsafe_allow_html=True) 

        col3, col4 = st.columns(2)
        with col3:             
            st.image(emmanuel_image, width =243)
        with col4:
            st.subheader("Fielami Emmanuel David")
            st.markdown('**Data Scientist**')
            st.markdown("<p><b>LinkedIn</b>: <a href='www.linkedin.com/in/fielami'>Emmanuel Fielami</a>"
                "<br> <b>Email</b>: emmzytamara2@gmail.com<br> <b>Contact</b>: +234 706 780 5884<br> <b>About</b> </p>", unsafe_allow_html=True)
            # st.markdown('<p style="font-family:Savana; color:Black; font-size: 18px;">Contact:<br>Name<br>Phone</p>', 
            #         unsafe_allow_html=True)
                                   

        col5, col6 = st.columns(2)             
#        with col5:
#            st.image('resources/images/kelvin.jpeg', width=243)                                          
        with col6:
            st.subheader("Kelvin Mwaniki")
            st.write("**Data Scientist**")
            st.markdown("<p><b>LinkedIn</b>: <a href=''>Kelvin Mwaniki</a>"
                "<br><b>Email</b>: <br><b>Contact</b>: +234 812 323 4582<br>"
                "<b>About</b>: </p>", unsafe_allow_html=True)

        col11, col12 = st.columns(2)
        with col11:            
            st.image(hakim_image, width =243) 
        with col12:
            st.subheader("Hakim Balogun")
            st.write("**Data Engineer**") 
            st.markdown("<p><b>LinkedIn</b>: <a href='https://www.linkedin.com/in/hakimbalogun'>Hakim Balogun</a>"
                "<br><b>Email</b>: hakim.obalogun@gmail.com<br><b>Contact</b>: +234 803 910 0897<br>"
                "<b>About</b>: </p>", unsafe_allow_html=True)

        col7, col8 = st.columns(2)  
        with col7:  
            st.image(harmony_image, width=243) 
        with col8:
            st.subheader("Hamony Odumuko")
            st.write("**Data Scientist**")
            st.markdown("<p><b>LinkedIn</b>: <a href='www.linkedin.com/in/hodumuko'>Harmony Odumuko</a>"
                "<br><b>Email</b>: nibotics@gmail.com<br><b>Contact</b>: +234 802 454 1916<br>"
                "<b>About</b>: </p>", unsafe_allow_html=True)

        col9, col10 = st.columns(2)    
        with col9:            
            st.image(endurance_image, width=243)
        with col10:
            st.subheader("Endurance Arienkhe")
            st.write("**Data Scientist**")
            st.markdown("<p><b>LinkedIn</b>: <a href='https://www.linkedin.com/in/endurance-arienkhe'>Endurance Arienkhe</a>"
                "<br><b>Email</b>: endurance.arienkhe@gmail.com<br><b>Contact</b>: +234 706 551 6048<br>"
                "<b>About</b>: </p>", unsafe_allow_html=True)   

        # st.subheader("More information")
        # if st.checkbox('Show contact information'): # data is hidden if box is unchecked
        #     st.info("francisikegwu@yahoo.com, kininiabigael@gmail.com, mamahchidike@gmail.com,icontola@gmail.com, vicmeleka@gmail.com, nibotics@gmail.com") 
        # with st.expander("Expand to see Company's video profile"):       
        #     video_file = open('Kinini.mp4', 'rb')
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)            
        # st.sidebar.subheader("Defining growth through data")            
        #     #st.success("Text Categorized as: {}".format(prediction))



if __name__ == '__main__':
    main()
