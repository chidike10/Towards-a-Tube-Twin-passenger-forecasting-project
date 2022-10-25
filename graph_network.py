# Importing required libraries 
import streamlit as st 
import networkx as nx 
import colorsys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from bokeh.plotting import figure 

header = st.container() 
dataset = st.container() 
st.set_option('deprecation.showPyplotGlobalUse', False) 

lu_stations = pd.read_csv('Data/TfL-Station-Data-detailed/Transformed/Stations_Coodinates.csv', index_col=0) 
connections = pd.read_csv('Data/TfL-Station-Data-detailed/Transformed/LU_Loading_Data.csv') 
lu_lines = pd.read_csv('Data/TfL-Station-Data-detailed/Transformed/Wiki/Lines.csv', index_col=0) 

# Graph Network Files 

page_options = [
    "Landing Page", 
	"Graph Network", 
    "Explorative Data Analysis [EDA]", 
    "Passenger Forecast Modeling", 
    "About The Team" 
]

def main(): 
    st.sidebar.image('C:/Internship/EXPLORE-TubeTwinProject-Team6/Assets/LUnderground_Map_Snippet.png', use_column_width=True)
    page_selection = st.sidebar.selectbox("PAGE SELECTION", page_options)

    if page_selection == "Graph Network": 
        with header:
                st.title('The London Undergroung Graphical Network') 
                
                st.markdown('## Graphical Representation and Analyses of the Tube Network') 
                st.markdown('In this section of the project, we will be representing the London Tube rail network as a graph. And we think utilizing NetworkX will be the esiest way to achieve this. Nodes will be the stations and edges are the connections between them. We will make some analyses our graphs such as pageranking, calculating Hits, degree of centralities and inbetweeness etc.') 
                                        
                st.write("### Let's start by loading all needed dataframes") 
                                
                st.write('Showing top entries of the London Underground stations (aka Nodes).') 
                st.write(lu_stations.head(3)) 

                st.write('Showing top entries of the London Underground loading data, which will serve as the Connections between the stations (aka Edges).') 
                st.write(connections.head(3)) 
                
                st.write('Below are the London Underground Lines (aka Edge Labels).') 
                st.write(lu_lines) 

                st.markdown('### A simplified graph') 
                st.markdown('Now that we have our dataframes, we can create a simple graph of the network.')
                
                simple_graph = nx.Graph() 
                simple_graph.add_nodes_from(lu_stations['name']) 
                simple_graph.add_edges_from(list(zip(connections['from_station'], connections['to_station']))) 
                
                plt.figure(figsize =(8, 5)) 
                st.pyplot(nx.draw(simple_graph, node_size = 5)) 
                
                st.markdown('We can see from the above graph what the London Tube connections look like. Nodes which are ditached from the network are stations from the Stations_Coordinates.csv file that has no loading record in the LU_Loading_Data.csv file. \
                            Although this is not a realistic representation of the stations location compared to what they would look like on a geographical map.') 
                st.markdown('Already we can even do some analysing on the graph, like getting a reasonable (shortest) path between the stations `Oxford Circus` and `Canary Wharf`') 

                st.write(nx.shortest_path(simple_graph, 'Oxford Circus', 'Canary Wharf')) 

                st.markdown('Also we can run the PageRank and Hits algorithm on the network to messure the connections between the LU stations. Both of these compares the nodes (LU stations) using the numbers of connections found between them.')
                st.markdown("This time though, we'll focus on the stations that has connections between them as edges.") 

                graph = nx.Graph() 
                graph.add_edges_from(list(zip(connections['from_station'], connections['to_station']))) 

                pagerank = nx.pagerank_numpy(graph) 
                pagerank = pd.DataFrame(pagerank.items(), columns=['name', 'pagerank'])
                stations = pd.merge(lu_stations, pagerank, on='name') 

                st.write(stations.sort_values('pagerank', ascending=False).head(10)) 

                hits = nx.hits_scipy(graph, max_iter=1000)[0]
                hits = pd.DataFrame(hits.items(), columns=['name', 'hits'])
                stations = pd.merge(stations, hits, on='name') 

                st.write(stations.sort_values('hits', ascending=False).head(10)) 

                st.markdown('We show the top 10 station rank for both PageRank and Hits comparison above. Where PageRank finds the most important stations, the HITS algorithm seems to be pretty good at finding the busiest stations. To fully understand this, we can say the network relies on the important stations to function and without them, operations will be affected significantly. But the busiest stations does not impact the network operation in such significant way, they only tell us which stations has the highest traffic.') 
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
                    x_range = (.4,.7), 
                    y_range = (.2,.5), 
                    height= 700, 
                    width= 1100,
                    toolbar_location='above'  
                ) 

                # p.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool()) 

                for edge in graph.edges(): 
                    try: 
                        p.line( 
                            x= [locations[pt][0] for pt in edge],
                            y= [locations[pt][1] for pt in edge],
                        )
                    except KeyError:
                        pass 

                for node in graph.nodes():
                    try: 
                        x = [locations[node][0]]
                        y = [locations[node][1]]
                        p.circle( 
                            x, y, 
                            radius = .01 * pageranks[node], 
                            fill_color = pseudocolor(pageranks[node]), 
                            line_alpha=0) 
                        p.text(
                            x, y,                              
                            text = {'value':node}, 
                            text_font_size = str(min(pageranks[node] * 12, 10)) + "pt", 
                            text_alpha = pageranks[node],
                            text_align='center',
                            text_font_style='bold') 
                    except KeyError:
                        pass                 

                st.bokeh_chart(p) # Optional argument (use_container_width=True) 
                # show(p) 

                st.markdown('### Further analyses that can be done on this graph include the following:') 
                st.markdown('* Degree Centrality') 
                st.markdown('* Edge labeling and ranking') 
                st.markdown('* And more...') 

                st.markdown(' ### End of Analyses') 

#               st.markdown(' 
#               st.write(
#               st.pyplot(  

if __name__ == '__main__':
    main()

