# import requests 
# from tfl.client import Client 
# from tfl.api_token import ApiToken 
# import urllib.request, json 

# ====================================================================================

# app_id = ''
# app_key = '' 

# token = ApiToken(app_id, app_key) 

# client = Client(token)
# print (client.get_line_meta_modes())
# print (client.get_lines(mode="tube")[0]) 

# ====================================================================================

# try:
#     url = "https://api.tfl.gov.uk/crowding/940GZZLUACT/Live"

#     hdr ={
#     # Request headers
#     'Cache-Control': 'no-cache',
#     'app_key': '741afe7ce3794ee7b4fc413cfe352344', 
#     }

#     req = urllib.request.Request(url, headers=hdr)

#     req.get_method = lambda: 'GET'
#     response = urllib.request.urlopen(req)
#     print(response.getcode())
#     print(response.read())
# except Exception as e:
#     print(e) 

# ====================================================================================

# print(
#     "\n" +
#     "Which line do you want the status for?" +
#     "\n" +
#     "\n" + "Your options are:" +
#     "\n" +
#     "\n" + "District" +
#     "\n" + "Central" +
#     "\n" + "Circle" +
#     "\n" + "Piccadilly" +
#     "\n" + "Waterloo-City" +
#     "\n" + "Bakerloo" +
#     "\n" + "Hammersmith-City" +
#     "\n" + "Jubilee" +
#     "\n" + "Metropolitan" +
#     "\n" + "Victoria" +
#     "\n" + "Northern" +
#     "\n"
#     )
    
# line = input()

# reply = requests.get("https://api.tfl.gov.uk/Line/" + line + "/Status")

# data = reply.json()

# Status = (data[0]["lineStatuses"][0]["statusSeverityDescription"])

# print(
#     "\n" + "You have chosen: " + line +
#     "\n" + "Your status request: " + Status +
#     "\n"
#     ) 

# ====================================================================================

import pandas as pd
import glob
import os

# specifying the path to csv files
path = "../Data/QhrEntryExit-TubeData/"
 
# csv files in the path
file_list = glob.glob(path + "*.xlsx") 
 
# list of excel files we want to merge.
# pd.read_excel(file_path) reads the 
# excel data into pandas dataframe.
excl_list = []
 
for file in file_list:
    excl_list.append(pd.read_excel(file, sheet_name='ByQhr', header=6)) 
 
# concatenate all DataFrames in the list into a single DataFrame, returns new DataFrame.
excl_merged = pd.concat(excl_list, ignore_index=True) 
 
# exports the dataframe into excel file with specified name.
excl_merged.to_csv('../Data/historical_tube_data.csv', index=False) 
