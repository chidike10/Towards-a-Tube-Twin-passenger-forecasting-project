# We first import the packages needed to make an API call. 
import json # <-- Used to parse the JSON files we receive from the API 
import csv # Used to write the parsed response into a CSV file 
import urllib.request # <-- Used to... 

# =============================================================
# Fill in your details required to run the script 
source_url = 'https://api.tfl.gov.uk/' 
app_id = 'TubeTwin'
app_key = '741afe7ce3794ee7b4fc413cfe352344' 

# =============================================================

# Here we create a function that takes in our credentials to make the request using `urllib`.
def get_tfl_data(): 
    """Retrieves the a specified tube data using the TfL API.  
    Args:
        url (string): url link to the open data on tfl website. 
        pry_key (string): the required app_key to gain access to tube data. 
    Returns:
        json object: the required tube data from tfl website.
    """   
    url = source_url + 'Line/Mode/tube?app_id={app_id}&app_key={app_key}' 
    try:        
        print(f'Calling the Tfl API with the following URL: {url}')
        
        # Use urllib to perform a GET request to the API using our formatted URL.
        with urllib.request.urlopen(url) as url: 
            # Load and decode the resulting JSON data received from the API.
            data = json.loads(url.read().decode()) 
    except Exception as e: 
        print (f"Something went wrong in contacting the API service.\n The following exception was obtained:{e}") 
        data = None
    return data 

def save_tfl_data(json_data): 
    """Function to save the TfL response data as a CSV file. 
    Args:
        json_data (json object): The JSON data payload received from the Tfl API
    """ 
    data_file = open('C:\Internship\EXPLORE-TubeTwinProject-Team6\Data\Output\sample_data.csv', 'w', newline='')
    csv_writer = csv.writer(data_file) 
    
    count = 0
    for data in json_data:
        if count == 0:
            header = data.keys()
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(data.values())
    
    data_file.close() 

    print ("*** JSON data successfully saved as CSV file! ***")

def main():
    response = get_tfl_data() 
    if response is not None:
        print ("***API Call Successful! The Following data was received*** \n")
        print(response)
        print ("\n")
        save_tfl_data(response) 
        print ("\n")

if __name__ == "__main__":
    main() 