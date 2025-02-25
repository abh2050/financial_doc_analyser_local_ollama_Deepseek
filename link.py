import requests

# URL of the SEC's ticker-to-CIK mapping file
url = 'https://www.sec.gov/include/ticker.txt'

# Define headers with a proper User-Agent
headers = {
    'User-Agent': 'jaku/1.0 (abh2050@gmail.com)'
}

# Send a GET request to fetch the content of the file
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Decode the content to a string
    data = response.content.decode('utf-8')
    
    # Specify the filename to save the data
    filename = 'ticker_to_cik.txt'
    
    # Write the data to a file
    with open(filename, 'w') as file:
        file.write(data)
    
    print(f"Data successfully saved to {filename}")
else:
    print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
