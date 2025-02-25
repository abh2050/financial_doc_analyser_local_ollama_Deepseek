import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key and secret from environment variables
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Define the Gemini API endpoint
url = "https://api.gemini.com/v1/pubticker/btcusd"

# Make a request to the Gemini API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    print("Gemini API is working. Response:")
    print(response.json())
else:
    print(f"Failed to connect to Gemini API. Status code: {response.status_code}")
    print(response.text)