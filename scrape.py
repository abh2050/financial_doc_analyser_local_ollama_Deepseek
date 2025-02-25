import requests
from datetime import datetime
from bs4 import BeautifulSoup
import time
import os

# Load the ticker-to-CIK mapping from the local file
ticker_cik_mapping = {}
with open('/Users/abhishekshah/Desktop/financial_doc_analyser/ticker_to_cik.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            ticker, cik = parts
            ticker_cik_mapping[ticker.lower()] = cik

def get_filing_urls(cik, form_types=None, start_year=None, end_year=None):
    """
    Retrieve a list of filing URLs for a given CIK and form types within the specified date range.
    """
    if not start_year:
        start_year = datetime.now().year - 1
    if not end_year:
        end_year = datetime.now().year

    # Construct the URL for the company's submissions
    submissions_url = f'https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json'
    headers = {
        'User-Agent': 'YourAppName/1.0 (your_email@example.com)'
    }

    # Fetch the submissions data
    response = requests.get(submissions_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Filter filings based on form types and date range
    filing_urls = []
    for form_type, filing_date, accession_number, primary_document in zip(
        data['filings']['recent']['form'],
        data['filings']['recent']['filingDate'],
        data['filings']['recent']['accessionNumber'],
        data['filings']['recent']['primaryDocument']
    ):
        filing_year = int(filing_date.split('-')[0])
        if form_types and form_type not in form_types:
            continue
        if not (start_year <= filing_year <= end_year):
            continue
        accession_number = accession_number.replace('-', '')
        filing_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_document}'
        filing_urls.append(filing_url)

    return filing_urls

def download_and_parse_filing(url, save_dir):
    """
    Download the filing from the given URL and parse its content.
    Save the content to a file in the specified directory.
    """
    headers = {
        'User-Agent': 'jaku/1.0 (abh2050@gmail.com)'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from the filing
    text_content = soup.get_text(separator='\n', strip=True)

    # Save the content to a file
    filename = os.path.join(save_dir, os.path.basename(url))
    with open(filename, 'w') as file:
        file.write(text_content)

    return text_content

def main():
    ticker = input("Enter the company ticker symbol (e.g., AAPL): ").strip().lower()
    form_types = ['10-K', '10-Q', '8-K']  # Specify the form types
    start_year = int(input("Enter the start year (e.g., 2023): ").strip())
    end_year = int(input("Enter the end year (e.g., 2025): ").strip())

    cik = ticker_cik_mapping.get(ticker)
    if not cik:
        print(f"CIK not found for ticker {ticker.upper()}")
        return

    print(f"Retrieving {', '.join(form_types)} filings for {ticker.upper()} from {start_year} to {end_year}...")
    filing_urls = get_filing_urls(cik, form_types, start_year, end_year)

    if not filing_urls:
        print("No filings found for the specified criteria.")
        return

    print(f"Found {len(filing_urls)} filings. Downloading and parsing...")

    # Create a directory to save the filings
    save_dir = os.path.join('/Users/abhishekshah/Desktop/financial_doc_analyser', ticker.upper())
    os.makedirs(save_dir, exist_ok=True)

    for url in filing_urls:
        print(f"Processing: {url}")
        content = download_and_parse_filing(url, save_dir)
        # Process the content as needed
        print(content[:500])  # Print the first 500 characters as a sample
        print("\n" + "-"*80 + "\n")
        time.sleep(0.2)  # Respectful delay between requests

if __name__ == "__main__":
    main()