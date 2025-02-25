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
    headers = {'User-Agent': 'YourAppName/1.0 (your_email@example.com)'}

    # Fetch the submissions data
    response = requests.get(submissions_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Filter filings based on form types and date range
    filing_data = []
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
        filing_data.append((filing_url, form_type, filing_date))

    return filing_data

def download_and_save_filing(url, form_type, filing_date, ticker, save_dir):
    """
    Download the filing from the given URL, parse its content, and save it with a structured filename.
    """
    headers = {'User-Agent': 'jaku/1.0 (abh2050@gmail.com)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from the filing
    text_content = soup.get_text(separator='\n', strip=True)

    # Construct a structured filename: TICKER_YYYYMMDD_FORMTYPE.htm
    formatted_date = filing_date.replace("-", "")
    filename = f"{ticker.upper()}_{formatted_date}_{form_type}.htm"
    file_path = os.path.join(save_dir, filename)

    # Save the content to a file
    with open(file_path, 'w') as file:
        file.write(text_content)

    return file_path

def main():
    ticker = input("Enter the company ticker symbol (e.g., AAPL): ").strip().lower()
    form_types = ['10-K', '10-Q', '8-K']  # Specify the form types
    start_year = int(input("Enter the start year (e.g., 2023): ").strip())
    end_year = int(input("Enter the end year (e.g., 2025): ").strip())

    cik = ticker_cik_mapping.get(ticker)
    if not cik:
        print(f"❌ CIK not found for ticker {ticker.upper()}. Please check the ticker symbol.")
        return

    print(f"🔍 Retrieving {', '.join(form_types)} filings for {ticker.upper()} from {start_year} to {end_year}...")
    filing_data = get_filing_urls(cik, form_types, start_year, end_year)

    if not filing_data:
        print("❌ No filings found for the specified criteria.")
        return

    print(f"✅ Found {len(filing_data)} filings. Downloading and saving...")

    # Dynamically create a directory for the ticker
    save_dir = os.path.join('/Users/abhishekshah/Desktop/financial_doc_analyser', ticker.upper())
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Directory created: {save_dir}")

    for url, form_type, filing_date in filing_data:
        print(f"📥 Processing: {url} ({form_type} on {filing_date})")
        file_path = download_and_save_filing(url, form_type, filing_date, ticker, save_dir)
        print(f"✔ Saved: {file_path}")

        time.sleep(0.2)  # Respectful delay between requests

if __name__ == "__main__":
    main()
