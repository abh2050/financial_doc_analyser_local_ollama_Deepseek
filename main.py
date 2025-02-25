import streamlit as st
import requests
import ollama
import re
import os
import shutil
import tempfile
import time
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory for saving financial documents
BASE_DIR = os.path.expanduser("~/financial_doc_analyzer")
os.makedirs(BASE_DIR, exist_ok=True)

# Add a config file for user settings
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DEFAULT_CONFIG = {
    "user_agent": "YourCompanyName/1.0 (contact@example.com)",
    "models": {
        "embedding": "deepseek-r1:1.5b",
        "generation": "deepseek:latest"
    },
    "chunk_size": 500,
    "chunk_overlap": 100,
    "max_workers": 5,
    "cache_expiry_days": 7
}

def load_config():
    """Load or create configuration file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

CONFIG = load_config()

# Cache management
def clear_expired_cache():
    """Clear cache files older than configured expiry period"""
    cache_dir = os.path.join(BASE_DIR, "cache")
    if not os.path.exists(cache_dir):
        return
        
    expiry_days = CONFIG.get("cache_expiry_days", 7)
    expiry_time = datetime.now() - timedelta(days=expiry_days)
    
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_time < expiry_time:
                os.remove(file_path)
                logger.info(f"Removed expired cache file: {file_path}")

# Load ticker-to-CIK mapping with error handling and caching
@st.cache_data(ttl=3600*24)
def load_ticker_cik_mapping():
    """Load ticker-to-CIK mapping with cache and fallback"""
    mapping_file = os.path.join(BASE_DIR, "ticker_to_cik.txt")
    
    # Check if mapping file exists
    if not os.path.exists(mapping_file):
        # Create backup plan - download from SEC website
        try:
            st.info("Ticker-to-CIK mapping file not found. Downloading from SEC...")
            url = "https://www.sec.gov/include/ticker.txt"
            headers = {'User-Agent': CONFIG["user_agent"]}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Save the file
            os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
            with open(mapping_file, 'wb') as f:
                f.write(response.content)
                
            st.success("Ticker-to-CIK mapping downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download ticker mapping: {e}")
            # Create an empty mapping as fallback
            return {}
    
    ticker_cik_mapping = {}
    try:
        with open(mapping_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    ticker, cik = parts
                    ticker_cik_mapping[ticker.lower()] = cik.zfill(10)
        return ticker_cik_mapping
    except Exception as e:
        st.error(f"Error loading ticker-to-CIK mapping: {e}")
        return {}

def get_company_name(cik):
    """Get company name from CIK"""
    try:
        url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        headers = {'User-Agent': CONFIG["user_agent"]}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("name", "Unknown Company")
    except Exception as e:
        logger.error(f"Error getting company name: {e}")
        return "Unknown Company"

def get_filing_urls(cik, form_types, start_year, end_year, quarter=None):
    """Retrieve a list of SEC filing URLs for a given CIK and form types with better error handling."""
    try:
        submissions_url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        headers = {'User-Agent': CONFIG["user_agent"]}

        response = requests.get(submissions_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        filing_data = []
        
        # Process both recent and historical filings if available
        filings_data = data.get('filings', {})
        for filing_set in ['recent', 'files']:
            if filing_set not in filings_data:
                continue
                
            filing_set_data = filings_data[filing_set]
            
            if filing_set == 'recent':
                forms = filing_set_data.get('form', [])
                dates = filing_set_data.get('filingDate', [])
                accessions = filing_set_data.get('accessionNumber', [])
                primary_docs = filing_set_data.get('primaryDocument', [])
                
                for form_type, filing_date, accession_number, primary_document in zip(
                    forms, dates, accessions, primary_docs
                ):
                    filing_year = int(filing_date.split('-')[0])
                    if form_type not in form_types or not (start_year <= filing_year <= end_year):
                        continue
                        
                    if quarter:
                        filing_month = int(filing_date.split('-')[1])
                        filing_quarter = (filing_month - 1) // 3 + 1
                        if filing_quarter != quarter:
                            continue

                    accession_number = accession_number.replace('-', '')
                    filing_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_document}'
                    filing_data.append((filing_url, form_type, filing_date))
                    
            elif filing_set == 'files':
                for file_obj in filing_set_data:
                    file_name = file_obj.get('name')
                    if not file_name:
                        continue
                        
                    year_match = re.search(r'(\d{4})', file_name)
                    if not year_match:
                        continue
                        
                    year = int(year_match.group(1))
                    if not (start_year <= year <= end_year):
                        continue
                        
                    try:
                        file_url = f'https://data.sec.gov/submissions/{file_name}'
                        file_response = requests.get(file_url, headers=headers)
                        file_response.raise_for_status()
                        file_data = file_response.json()
                        
                        forms = file_data.get('form', [])
                        dates = file_data.get('filingDate', [])
                        accessions = file_data.get('accessionNumber', [])
                        primary_docs = file_data.get('primaryDocument', [])
                        
                        for form_type, filing_date, accession_number, primary_document in zip(
                            forms, dates, accessions, primary_docs
                        ):
                            if form_type not in form_types:
                                continue
                                
                            if quarter:
                                filing_month = int(filing_date.split('-')[1])
                                filing_quarter = (filing_month - 1) // 3 + 1
                                if filing_quarter != quarter:
                                    continue

                            accession_number = accession_number.replace('-', '')
                            filing_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_document}'
                            filing_data.append((filing_url, form_type, filing_date))
                    except Exception as e:
                        logger.error(f"Error processing filing file {file_name}: {e}")

        return filing_data
        
    except Exception as e:
        st.error(f"Error retrieving filings: {e}")
        logger.error(f"Error retrieving filings for CIK {cik}: {e}")
        return []

def download_and_save_filing(url, form_type, filing_date, ticker, save_dir):
    """Download an SEC filing, parse its content, and save it with a structured filename."""
    try:
        headers = {'User-Agent': CONFIG["user_agent"]}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        is_pdf = 'pdf' in content_type or url.lower().endswith('.pdf')
        
        formatted_date = filing_date.replace("-", "")
        
        if is_pdf:
            filename = f"{ticker.upper()}_{formatted_date}_{form_type}.pdf"
            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'wb') as file:
                file.write(response.content)
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup.find_all(['script', 'style']):
                tag.decompose()
                
            text_content = soup.get_text(separator='\n', strip=True)
            filename = f"{ticker.upper()}_{formatted_date}_{form_type}.htm"
            file_path = os.path.join(save_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text_content)

        logger.info(f"Successfully downloaded and saved: {filename}")
        return file_path
    except Exception as e:
        logger.error(f"Error downloading filing {url}: {e}")
        st.error(f"Failed to download {url}: {str(e)}")
        return None

def download_filings_parallel(filings, ticker, save_dir):
    """Download multiple filings in parallel with progress tracking"""
    successful_downloads = 0
    failed_downloads = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    max_workers = CONFIG.get("max_workers", 5)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for url, form_type, filing_date in filings:
            future = executor.submit(
                download_and_save_filing, 
                url, form_type, filing_date, ticker, save_dir
            )
            futures.append(future)
        
        total = len(futures)
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
            except Exception as e:
                logger.error(f"Error in download thread: {e}")
                failed_downloads += 1
                
            progress = min(int(100 * (i + 1) / total), 100)
            progress_bar.progress(progress)
            status_text.text(f"Downloaded {i+1}/{total} filings...")
    
    progress_bar.empty()
    status_text.empty()
    
    return successful_downloads, failed_downloads

def extract_financial_metrics(text):
    """Attempt to extract key financial metrics from filing text"""
    metrics = {}
    
    revenue_patterns = [
        r'total revenue.*?\$([\d,\.]+)\s+(?:million|billion|thousand)',
        r'revenue.*?\$([\d,\.]+)\s+(?:million|billion|thousand)',
        r'net sales.*?\$([\d,\.]+)\s+(?:million|billion|thousand)'
    ]
    
    for pattern in revenue_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            metrics['revenue'] = matches.group(1)
            break
    
    income_patterns = [
        r'net income.*?\$([\d,\.]+)\s+(?:million|billion|thousand)',
        r'income.*?\$([\d,\.]+)\s+(?:million|billion|thousand)'
    ]
    
    for pattern in income_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            metrics['net_income'] = matches.group(1)
            break
    
    eps_patterns = [
        r'earnings per share.*?\$([\d,\.]+)',
        r'diluted earnings per share.*?\$([\d,\.]+)',
        r'EPS.*?\$([\d,\.]+)'
    ]
    
    for pattern in eps_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            metrics['eps'] = matches.group(1)
            break
    
    return metrics

def process_pdf(pdf_path):
    """Processes a PDF file and returns document chunks."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        full_text = "\n".join([doc.page_content for doc in documents])
        metrics = extract_financial_metrics(full_text)
        
        for doc in documents:
            doc.metadata.update({
                "source_type": "pdf",
                "financial_metrics": metrics
            })
            
        return documents
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return [Document(
            page_content=f"Error processing file: {str(e)}",
            metadata={"source": pdf_path, "error": str(e)}
        )]

def process_html(html_path):
    """Extracts text from an HTML file."""
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text(separator="\n")
    return [Document(page_content=text, metadata={"source": html_path})]

def load_all_documents(directory):
    """Loads and processes all PDF and HTML files in the directory."""
    all_documents = []
    file_list = os.listdir(directory)

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if file_name.lower().endswith(".pdf"):
            all_documents.extend(process_pdf(file_path))
        elif file_name.lower().endswith((".htm", ".html")):
            all_documents.extend(process_html(file_path))

    if not all_documents:
        raise ValueError("No valid PDF or HTML files found.")

    return all_documents, file_list

@st.cache_data
def load_all_documents(directory):
    """Loads and processes all PDF and HTML files in the directory with progress tracking."""
    all_documents = []
    file_paths = []
    
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
        
    file_list = os.listdir(directory)
    total_files = len(file_list)
    
    if total_files == 0:
        raise ValueError(f"No files found in directory: {directory}")
    
    st.write(f"📊 Processing {total_files} files...")
    progress_bar = st.progress(0)
    
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(directory, file_name)
        file_paths.append(file_path)
        
        if file_name.lower().endswith(".pdf"):
            all_documents.extend(process_pdf(file_path))
        elif file_name.lower().endswith((".htm", ".html")):
            all_documents.extend(process_html(file_path))
        
        progress = min(int(100 * (i + 1) / total_files), 100)
        progress_bar.progress(progress)
    
    if not all_documents:
        raise ValueError("No valid PDF or HTML files found.")
    
    metrics_found = {}
    for doc in all_documents:
        if "financial_metrics" in doc.metadata and doc.metadata["financial_metrics"]:
            for k, v in doc.metadata["financial_metrics"].items():
                if k not in metrics_found:
                    metrics_found[k] = []
                metrics_found[k].append(v)
    
    return all_documents, file_list, metrics_found

def build_vector_database(directory):
    """Processes files and builds the Vector Database with a progress bar."""
    st.write("🚀 **Building VectorDatabase...**")
    progress_bar = st.progress(0)

    try:
        # Updated to handle three return values
        documents, file_list, metrics = load_all_documents(directory)
        total_files = max(len(file_list), 1)
        progress_step = 100 / total_files

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.get("chunk_size", 500),
            chunk_overlap=CONFIG.get("chunk_overlap", 100)
        )
        chunks = text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["line"] = i * CONFIG.get("chunk_size", 500)
            progress_bar.progress(min(int(progress_step * (i+1)), 100))

        embedding_model = CONFIG.get("models", {}).get("embedding", "deepseek-r1:1.5b")
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        temp_dir = tempfile.mkdtemp()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=temp_dir
        )
        retriever = vectorstore.as_retriever()

        progress_bar.progress(100)
        return retriever, temp_dir, metrics  # Return metrics instead of empty dict

    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        raise

def rag_chain(question, retriever, company_name):
    """Retrieve relevant document chunks and generate an answer with improved context handling."""
    try:
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            return "No relevant information found in the documents.", None

        context_parts = []
        sources = set()
        
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "Unknown")
            source_type = doc.metadata.get("source_type", "unknown")
            line = doc.metadata.get("line", 0)
            
            if source != "Unknown":
                source_filename = os.path.basename(source)
                sources.add(source_filename)
            
            context_parts.append(f"[DOCUMENT {i+1}] Source: {source_filename if source != 'Unknown' else 'Unknown'}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        source_list = ", ".join(list(sources))

        formatted_prompt = f"""
        You are an expert financial document analyst specializing in SEC filings, including:
        - **10-K Reports** (Annual financial reports, risk factors, financial performance)
        - **10-Q Reports** (Quarterly earnings updates, cash flow statements, market conditions)
        - **8-K Reports** (Material events, executive changes, earnings surprises)

        Your task is to analyze the extracted financial data and provide structured financial insights.

        **Question:** {question}

        **Relevant Context:**
        {context}

        **Sources:** {source_list}

        Please provide a clear and concise answer focusing on:
        1. Direct response to the question
        2. Key financial metrics and data points
        3. Important trends or patterns
        4. Citations to specific documents

        Format your response in clear, structured paragraphs.
        """

        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[{"role": "user", "content": formatted_prompt}]
        )

        return response["message"]["content"], retrieved_docs

    except Exception as e:
        logger.error(f"Error in RAG chain: {e}")
        return f"An error occurred while processing your question: {str(e)}", None

def compare_filings(directory, form_type=None):
    """Compare filings over time to identify trends"""
    try:
        files = os.listdir(directory)
        
        if form_type:
            files = [f for f in files if form_type in f]
        
        filing_data = []
        for file in files:
            if not (file.endswith('.htm') or file.endswith('.html') or file.endswith('.pdf')):
                continue
                
            parts = file.split('_')
            if len(parts) >= 3:
                ticker = parts[0]
                date_str = parts[1]
                form = parts[2].split('.')[0]
                
                try:
                    date = datetime.strptime(date_str, "%Y%m%d")
                    file_path = os.path.join(directory, file)
                    
                    if file.endswith('.pdf'):
                        docs = process_pdf(file_path)
                    else:
                        docs = process_html(file_path)
                        
                    full_text = "\n".join([doc.page_content for doc in docs])
                    metrics = extract_financial_metrics(full_text)
                    
                    filing_data.append({
                        'ticker': ticker,
                        'date': date,
                        'form': form,
                        'file': file,
                        'metrics': metrics
                    })
                except:
                    continue
        
        filing_data.sort(key=lambda x: x['date'])
        
        rows = []
        for filing in filing_data:
            row = {
                'Date': filing['date'].strftime('%Y-%m-%d'),
                'Form': filing['form'],
                'File': filing['file']
            }
            
            for metric, value in filing['metrics'].items():
                row[metric.capitalize()] = value
                
            rows.append(row)
            
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error comparing filings: {e}")
        return pd.DataFrame()

def main():
    global CONFIG
    st.set_page_config(
        page_title="SEC Financial Document Explorer",
        page_icon="📄",
        layout="wide"
    )
    
    # Remove all custom CSS styling and let Streamlit use its default styling
    
    # Title and description
    st.title("📄 SEC Financial Document Explorer")
    st.markdown("""
    Analyze SEC filings (10-K, 10-Q, 8-K) for any publicly traded company.
    Download reports, process them with AI, and gain financial insights.
    """)
    
    # Clear expired cache
    clear_expired_cache()
    
    # Create tabs for different functionalities
    tabs = st.tabs(["📥 Download Filings", "🔍 Analyze Documents", "📊 Compare Filings", "⚙️ Settings"])
    
    # Load ticker-to-CIK mapping
    ticker_cik_mapping = load_ticker_cik_mapping()
    
    with tabs[0]:  # Download Filings tab
        st.header("Download SEC Filings")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", key="download_ticker").strip().upper()
            
            date_col1, date_col2 = st.columns(2)
            
            with date_col1:
                start_year = st.number_input(
                    "Start Year:", 
                    min_value=2000, 
                    max_value=datetime.now().year, 
                    value=datetime.now().year - 1,
                    key="start_year"
                )
                
                use_quarter_filter = st.checkbox("Filter by Quarter", key="use_quarter")
                
                if use_quarter_filter:
                    quarter = st.selectbox(
                        "Quarter:", 
                        options=[1, 2, 3, 4],
                        format_func=lambda x: f"Q{x}",
                        key="quarter"
                    )
                else:
                    quarter = None
                
            with date_col2:
                end_year = st.number_input(
                    "End Year:", 
                    min_value=2000, 
                    max_value=datetime.now().year, 
                    value=datetime.now().year,
                    key="end_year"
                )
                
                form_types = st.multiselect(
                    "Filing Types:", 
                    options=['10-K', '10-Q', '8-K', '20-F', '6-K', 'DEF 14A'],
                    default=['10-K', '10-Q', '8-K'],
                    key="form_types"
                )
            
        with col2:
            st.subheader("Company Info")
            if ticker:
                cik = ticker_cik_mapping.get(ticker.lower())
                if cik:
                    company_name = get_company_name(cik)
                    st.success(f"Found: {company_name}")
                    st.write(f"CIK: {cik}")
                    
                    st.session_state["company_name"] = company_name
                    st.session_state["current_cik"] = cik
                    st.session_state["current_ticker"] = ticker
                else:
                    st.error(f"CIK not found for ticker {ticker}")
        
        if st.button("📥 Download SEC Filings", key="download_btn"):
            if not ticker:
                st.error("Please enter a ticker symbol")
            else:
                cik = ticker_cik_mapping.get(ticker.lower())
                if not cik:
                    st.error(f"❌ CIK not found for ticker {ticker}.")
                    st.info("Please check if the ticker is correct or if the company is publicly traded.")
                    return

                with st.spinner(f"Retrieving filing list for {ticker}..."):
                    filing_data = get_filing_urls(cik, form_types, start_year, end_year, quarter)
                
                if not filing_data:
                    st.warning("❌ No filings found matching your criteria.")
                    return

                st.success(f"Found {len(filing_data)} filings to download.")
                
                filing_df = pd.DataFrame([
                    {"Type": f_type, "Date": f_date, "URL": url} 
                    for url, f_type, f_date in filing_data
                ])
                st.dataframe(filing_df)

                company_dir = os.path.join(BASE_DIR, ticker)
                os.makedirs(company_dir, exist_ok=True)

                st.info("📥 Downloading filings...")
                successful, failed = download_filings_parallel(filing_data, ticker, company_dir)

                if successful > 0:
                    st.success(f"✅ Successfully downloaded {successful} filings!")
                    st.session_state["download_dir"] = company_dir
                if failed > 0:
                    st.error(f"❌ Failed to download {failed} filings.")

    with tabs[1]:  # Analyze Documents tab
        st.header("Analyze Downloaded Filings")

        if "download_dir" not in st.session_state:
            st.warning("Please download filings first.")
        else:
            company_dir = st.session_state["download_dir"]
            company_name = st.session_state.get("company_name", "Unknown Company")

            if st.button("🔍 Build/Refresh Vector Database", key="build_vector_db"):
                with st.spinner("Building vector database..."):
                    try:
                        retriever, persist_dir, metrics = build_vector_database(company_dir)
                        st.session_state["retriever"] = retriever
                        st.session_state["metrics"] = metrics
                        st.success("Vector database built successfully!")
                    except Exception as e:
                        st.error(f"Error building vector database: {e}")

            if "retriever" in st.session_state:
                st.subheader("Ask Questions About the Filings")
                question = st.text_area("Enter your question:", height=100)

                if st.button("🔍 Get Answer", key="get_answer"):
                    if not question.strip():
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Analyzing documents..."):
                            answer, docs = rag_chain(question, st.session_state["retriever"], company_name)
                            st.markdown("### Answer")
                            st.markdown(answer)

                            if docs:
                                st.markdown("### Relevant Documents")
                                for i, doc in enumerate(docs):
                                    source = doc.metadata.get("source", "Unknown")
                                    st.markdown(f"**Document {i+1}:** {os.path.basename(source)}")
                                    st.text(doc.page_content[:500] + "...")

                if "metrics" in st.session_state and st.session_state["metrics"]:
                    st.subheader("Extracted Financial Metrics")
                    metrics_df = pd.DataFrame(st.session_state["metrics"])
                    st.dataframe(metrics_df)

    with tabs[2]:  # Compare Filings tab
        st.header("Compare Filings Over Time")

        if "download_dir" not in st.session_state:
            st.warning("Please download filings first.")
        else:
            company_dir = st.session_state["download_dir"]
            form_type = st.selectbox(
                "Select Form Type to Compare:",
                options=['10-K', '10-Q', '8-K', '20-F', '6-K', 'DEF 14A'],
                key="compare_form_type"
            )

            if st.button("📊 Compare Filings", key="compare_filings"):
                with st.spinner("Comparing filings..."):
                    comparison_df = compare_filings(company_dir, form_type)
                    if not comparison_df.empty:
                        st.success("Comparison complete!")
                        st.dataframe(comparison_df)

                        if 'Revenue' in comparison_df.columns:
                            st.line_chart(comparison_df.set_index('Date')['Revenue'])
                        if 'Net_income' in comparison_df.columns:
                            st.line_chart(comparison_df.set_index('Date')['Net_income'])
                    else:
                        st.warning("No filings found for comparison.")

    with tabs[3]:  # Settings tab
        st.header("Settings")

        st.subheader("Configuration")
        st.json(CONFIG)  # Safe to use since global CONFIG is declared at the top

        if st.button("🔄 Reload Configuration", key="reload_config"):
            CONFIG = load_config()  # Update the global CONFIG variable
            st.success("Configuration reloaded!")

        st.subheader("Cache Management")
        if st.button("🗑️ Clear Expired Cache", key="clear_cache"):
            clear_expired_cache()
            st.success("Expired cache cleared!")

        if st.button("🧹 Clear All Cache", key="clear_all_cache"):
            cache_dir = os.path.join(BASE_DIR, "cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                st.success("All cache cleared!")
            else:
                st.info("No cache directory found.")

if __name__ == "__main__":
    main()
