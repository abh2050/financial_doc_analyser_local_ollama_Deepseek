import streamlit as st
import requests
import ollama
import re
import os
import shutil
import tempfile
import time
from datetime import datetime
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Base directory for saving financial documents
BASE_DIR = "/Users/abhishekshah/Desktop/financial_doc_analyser"

# Load ticker-to-CIK mapping
ticker_cik_mapping = {}
with open(os.path.join(BASE_DIR, "ticker_to_cik.txt"), 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            ticker, cik = parts
            ticker_cik_mapping[ticker.lower()] = cik

def get_filing_urls(cik, form_types, start_year, end_year):
    """Retrieve a list of SEC filing URLs for a given CIK and form types."""
    submissions_url = f'https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json'
    headers = {'User-Agent': 'YourAppName/1.0 (your_email@example.com)'}

    response = requests.get(submissions_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    filing_data = []
    for form_type, filing_date, accession_number, primary_document in zip(
        data['filings']['recent']['form'],
        data['filings']['recent']['filingDate'],
        data['filings']['recent']['accessionNumber'],
        data['filings']['recent']['primaryDocument']
    ):
        filing_year = int(filing_date.split('-')[0])
        if form_type not in form_types or not (start_year <= filing_year <= end_year):
            continue

        accession_number = accession_number.replace('-', '')
        filing_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_document}'
        filing_data.append((filing_url, form_type, filing_date))

    return filing_data

def download_and_save_filing(url, form_type, filing_date, ticker, save_dir):
    """Download an SEC filing, parse its content, and save it with a structured filename."""
    headers = {'User-Agent': 'jaku/1.0 (abh2050@gmail.com)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    text_content = soup.get_text(separator='\n', strip=True)

    formatted_date = filing_date.replace("-", "")
    filename = f"{ticker.upper()}_{formatted_date}_{form_type}.htm"
    file_path = os.path.join(save_dir, filename)

    with open(file_path, 'w') as file:
        file.write(text_content)

    return file_path

@st.cache_data
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

@st.cache_resource
def build_vector_database(directory):
    """Processes files and builds the Vector Database with a progress bar."""
    st.write("🚀 **Building VectorDatabase...**")
    progress_bar = st.progress(0)

    documents, file_list = load_all_documents(directory)
    total_files = max(len(file_list), 1)
    progress_step = 100 / total_files

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["line"] = i * 500
        progress_bar.progress(min(int(progress_step * (i+1)), 100))

    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    temp_dir = tempfile.mkdtemp()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=temp_dir
    )
    retriever = vectorstore.as_retriever()

    progress_bar.progress(100)
    return retriever, temp_dir

def process_pdf(pdf_path):
    """Processes a PDF file and returns document chunks."""
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()

def process_html(html_path):
    """Extracts text from an HTML file."""
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text(separator="\n")
    return [Document(page_content=text, metadata={"source": html_path})]

def rag_chain(question, retriever):
    """Retrieve relevant document chunks and generate an answer."""
    retrieved_docs = retriever.invoke(question)
    
    if not retrieved_docs:
        return "No relevant information found.", None

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    formatted_prompt = f"""
    You are an expert financial document analyst specializing in SEC filings, including:
    - **10-K Reports** (Annual financial reports, risk factors, financial performance)
    - **10-Q Reports** (Quarterly earnings updates, cash flow statements, market conditions)
    - **8-K Reports** (Material events, executive changes, earnings surprises)

    Your task is to analyze the extracted financial data and provide structured financial insights.

    **Question:** {question}

    **Relevant Context:**
    {context}

    Provide a structured financial summary including key insights.
    """

    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": formatted_prompt}]
    )

    return response["message"]["content"], retrieved_docs

def main():
    st.title("📄 SEC Financial Document Explorer")

    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):").strip().lower()
    start_year = st.number_input("Start Year:", min_value=2000, max_value=datetime.now().year, value=datetime.now().year - 1)
    end_year = st.number_input("End Year:", min_value=2000, max_value=datetime.now().year, value=datetime.now().year)

    if st.button("📥 Download SEC Filings"):
        cik = ticker_cik_mapping.get(ticker)
        if not cik:
            st.error(f"❌ CIK not found for ticker {ticker.upper()}.")
            return

        form_types = ['10-K', '10-Q', '8-K']
        save_dir = os.path.join(BASE_DIR, ticker.upper())
        os.makedirs(save_dir, exist_ok=True)

        filing_data = get_filing_urls(cik, form_types, start_year, end_year)
        if not filing_data:
            st.error("❌ No filings found.")
            return

        for url, form_type, filing_date in filing_data:
            st.write(f"📥 Downloading: {url}")
            download_and_save_filing(url, form_type, filing_date, ticker, save_dir)

        st.success(f"✔ Downloaded {len(filing_data)} filings to `{save_dir}`.")

    if st.button("🚀 Build VectorDatabase"):
        retriever, _ = build_vector_database(os.path.join(BASE_DIR, ticker.upper()))
        st.session_state["retriever"] = retriever
        st.success("✔ VectorDatabase built successfully!")

    if "retriever" in st.session_state:
        question = st.text_area("Ask a question about SEC filings:")
        if st.button("🔍 Get Answer"):
            answer, _ = rag_chain(question, st.session_state["retriever"])
            st.subheader("💡 Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
