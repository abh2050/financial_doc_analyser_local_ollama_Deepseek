# SEC Financial Document Explorer

A Streamlit-based application designed to download, process, and analyze SEC filings (e.g., 10-K, 10-Q, 8-K). This tool leverages AI-powered embeddings, vector databases, and advanced document processing techniques to provide financial insights and enable trend comparisons over time.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [File Download](#file-download)
  - [Document Processing](#document-processing)
  - [Vector Database & Local RAG](#vector-database--local-rag)
  - [Retrieval Function](#retrieval-function)
- [File Structure](#file-structure)
- [License](#license)

---

## Overview

The SEC Financial Document Explorer is a comprehensive tool that:
- Downloads SEC filings from the SEC website.
- Processes and extracts financial metrics from PDF and HTML documents.
- Builds a vector database using document embeddings for advanced retrieval.
- Supports a retrieval-augmented generation (RAG) chain to answer user questions.
- Compares filings over time to highlight trends in financial performance.

Using a combination of Python libraries (Streamlit, Requests, BeautifulSoup, LangChain, and others), the application offers an interactive interface for analyzing financial documents and obtaining structured insights.

---

## Features

- **Download SEC Filings:**  Retrieves filings based on ticker symbols, date ranges, and optional quarter filtering.
- **Document Processing:**  Parses PDFs and HTML files to extract key financial metrics like revenue, net income, and EPS.
- **Vector Database Construction:**  Builds a searchable vector database from processed document chunks.
- **Question Answering (RAG Chain):**  Uses AI to generate structured answers based on relevant document sections.
- **Filing Comparison:**  Compares multiple filings over time, displaying trends with visual charts.
- **Cache and Configuration Management:**  Implements caching mechanisms and customizable settings.

---

## Requirements

- **Python 3.7+**
- **Key Libraries:**
  - `streamlit`
  - `requests`
  - `ollama`
  - `pandas`
  - `beautifulsoup4`
  - `langchain_community`
  - `langchain_chroma`
  - `langchain_core`
  - `tqdm`
  - `concurrent.futures`
  
Install required packages via pip:
```bash
pip install streamlit requests ollama pandas beautifulsoup4 langchain_community langchain_chroma tqdm
```

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/sec-financial-document-explorer.git
   cd sec-financial-document-explorer
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Configuration:**  
   The application creates `config.json` in `~/financial_doc_analyzer` on first run. Adjust settings as needed.

---

## Configuration

- **Configuration File:**
  Located at `~/financial_doc_analyzer/config.json`, containing settings like:
  - **User Agent** for SEC requests.
  - **Model Settings** for embeddings and generation.
  - **Cache Expiry** duration (in days).

- **Cache Management:**
  - Expired cache files are automatically cleared.
  - Users can manually clear cache via the app's Settings tab.

---

## Usage

1. **Run the Application:**
   ```bash
   streamlit run your_script_name.py
   ```

2. **Navigate the Tabs:**
   - **Download Filings:** Enter a ticker, select filing types and date range, and download SEC filings.
   - **Analyze Documents:** Build or refresh the vector database and ask financial-related questions.
   - **Compare Filings:** View financial trends over time using visual charts.
   - **Settings:** Manage configuration and cache settings.

---

## How It Works

### File Download

- Retrieves SEC filings from the official SEC website using requests.
- Ticker-to-CIK mapping is used to fetch the correct company filings.
- Downloads filings in either PDF or HTML format and stores them locally.
- Uses multi-threading to accelerate the downloading process.
- Implements retry mechanisms and error handling to manage failed requests.

### Document Processing

- Uses `BeautifulSoup` to parse HTML filings and extract financial metrics.
- Converts PDFs using `PyMuPDFLoader` to extract text and structure the data.
- Extracts financial metrics (revenue, net income, EPS) using regex-based patterns.
- Cleans extracted text to remove unnecessary symbols and formatting issues.

### Vector Database & Local RAG

- Utilizes `langchain_chroma` to store embeddings of document chunks.
- Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- Uses `OllamaEmbeddings` to generate embeddings for efficient retrieval.
- Implements a retrieval-augmented generation (RAG) pipeline for querying document information.
- Allows users to search filings using natural language queries.

### Retrieval Function

- Queries the vector database to retrieve relevant document chunks.
- Uses Ollama’s AI models for generating responses based on retrieved data.
- Provides citations and sources from SEC filings in responses.
- Implements ranking mechanisms to prioritize the most relevant document chunks.

---

## File Structure

```
├── your_script_name.py          # Main Streamlit application
├── config.json                  # Configuration file (created on first run)
├── README.md                    # Project documentation (this file)
└── requirements.txt             # Python dependencies
```

- **Main Application:** Integrates SEC data retrieval, AI processing, and interactive UI.
- **Caching:** Uses Streamlit's caching functions (`@st.cache_data`) for performance optimization.
- **Vector Database:** Implements LangChain and OllamaEmbeddings for efficient document search.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Maintained by [YourCompanyName]. Contact [contact@example.com] for inquiries.*

