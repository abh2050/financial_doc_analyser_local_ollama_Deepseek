# SEC Financial Document Explorer

A Streamlit-based application designed to download, process, and analyze SEC filings (e.g., 10-K, 10-Q, 8-K). This tool leverages AI-powered embeddings, vector databases, and advanced document processing techniques to provide financial insights and enable trend comparisons over time.

---
## Why Local RAG is the Future

- Retrieval-Augmented Generation (RAG) is revolutionizing how enterprises interact with their data, enabling more precise and context-aware responses. Local RAG offers several advantages:
- Data Privacy & Security: Running the model locally ensures that sensitive financial documents never leave the system, protecting against leaks and compliance risks.
- Faster Querying: Since data retrieval happens on-premise, response times are significantly lower compared to cloud-based alternatives.
- Customization & Control: Businesses can fine-tune embeddings, chunk sizes, and document retrieval strategies without vendor lock-in.
- Reduced Cloud Costs: Eliminating dependence on cloud API calls results in significant cost savings, especially for high-volume queries.

## Speeding Up Local RAG with GPUs & Parallelization

To enhance performance, Local RAG implementations can be accelerated using:

- High-Performance GPUs: Running embedding generation and vector searches on modern GPUs (e.g., NVIDIA A100, H100) dramatically improves inference speeds.
- Parallel Processing: Using multi-threading and distributed computing across multiple GPUs or CPU cores enables batch processing of filings and retrieval queries.
- Optimized Indexing: Advanced vector databases like FAISS and ChromaDB support GPU-based acceleration, further reducing search latency.
- Memory Mapping Techniques: Keeping embeddings in shared memory allows for faster access, minimizing disk I/O bottlenecks.

By leveraging these techniques, SEC Financial Document Explorer ensures scalable and efficient document retrieval, making financial analysis faster and more secure.

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
![](https://github.com/abh2050/financial_doc_analyser_local_ollama_Deepseek/blob/main/demo/pic1.png)
![](https://github.com/abh2050/financial_doc_analyser_local_ollama_Deepseek/blob/main/demo/pic2.png)
![](https://github.com/abh2050/financial_doc_analyser_local_ollama_Deepseek/blob/main/demo/pic3.png)
![](https://github.com/abh2050/financial_doc_analyser_local_ollama_Deepseek/blob/main/demo/pic4.png)
![](https://github.com/abh2050/financial_doc_analyser_local_ollama_Deepseek/blob/main/demo/pic5.png)
![](https://github.com/abh2050/financial_doc_analyser_local_ollama_Deepseek/blob/main/demo/pic6.png)


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

- **Ollama CLI:** Required to run DeepSeek models.
- **DeepSeek Model:** AI model for processing and analyzing documents.

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

3. **Install Ollama CLI:**
   - Visit the [Ollama website](https://ollama.com) and follow the installation instructions for your operating system.

4. **Download DeepSeek Model:**
   - After installing the Ollama CLI, download the DeepSeek model by running:
     ```bash
     ollama run deepseek-r1:671b
     ```
   - This command will download the DeepSeek-R1 model, which is essential for processing and analyzing documents within the application.

5. **Set Up Configuration:**  
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
   streamlit run main1.py
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
├── main1.py          # Main Streamlit application
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

