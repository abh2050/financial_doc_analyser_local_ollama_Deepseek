import streamlit as st
import ollama
import re
import os
import shutil
import tempfile
import time  # For simulating streaming

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from bs4 import BeautifulSoup  # HTML parsing
from langchain_core.documents import Document 

# Set directory where the files are stored
DOCUMENTS_DIR = "/Users/abhishekshah/Desktop/financial_doc_analyser/AAPL"

@st.cache_data  # Cache extracted documents
def load_all_documents():
    """Loads and processes all PDF and HTML files in the directory."""
    all_documents = []
    file_list = os.listdir(DOCUMENTS_DIR)
    
    for file_name in file_list:
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        if file_name.lower().endswith(".pdf"):
            all_documents.extend(process_pdf(file_path))
        elif file_name.lower().endswith((".htm", ".html")):
            all_documents.extend(process_html(file_path))

    if not all_documents:
        raise ValueError("No valid PDF or HTML files found in the directory.")

    return all_documents, file_list

@st.cache_resource  # Cache retriever so it persists across interactions
def build_vector_database():
    """Processes all files and creates a retriever for answering questions with a progress bar."""
    st.write("🚀 **Building VectorDatabase...**")
    
    # Initialize progress bar in Streamlit
    progress_bar = st.progress(0)

    documents, file_list = load_all_documents()
    total_files = max(len(file_list), 1)
    progress_step = 100 / total_files

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["line"] = i * 500  # Approximate line tracking

    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    temp_dir = tempfile.mkdtemp()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=temp_dir
    )
    retriever = vectorstore.as_retriever()

    # Update progress in Streamlit
    progress_bar.progress(100)

    return retriever, temp_dir

def process_pdf(pdf_path):
    """Processes a single PDF file and returns document chunks."""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def process_html(html_path):
    """Extracts text from an HTML file and returns a list of Document objects."""
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text(separator="\n")

    if not text.strip():
        raise ValueError(f"No readable text found in {html_path}.")

    return [Document(page_content=text, metadata={"source": html_path})]  # Convert to Document

def combine_docs_with_sources(docs):
    """Combine retrieved document chunks and attach source references."""
    combined_text = ""
    sources = []
    
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        text = doc.page_content.strip()
        line_number = doc.metadata.get("line", "N/A")  # Now tracks line numbers

        combined_text += f"{text}\n\n"
        sources.append(f"📄 {source}, Line {line_number}")

    return combined_text, sources

def ollama_llm(question, context):
    """Generates an answer using Ollama's LLM with enhanced financial analysis prompting."""
    
    formatted_prompt = f"""
    You are an expert financial document analyst specializing in SEC filings, including:
    - **10-K Reports** (Annual financial reports, risk factors, financial performance)
    - **10-Q Reports** (Quarterly earnings updates, cash flow statements, market conditions)
    - **8-K Reports** (Material events, executive changes, earnings surprises)

    Your task:
    1. Read and analyze the extracted financial data.
    2. Identify key financial insights, risks, trends, and important disclosures.
    3. Present the response in a structured financial format.
    
    Question: {question}
    
    **Relevant Context (Extracted Data from SEC Filings)**:
    {context}
    
    **Instructions for Response Formatting**:
    - Provide a **brief financial summary** (e.g., earnings, risks, business updates).
    - Identify **trends or irregularities** (e.g., revenue changes, executive turnover).
    - If applicable, include **quantitative insights** (e.g., financial ratios, YoY changes).
    - Reference the **most relevant sections** from the filings.

    **Your structured financial response should look like this**:
    ---
    📊 **Summary**: [Brief overview]
    
    📈 **Key Financial Metrics**:
    - Revenue Growth: [Data if available]
    - Profitability: [Data if available]
    - Cash Flow: [Data if available]
    
    🚨 **Risks & Disclosures**:
    - [List major risks found in the document]
    
    🔍 **Insights & Trends**:
    - [Mention any noteworthy patterns or executive changes]
    ---
    """

    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    full_output = response["message"]["content"]

    matches = re.findall(r"<think>(.*?)</think>", full_output, flags=re.DOTALL)
    chain_of_thought = "\n\n".join(match.strip() for match in matches) if matches else None

    final_answer = re.sub(r"<think>.*?</think>", "", full_output, flags=re.DOTALL).strip()

    return final_answer, chain_of_thought

def rag_chain(question, retriever):
    """Retrieve relevant document chunks and generate an answer with sources."""
    retrieved_docs = retriever.invoke(question)
    
    if not retrieved_docs:
        return "No relevant information found.", None, None

    formatted_content, sources = combine_docs_with_sources(retrieved_docs)
    answer, chain_of_thought = ollama_llm(question, formatted_content)
    
    return answer, chain_of_thought, sources

def ask_question(question):
    """
    Handles document processing and answering the user's question.
    Returns a tuple containing the final answer, chain-of-thought, and sources.
    """
    retriever, _ = build_vector_database()  # No need to reprocess every time
    answer, chain_of_thought, sources = rag_chain(question, retriever)
    return answer, chain_of_thought, sources

def clear_memory():
    """Clears session state and refreshes the page."""
    st.cache_data.clear()  # Clears cached documents
    st.cache_resource.clear()  # Clears retriever
    st.success("Memory cleared! Refreshing page...")
    st.markdown(
        """
        <script>
        setTimeout(function(){
            window.location.reload();
        }, 1000);
        </script>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.title("📄 Multi-File RAG: Financial Document Explorer")
    st.write(f"Processing files from: `{DOCUMENTS_DIR}`")

    if st.button("🧹 Clear Memory", key="clear_btn"):
        clear_memory()

    question = st.text_area("Your Question", placeholder="Type your question here...")

    if st.button("🚀 Ask Question"):
        if not question:
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("🔍 Analyzing documents and generating answer..."):
                answer, chain_of_thought, sources = ask_question(question)

            st.subheader("🧠 Chain-of-Thought Reasoning")
            st.write(chain_of_thought if chain_of_thought else "No chain-of-thought available.")

            st.subheader("💡 Answer")
            st.write(answer)

            st.subheader("📂 References")
            if sources:
                for ref in sources:
                    st.write(f"{ref}")
            else:
                st.write("No sources available.")

if __name__ == "__main__":
    main()
