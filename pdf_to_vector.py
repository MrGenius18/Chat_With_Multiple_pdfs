# from PyPDF2 import PdfReader
import pdfplumber


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

from clean_text import clean_text

# PyPDF2's may not work well for complex PDFs  so use pdfplumber (better for text-based PDFs)
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:

        with pdfplumber.open(pdf) as pdf_reader:

            for page in pdf_reader.pages:
                extracted_text = page.extract_text()

                if extracted_text:
                    text += extracted_text + "\n"

    if not text.strip():
        raise ValueError("⚠️ Error: No text found in the PDF! It might be an image-based PDF.")

    return text

# This ensures chunks contain full sentences, improving retrieval accuracy.
def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", ".", "?", "!"],  # Split at sentence boundaries
    )
    return text_splitter.split_text(text) 

# default (FAISS similarity search) may not always retrieve the most relevant chunk.
# Use Hybrid Search (BM25 + Embeddings) for better retrieval.
def get_vector_store(chunks):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    cleaned_chunks = [clean_text(chunk) for chunk in chunks]
    
    vector_store = FAISS.from_texts(cleaned_chunks, embedding=embeddings)
    bm25_retriever = BM25Retriever.from_texts(cleaned_chunks)     # Use BM25 for keyword-based retrieval

    vector_store.save_local("faiss_index")

    return vector_store, bm25_retriever
