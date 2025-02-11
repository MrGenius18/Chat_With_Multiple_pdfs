from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import clean_text


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            extracted_text = page.extract_text()  

            if extracted_text:
                text += clean_text(extracted_text)

    return text

def get_text_chunk(text):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_spliter.split_text(text)

    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    cleaned_chunks = [clean_text(chunk) for chunk in chunks]
    
    vector_store = FAISS.from_texts(cleaned_chunks, embedding=embeddings)
    
    vector_store.save_local("faiss_index")
