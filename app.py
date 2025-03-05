import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

import re
import unicodedata

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

########## clean Text ##############
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)  # Normalize Unicode
    text = text.encode("utf-8", "ignore").decode("utf-8")  # Remove unsupported chars
    text = re.sub(r'[\ud800-\udfff]', '', text)  # Remove surrogate pairs
    text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text
####################################

########## convert Pdf to Vectors #############
def get_pdf_text(pdf_docs): # read pdf files and takes all text in one string
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

def get_text_chunk(text): # full sentences to convert chunks contain & improving retrieval accuracy.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", ".", "?", "!"])  # Split at sentence boundaries
    
    return text_splitter.split_text(text) 

def get_vector_store(chunks): # Use Hybrid Search (Embeddings + BM25) for better retrieval.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    cleaned_chunks = [clean_text(chunk) for chunk in chunks]
    vector_store = FAISS.from_texts(cleaned_chunks, embedding=embeddings) # FAISS use similarity search 
    bm25_retriever = BM25Retriever.from_texts(cleaned_chunks)     # Use BM25 for keyword-based retrieval
    vector_store.save_local("faiss_index")

    return vector_store, bm25_retriever
###############################################

########## Take User input - LangChain ##############
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, 'answer is not available in the context', don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# chatbot forgets previous messages so Store chat history in st.session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True)
    
    st.write("### 🤖 Reply:")
    st.write(response["output_text"])
    response_text = response["output_text"] # bot remembers past questions!
    st.session_state.chat_history.insert(0, {"question": user_question, "answer": response_text}) # Store conversation history

    st.success("### Chat History")    # Display full chat history
    for chat in st.session_state.chat_history:
        st.write(f"👤 **You:** {chat['question']}")
        st.write(f"🤖 **Bot:** {chat['answer']}")
#########################################


########## Streamlit UI ##############
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("📄 Chat with Multiple PDFs using Gemini 💁🏻‍♂️")

    col1, col2 = st.columns([3, 2])

    with col1: # chat page
        user_question = st.text_input("💬 Ask a question:")
        if user_question:
            user_input(user_question)

    with col2: # Sidebar
        with st.sidebar:
            st.title("📂 Upload PDFs")
            pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunk(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done!")
#########################################
#########################################
if __name__ == "__main__":
    main()
