from pdf_to_vector import *
from user_input import *

import os
from dotenv import load_dotenv
import streamlit as st

import google.generativeai as genai


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("📄 Chat with Multiple PDFs using Gemini 💁🏻‍♂️")

    col1, col2 = st.columns([3, 2])

    # chat page
    with col1:
        user_question = st.text_input("💬 Ask a question:")
        if user_question:
            user_input(user_question)

    # Sidebar
    with col2:
        with st.sidebar:
            st.title("📂 Upload PDFs")
            pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
            
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunk(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done!")

if __name__ == "__main__":
    main()
