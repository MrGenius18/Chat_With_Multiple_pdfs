from pdf_to_vector import *
from user_input import *

import os
from dotenv import load_dotenv
import streamlit as st

import google.generativeai as genai


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat With Multiple PDF using Gemini 💁🏻‍♀️")

    user_question = st.text_input("Ask a Question from the PDF Files...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type="pdf", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunk = get_text_chunk(raw_text)
                get_vector_store(text_chunk)
                st.success("Done")

if __name__ == "__main__":
    main()
