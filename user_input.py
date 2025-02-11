import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS


def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, 'answer is not available in the context', don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True)
    
    # print(response)
    st.write("Reply: ", response["output_text"])