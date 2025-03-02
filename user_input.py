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

    # bot remembers past questions!
    response_text = response["output_text"]

    # Store conversation history
    st.session_state.chat_history.insert(0, {"question": user_question, "answer": response_text})

    # Display full chat history
    st.success("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"👤 **You:** {chat['question']}")
        st.write(f"🤖 **Bot:** {chat['answer']}")