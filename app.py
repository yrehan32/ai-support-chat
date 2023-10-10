import os
import streamlit as st
from dotenv import load_dotenv
from torch import cuda
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Load the .env file
load_dotenv()

# Define the path for generated embeddings
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Device
device = 'cuda' if cuda.is_available() else 'cpu'

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':device})

# Vectorstore
vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings)

# Create llm
llm = CTransformers(model=os.getenv('MODEL_FILE'), model_type="llama",
                    config={'max_new_tokens':256,'temperature':0.5})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)

st.title(os.getenv('SUPPORT_CONTEXT') + " Support")
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + os.getenv('SUPPORT_CONTEXT')]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your " + os.getenv('SUPPORT_CONTEXT'), key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()