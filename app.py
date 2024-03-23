import streamlit as st
import io
import fitz
import requests
from streamlit_chat import message
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from huggingface_hub import hf_hub_download

import pandas as pd


# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_data
def load_reviews(url):
    #url = "https://raw.githubusercontent.com/grantjw/aesop-review/main/output_transcripts.csv"
    df = pd.read_csv(url)
    # remove non-scraped transcript
    df = df[(df['Transcript'] != ' ') & (df['Transcript'] != '')]
    # Assuming df DataFrame containing 'Transcript' and 'Video URL' columns
    review = df['Transcript'].str.cat(sep='\n')
    return review

@st.cache_resource
def get_retriever(url):
    reviews = load_reviews(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40,separators=['\n',"',",' ', ''])
    chunk_list = []
    chunks = text_splitter.split_text(reviews)
    chunk_list.extend(chunks)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunk_list, embeddings)
    #db.similarity_search("customer service",k=5)
    retriever = db.as_retriever()
    return retriever


@st.cache_resource
def create_chain(_retriever):
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    # stream_handler = StreamHandler(st.empty())

    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear as if the LLM is typing the
    # responses in real time.
    # callback_manager = CallbackManager([stream_handler])
    (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                                  "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

    model_path = hf_hub_download(repo_id=repo_id,
                                 filename=model_file_name,
                                 repo_type="model")
    
    n_gpu_layers = 1  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 1024 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    llm = LlamaCpp(
            model_path=model_path,
            n_batch=n_batch,
            n_ctx=4096,
            max_tokens=2048,
            temperature=.33,
            # callback_manager=callback_manager,
            top_p=1, 
            verbose=True,
            streaming=True,
            )

    # Template for the prompt.
    # template = "{question}"

    # We create a prompt from the template so we can use it with langchain
    # prompt = PromptTemplate(template=template, input_variables=["question"])

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # We create a qa chain with our llm, retriever, and memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=_retriever, memory=memory, verbose=False
    )

    return qa_chain



def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi, I know what Youtubers said about Aesop's products. Ask me!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="      ", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

base_url = "https://raw.githubusercontent.com/grantjw/product_chatbot_rag/main/data/output_transcripts.csv"
retriever = get_retriever(base_url)
llm_chain = create_chain(retriever)
initialize_session_state()
st.title("Aesop Product Reviewer from YouTube Reviews")
st.image("aesop.png", width=550)
st.markdown("""
    This app provides insights into Aesop products based on YouTube reviews.
    
    [![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=github)](https://github.com/grantjw/product_chatbot_rag)
    """, unsafe_allow_html=True)
display_chat_history(llm_chain)
