import streamlit as st
from bs4 import BeautifulSoup
import io
import fitz
import requests
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    # Assuming your DataFrame is named df
    df = df[(df['Transcript'] != ' ') & (df['Transcript'] != '')]
    # Assuming df DataFrame containing 'Transcript' and 'Video URL' columns
    all_content = [(row['Video URL'], row['Transcript']) for index, row in df.iterrows()]
    documents = [Document(page_content=doc, metadata={'url': url}) for (url, doc) in all_content]
    return documents

@st.cache_resource
def get_retriever(url):
    documents = load_reviews(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    # Split each review into chunks and create a Document object for each chunk
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    print("at least we ar ehere?")
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
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

    n_gpu_layers = 1  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 1024 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    llm = LlamaCpp(
            model_path="models/mistral-7b-instruct-v0.1.Q5_0.gguf",
            n_batch=n_batch,
            n_ctx=2048,
            max_tokens=2048,
            temperature=0,
            # callback_manager=callback_manager,
            verbose=False,
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


# Set the webpage title
st.set_page_config(
    page_title="Youtube Aesop Product Reviewer"
)

# Create a header element
st.header("Youtube Aesop Product Reviewer")

#
system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt")

if "base_url" not in st.session_state:
    st.session_state.base_url = ""

base_url = st.text_input("Enter the site url here", key="base_url")

if st.session_state.base_url != "":
    
    retriever = get_retriever(base_url)

    # We store the conversation in the session state.
    # This will be used to render the chat conversation.
    # We initialize it with the first message we want to be greeted with.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you today?"}
        ]

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    # We loop through each message in the session state and render it as
    # a chat message.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # We initialize the quantized LLM from a local path.
    # Currently most parameters are fixed but we can make them
    # configurable.
    llm_chain = create_chain(retriever)

    # We take questions/instructions from the chat input to pass to the LLM
    if user_prompt := st.chat_input("Your message here", key="user_input"):

        # Add our input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

        # Add our input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Pass our input to the llm chain and capture the final responses.
        # It is worth noting that the Stream Handler is already receiving the
        # streaming response as the llm is generating. We get our response
        # here once the llm has finished generating the complete response.
        response = llm_chain.run(user_prompt)

        # Add the response to the session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response)
