# product_chatbot_rag
# LLama.cpp w/ Mistral 7b: RAG Product Reviewer
See the app live on here: [Aesop Review Chatbot](https://huggingface.co/spaces/grantjw/Aesop_Review_Chatbot)

## Retrieval Augmented Generation (RAG) and Vector Database

### Approach and Rationale
This project employs a novel approach to handling and analyzing transcribed YouTube product reviews by leveraging Retrieval Augmented Generation (RAG) with an emphasis on speed and efficiency. Instead of using traditional document embeddings, which can be computationally intensive and slow for real-time applications, the project adopts a text splitting strategy. This method significantly increases the processing speed, making the system more responsive and efficient for end-users.

### Text Splitting and Embeddings
- **Text Splitting:** The project utilizes the `RecursiveCharacterTextSplitter` from the `LangChain` library to split the transcribed text into smaller chunks. Specifically, the text is divided into chunks of 400 characters, with an overlap of 40 characters and using specific separators such as new lines and spaces. This fine-grained splitting strategy ensures that the context is preserved while making the retrieval process faster and more efficient.
- **Embeddings:** For converting text chunks into vector representations, the project uses `HuggingFaceEmbeddings` with the model `"all-MiniLM-L6-v2"`. These embeddings are known for their balance between performance and computational efficiency, making them an ideal choice for real-time applications where speed is crucial.

### Vector Database and FAISS
The vector representations of the text chunks are stored in a `FAISS` vector database. `FAISS` is an efficient library for similarity search and clustering of dense vectors, which further contributes to the speed and efficiency of the retrieval process. By using `FAISS`, the project can quickly perform similarity searches among the chunks, enabling the RAG model to retrieve relevant information in response to user queries.

### Why Not Document Embeddings?
The decision to use text splitting and `FAISS` over traditional document embeddings is driven by the need for speed. Document embeddings, while effective for capturing the semantic meaning of entire documents, can be slow to compute and cumbersome to work with in real-time applications. In contrast, splitting the text into smaller chunks and using efficient embeddings allow for faster processing and retrieval, significantly improving the responsiveness of the chatbot interface.

## Key Features
- **Efficient Data Handling:** By splitting texts instead of using traditional embeddings, the project achieves faster processing times.
- **Lightweight Model Usage:** `llama.cpp` offers an efficient way to use large language models without the need for extensive hardware resources.

-You can also compare it's performance by comparing with Llama-2-7b-chat-hf with the "LLAMA2-7B_Aesop_Reviewer.ipynb" file.
-This chatbot has conversational memory and can hold follow up conversations within the same session.
-It runs on Mac M2 pro. 

You will also need to change how you install `llama-cpp-python` package depending on your OS and whether  
you are planning on using a GPU or not.

# TL;DR instructions

1. Install llama-cpp-python
2. Install langchain
3. Install streamlit
4. Install beautifulsoup
6. Install sentence-transformers
7. Install docarray
8. Install pydantic 1.10.8
9. Download Mistral from HuggingFace from TheBloke's repo: mistral-7b-instruct-v0.1.Q5_0.gguf
10. Place model file in the `models` subfolder
11. Run streamlit

# Step by Step instructions

The setup assumes you have `python` already installed and `venv` module available.

1. Download the code or clone the repository.
2. Inside the root folder of the repository, initialize a python virtual environment:
```bash
python -m venv venv
```
3. Activate the python envitonment:
```bash
source venv/bin/activate
```
4. Install required modules (`langchain`, `llama.cpp`, and `streamlit` along with `beautifulsoup4`, `pymypdf`, `sentence-transformers`, `docarray`, and `pydantic 1.10.8`):
```bash
pip install -r requirements.txt
```
4. Create a subdirectory to place the models in:
```bash
mkdir -p models
```
6. Download the `Mistral7b` quantized model from `huggingface` from the following link:
[mistral-7b-instruct-v0.1.Q5_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_0.gguf)
7. Start `streamlit`:
```bash
streamlit run main.py
```

