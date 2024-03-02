# product_chatbot_rag
# Streamlit + Langchain + LLama.cpp w/ Mistral: Retrieval Augmented Generation

This chatbot has conversational memory and can hold follow up conversations within the same session.

It runs on Mac M2 pro. 

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
