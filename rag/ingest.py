import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

data_path = "../data"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="biomedical_pdf_embeddings",
    embedding_function=embedding_model,
    persist_directory=os.path.join(data_path,"chroma_db")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

for pdf_path in os.listdir(data_path):
    if pdf_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path=os.path.join(data_path, pdf_path)).lazy_load()
        splitted_doc = text_splitter.split_documents(loader)
        vector_store.add_documents(documents=splitted_doc)




