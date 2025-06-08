import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import cohere

# 환경 변수 설정
os.environ["COHERE_API_KEY"] = "your_api_key_here"
os.environ["CHROMA_API_IMPL"] = "chromadb"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

class CohereEmbeddings(Embeddings):
    def __init__(self, client, model="embed-multilingual-v3.0"):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        input_texts = [t if isinstance(t, str) else t.page_content for t in texts]
        response = self.client.embed(
            texts=input_texts,
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text):
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query"
        )
        return response.embeddings[0]

# PDF 파일 로드 및 문서 분할
pdf_files = ['H1J.pdf', 'H3J.pdf']
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = CohereEmbeddings(client=cohere_client)

persist_dir = "./chroma_db"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# Chroma 벡터스토어 생성 - embedding 단수형 주의
vector_store = Chroma.from_documents(
    texts,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name="school_docs"
)
