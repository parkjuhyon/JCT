import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import cohere

# --- Cohere API 키 설정 ---
os.environ["COHERE_API_KEY"] = "r1Fl17yD8nqp8yoYtnpiGKZXPMadYECdMJHZ1hCo"
cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

# --- Cohere 임베딩 클래스 정의 ---
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

# --- 여러 PDF 파일 로드 ---
pdf_files = ['H1J.pdf', 'H3J.pdf']  # 여기에 PDF 파일명 추가

documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# --- 텍스트 분할 ---
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# --- 벡터 저장소 생성 ---
embeddings = CohereEmbeddings(client=cohere_client)
vector_store = Chroma.from_documents(texts, embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# --- Cohere Chat API 호출 함수 ---
def cohere_chat_generate(prompt: str) -> str:
    response = cohere_client.chat(message=prompt)
    return response.text

# --- Streamlit UI 구성 ---
st.set_page_config(page_title="PDF 질문 답변 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 학교 전용 챗봇 (전공심화탐구)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요! 학교에 대해 궁금한 점을 물어보세요. (학사일정)"
    }]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 답변 생성 함수 ---
def generate_response(user_question: str) -> str:
    docs = retriever.get_relevant_documents(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Use the following pieces of context to answer the users question shortly.
Given the following summaries of a long document and a question, create a final answer with references ("SOURCES"), use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
If you need it, look it up on the internet.
You MUST answer in Korean and in Markdown format

----------------
{context}

질문: {user_question}
답변:"""

    answer = cohere_chat_generate(prompt)
    return answer

# --- 사용자 입력 처리 ---
if user_input := st.chat_input("질문을 입력하세요."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    answer = generate_response(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
