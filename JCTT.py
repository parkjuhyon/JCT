# main.py
import os
import streamlit as st
import cohere
# from dotenv import load_dotenv # 이 줄을 삭제하거나 주석 처리합니다.
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# --- 1. 초기 설정 및 상수 정의 ---
# API 키를 코드에 직접 입력합니다.
# "여기에_실제_Cohere_API_키를_입력하세요" 부분을 실제 키로 교체해주세요.
COHERE_API_KEY = "여기에_실제_Cohere_API_키를_입력하세요" 

PDF_FILES = ['H1J.pdf', 'H2J.pdf', 'H3J.pdf']
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "school_bot"

# --- 2. Cohere Embeddings 클래스 정의 ---
class CohereEmbeddings(Embeddings):
    """Cohere API를 사용하기 위한 LangChain Embedding 래퍼 클래스"""
    def __init__(self, client, model="embed-multilingual-v3.0"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            texts=texts, model=self.model, input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed(
            texts=[text], model=self.model, input_type="search_query"
        )
        return response.embeddings[0]

# --- 3. 핵심 로직 함수화 ---
@st.cache_resource
def load_vector_store():
    """PDF를 로드하고 Vector Store를 생성 또는 로드하는 함수 (Streamlit 캐싱 활용)"""
    if not COHERE_API_KEY or COHERE_API_KEY == "여기에_실제_Cohere_API_키를_입력하세요":
        st.error("Cohere API 키가 코드에 설정되지 않았습니다. COHERE_API_KEY 변수를 확인하세요.")
        st.stop()
    
    cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    embeddings = CohereEmbeddings(client=cohere_client)

    # Vector DB가 이미 존재하는지 확인
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        st.info("기존 Vector DB를 로드합니다.")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

    # Vector DB가 없으면 새로 생성
    st.info("새로운 Vector DB를 생성합니다. 잠시만 기다려주세요...")
    documents = []
    for file in PDF_FILES:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
        else:
            st.warning(f"PDF 파일을 찾을 수 없습니다: {file}")

    if not documents:
        st.error("처리할 문서가 없습니다. PDF 파일 경로를 확인하세요.")
        st.stop()

    text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(
        texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    st.success("Vector DB 생성이 완료되었습니다!")
    return vector_store

def generate_response(retriever, user_question: str) -> str:
    """RAG 파이프라인을 통해 답변을 생성하는 함수"""
    docs = retriever.get_relevant_documents(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""다음 컨텍스트를 사용하여 사용자의 질문에 간결하게 답변하세요.
참고 자료가 있을 경우, 대문자로 "SOURCES"를 명시하고 출처를 밝히세요.
답을 모를 경우, 꾸며내지 말고 "잘 모르겠습니다"라고만 답변하세요.
반드시 한국어와 마크다운 형식으로 답변해야 합니다.

----------------
{context}

질문: {user_question}
답변:"""
    
    cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    response = cohere_client.chat(message=prompt)
    return response.text

# --- 4. Streamlit UI 구성 ---
st.set_page_config(page_title="PDF 기반 학교 전용 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 학교 전용 챗봇 (전공심화탐구)")

try:
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "안녕하세요! 학교에 대해 궁금한 점을 물어보세요. (학사일정 등)"
        }]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("질문을 입력하세요."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        answer = generate_response(retriever, user_input)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
