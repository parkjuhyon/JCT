import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import cohere

# --- 1. API 키 및 기본 설정 ---
# ⚠️ 주의: 이 코드를 외부에 공유할 경우 API 키가 노출될 수 있습니다.
cohere_client = cohere.Client(api_key="r1Fl17yD8nqp8yoYtnpiGKZXPMadYECdMJHZ1hCo")

# --- 2. Cohere Embeddings 클래스 정의 ---
class CohereEmbeddings(Embeddings):
    def __init__(self, client, model="embed-multilingual-v3.0"):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        input_texts = [t if isinstance(t, str) else t.page_content for t in texts]
        response = self.client.embed(
            texts=input_texts, model=self.model, input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text):
        response = self.client.embed(
            texts=[text], model=self.model, input_type="search_query"
        )
        return response.embeddings[0]

# --- 3. 데이터 로딩 및 Retriever 준비 (메시지 없이 조용히 실행) ---
@st.cache_resource
def setup_retriever():
    # PDF 파일 로드
    pdf_files = ['H1J.pdf', 'H2J.pdf', 'H3J.pdf']
    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())

    # 텍스트 분할
    text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 임베딩 및 벡터 스토어 생성 (메모리에서)
    embeddings = CohereEmbeddings(client=cohere_client)
    vector_store = Chroma.from_documents(texts, embedding=embeddings) 
    
    # Retriever 반환
    return vector_store.as_retriever(search_kwargs={"k": 5})

def cohere_chat_generate(prompt: str) -> str:
    response = cohere_client.chat(message=prompt)
    return response.text

# --- 4. Streamlit UI 구성 ---
st.set_page_config(page_title="PDF 기반 학교 전용 챗봇", page_icon="🤖", layout="wide")
st.title("학교 전용 챗봇 ( 전공심화탐구 )")

# 앱 실행 시 백그라운드에서 조용히 retriever를 준비합니다.
retriever = setup_retriever()

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요! 학교에 대해 궁금한 점을 물어보세요. (학사일정/교육과정 편성표 학습 완료, 이에 관해 질문 가능)\n예시질문 : 1학기 2차고사 시험기간 알려줘 / 2학기 3학년 시험기간 알려줘 / 인공지능 기초 과목의 성적 처리 유형은 뭐야? / 진로과목 알려줘 / 공통과목의 기본학점, 운영학점 알려줘"
    }]

# 채팅 기록 표시
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 답변 생성 함수
def generate_response(user_question: str) -> str:
    docs = retriever.get_relevant_documents(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Use the following pieces of context to answer the user's question shortly.
Given the following summaries of a long document and a question, create a final answer with references ("SOURCES"), use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say "I don't know", don't try to make up an answer.
You MUST answer in Korean and in Markdown format.

----------------
{context}

질문: {user_question}
답변:"""
    return cohere_chat_generate(prompt)

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    answer = generate_response(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
