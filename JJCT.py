import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import cohere

# 1. 환경 변수 설정 (본인 키로 바꾸기)
os.environ["COHERE_API_KEY"] = "r1Fl17yD8nqp8yoYtnpiGKZXPMadYECdMJHZ1hCo"
os.environ["CHROMA_API_IMPL"] = "chromadb"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. 코히어 클라이언트 초기화
cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

# 3. Cohere 임베딩 클래스 정의
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

# 4. PDF 파일 로딩 및 문서 분할
pdf_files = ['H1J.pdf', 'H3J.pdf']  # 파일명 변경 가능
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 5. 임베딩 객체 생성
embeddings = CohereEmbeddings(client=cohere_client)

# 6. 벡터스토어 디렉토리 준비
persist_dir = "./chroma_db"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# 7. Chroma 벡터스토어 생성 (embedding 단수형)
vector_store = Chroma.from_documents(
    texts,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name="school_docs"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 8. Cohere 챗봇 생성 함수
def cohere_chat_generate(prompt: str) -> str:
    response = cohere_client.chat(message=prompt)
    return response.text

# 9. Streamlit 앱 UI 구성
st.set_page_config(page_title="PDF 질문 답변 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 학교 전용 챗봇 (전공심화탐구)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 학교에 대해 궁금한 점을 물어보세요. (학사일정)"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 10. 질문에 대한 답변 생성 함수
def generate_response(user_question: str) -> str:
    docs = retriever.get_relevant_documents(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""아래 문서 내용을 참고하여 사용자의 질문에 짧고 명확하게 답변해주세요.
다음 문서 요약과 질문이 주어지면, 반드시 'SOURCES'라는 단어와 함께 출처를 명시하며 답변을 작성하세요.
모르면 "모르겠습니다"라고 답변하고, 추측하지 마세요.
답변은 한국어 마크다운 형식으로 작성하세요.

----------------
{context}

질문: {user_question}
답변:"""

    answer = cohere_chat_generate(prompt)
    return answer

# 11. 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    answer = generate_response(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
