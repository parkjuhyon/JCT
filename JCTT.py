import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import cohere
#위에 말했던 라이브러리 불러오기
cohere_client = cohere.Client(api_key="r1Fl17yD8nqp8yoYtnpiGKZXPMadYECdMJHZ1hCo") #api키 불러오기

class CohereEmbeddings(Embeddings): #LangChain에서 쓸 Cohere 임베딩 클래스 정의하기
    def __init__(self, client, model="embed-multilingual-v3.0"): # 클라이언트랑 모델 이름 지정 여기선 embed-multilingual -v3.0 모델 지정
        self.client = client
        self.model = model

    def embed_documents(self, texts): #글->벡터 변환 함수
        input_texts = [t if isinstance(t, str) else t.page_content for t in texts] #문서가 객체일 경우 텍스트를 추출한다.
	#Cohere API를 써서 임베딩 실행
        response = self.client.embed(
            texts=input_texts,
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings #반환

    def embed_query(self, text):#질문->벡터로 변환 함수
	#Cohere API를 써서 임베딩 실행
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query"
        )
        return response.embeddings[0]#반환
#주요 기능
#pdf 문서 집어넣기
pdf_files = ['H1J.pdf','H2J.pdf', 'H3J.pdf'] #pdf 파일 목록을 리스트로 저장
documents = [] #저장할 전체 문서 리스트 만들기
for file in pdf_files: #위에서 정의한 pdf_files을 file에 넣으면서 반복
    loader = PyPDFLoader(file)#PyPDFLoader를 써서 파일을 읽음
    documents.extend(loader.load()) #pdf 파일을 페이지별로 읽어서 LangChain의 document 객체 형태로 저장되는 코드임

#텍스트 분할
#AI를 위한것
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)#문서를 700자 단위로 나눔, 청크에 겹치는 문자 200자 설정해서 짤리는 상황 안만들게 함
texts = text_splitter.split_documents(documents) #나눈 청크를 리스트로 저장

#벡터 저장
embeddings = CohereEmbeddings(client=cohere_client)#위에서 정의해놨던 Cohere 임베딩 만들기
vector_store = Chroma.from_documents(texts, embedding=embeddings)#문서 임베딩하고 Chroma에 저장 -> 검색 가능하게 설정함
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) #가장 연관성이 있는 문서를 찾을려고 쓰는 코드 / k값 = 유사도라고 생각(원래 문서 개수를 설정하는데 3으로 설정하니까 자꾸 문서를 못찾아서 임의로 5로 설정함

def cohere_chat_generate(prompt: str) -> str:#Cohere 함수 정의하기
    response = cohere_client.chat(message=prompt)#프롬프트(뒤에 정의됨)를 기반으로 답변 만들기
    return response.text#응답 텍스트로 반환
#streamlit 주요 UI 구성하기
st.set_page_config(page_title="PDF 기반 학교 전용 챗봇", page_icon="🤖", layout="wide") #제목,아이콘,화면 넓이 설정
st.title("🤖 학교 전용 챗봇 ( 전공심화탐구 )")#페이지에서 표시할 주 제목

if "messages" not in st.session_state:#만약에 전에 대화했던게 없으면 초기 메시지 출력하기
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요! 학교 생활에 대해 궁금한 점을 물어보세요. (학사일정/교육과정 편성표 학습 완료)\n\n**예시 질문:**\n- 1학기 2차고사 시험기간 알려줘\n- 3학년 진로과목에는 뭐가 있어?\n- 인공지능 기초 과목의 성취도 분할 방식은?"
    }]
#앞에서 만든 메시지를 화면에 출력하기 ( 한번쨰 묻기 -> 그거에 대답, 두 번째 묻기 -> 그거에 대답하는데 앞에 했던 대화 유지하고 대답하기 기능)
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
#답변 만드는 함수 정의
def generate_response(user_question: str) -> str:
    docs = retriever.get_relevant_documents(user_question) #질문 관련 문서 찾기
    context = "\n\n".join([doc.page_content for doc in docs]) #pdf문서 내용을 문자열로 만들어서 저장

    prompt = f"""You are an AI assistant for a school. Answer the user's question based ONLY on the provided context below.
Your main goal is to provide accurate information based on the documents.
If the answer is not available in the context, you MUST say "제공된 문서에서는 해당 정보를 찾을 수 없습니다."
Do not try to make up an answer. Answer in Korean and in Markdown format.
답변을 할 때 시간이 오래 걸려도 무조건 한번 더 생각해.
한번 더 생각 할 때 제대로 생각해.
개학식을 질문받으면 한번 더 생각하고 2학기 개학식은 8월 13일이야.
그리고 학사일정을 질문받았을 때 개학식도 2학기엔 8월 13일이야.

CONTEXT:
{context}

----------------

질문: {user_question}

답변:
""" # {context}와 {user_q~}는 앞에서 만든 변수를 프롬포트에 넣은것, 프롬프트 내에서 SOURCES를 사용해서 참고 문헌 표시하기, 

    return cohere_chat_generate(prompt) #보이는거랑 똑같이 Cohere에서 답변 만들기

#마지막 코드
#사용자 질문 -> 실행되는 코드
if user_input := st.chat_input("질문을 입력하세요."):
    st.session_state["messages"].append({"role": "user", "content": user_input})#입력값을 저장함
    st.chat_message("user").write(user_input)#화면에 사용자가 입력한 메시지 표시

    answer = generate_response(user_input)#pdf를 기반한 답변 만들기
    st.session_state["messages"].append({"role": "assistant", "content": answer})#Cohere가 만든 답변을 저장함
    st.chat_message("assistant").write(answer)#답변 화면에 출력하기
