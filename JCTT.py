import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, CohereRerank # CohereRerank 추가
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever # Reranker 사용을 위해 추가
import cohere

# --- 1. API 키 직접 설정 ---
# ⚠️ 주의: 이 코드를 외부에 공유할 경우 API 키가 노출될 수 있습니다.
COHERE_API_KEY = "r1Fl17yD8nqp8yoYtnpiGKZXPMadYECdMJHZ1hCo" 
cohere_client = cohere.Client(api_key=COHERE_API_KEY)


# --- 2. 데이터 로딩 및 Retriever 준비 (메시지 없이 조용히 실행) ---
@st.cache_resource
def setup_retriever():
    # PDF 파일 로드 및 메타데이터(소스) 추가
    pdf_files = ['H1J.pdf', 'H2J.pdf', 'H3J.pdf']
    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        # load_and_split을 사용하고, 각 문서에 파일명을 메타데이터로 추가
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata['source'] = file 
        documents.extend(loaded_docs)

    # 텍스트 분할 (Chunk Size 조정)
    # 조금 더 작은 단위로 나누어 Reranker가 세밀하게 평가하도록 함
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 임베딩 모델 정의
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)
    
    # Chroma 벡터 스토어 생성
    vector_store = Chroma.from_documents(texts, embedding=embeddings) 
    
    # 기본 Retriever 설정 (더 많은 후보군 확보)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # ✨ [개선] Cohere Rerank를 사용한 압축 Retriever 설정
    compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    return compression_retriever

# --- 3. Cohere Chat API 호출 함수 ---
# ✨ [개선] Preamble을 사용하여 모델의 역할과 답변 규칙을 명확히 정의
def cohere_chat_generate(user_question: str, documents: list) -> str:
    
    # 검색된 문서 내용을 컨텍스트로 변환
    context = "\n\n".join([f"출처: {doc.metadata.get('source', '알 수 없음')}, 페이지: {doc.metadata.get('page', '알 수 없음')}\n내용: {doc.page_content}" for doc in documents])

    # Cohere의 Preamble(사전 지시) 기능을 활용하여 역할 부여
    preamble = """
    당신은 대한민국 고등학교의 학사 정보 안내를 도와주는 AI 챗봇 '써바!'입니다.
    당신의 임무는 주어진 '문서' 내용을 기반으로 사용자의 '질문'에 대해 명확하고 친절하게 한국어로 답변하는 것입니다.
    
    ## 답변 규칙
    1. 반드시 '문서'에 명시된 내용만을 근거로 답변해야 합니다. 문서에 없는 내용은 "문서에서 관련 정보를 찾을 수 없습니다."라고 솔직하게 답변하세요. 절대로 정보를 지어내지 마세요.
    2. 답변은 항상 완전한 문장 형태의 서술형으로 작성하고, 핵심 내용을 먼저 말해주세요.
    3. 답변 마지막에는 **[근거 자료]** 항목을 만들고, 답변의 근거가 된 문서의 '출처'와 '페이지'를 명확히 밝혀주세요.
    """
    
    # documents를 활용하여 grounding 강화 (Cohere API의 네이티브 기능)
    # 이 방식은 모델이 제공된 문서를 더 잘 참조하게 함
    response = cohere_client.chat(
        message=user_question,
        preamble=preamble,
        documents=[{"title": doc.metadata.get('source', ''), "snippet": doc.page_content} for doc in documents]
    )
    
    # ✨ [개선] 답변과 함께 인용(근거) 정보도 함께 반환
    answer_text = response.text
    citations = response.citations
    
    if citations:
        cited_sources = set()
        for citation in citations:
            for doc_name in citation.document_ids:
                # 'doc_X' 형식의 ID에서 문서 제목을 찾기 위함
                try:
                    # Cohere API는 documents 리스트의 인덱스를 기반으로 ID를 생성합니다.
                    doc_index = int(doc_name.split('_')[-1])
                    source_doc = documents[doc_index]
                    source_name = source_doc.metadata.get('source', '알 수 없음')
                    page_num = source_doc.metadata.get('page', 'N/A')
                    cited_sources.add(f"{source_name} (p. {page_num})")
                except (ValueError, IndexError):
                    continue

        if cited_sources:
            answer_text += "\n\n**[근거 자료]**\n- " + "\n- ".join(sorted(list(cited_sources)))
            
    return answer_text


# --- 4. Streamlit UI 구성 ---
st.set_page_config(page_title="PDF 기반 학교 전용 챗봇", page_icon="🤖", layout="wide")
st.title("학교 전용 챗봇 (전공심화탐구) 🚀")

# 앱 실행 시 백그라운드에서 retriever 준비
retriever = setup_retriever()

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요! 학교 생활에 대해 궁금한 점을 물어보세요. (학사일정/교육과정 편성표 학습 완료)\n\n**예시 질문:**\n- 1학기 2차고사 시험기간 알려줘\n- 3학년 진로과목에는 뭐가 있어?\n- 인공지능 기초 과목의 성취도 분할 방식은?"
    }]

# 채팅 기록 표시
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중입니다..."):
            # ✨ [개선] Reranker를 통해 압축된 관련성 높은 문서 가져오기
            relevant_docs = retriever.get_relevant_documents(user_input)
            
            # 문서가 있는지 확인
            if not relevant_docs:
                answer = "죄송합니다, 관련 정보를 찾을 수 없습니다."
            else:
                answer = cohere_chat_generate(user_input, relevant_docs)
            
            st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
