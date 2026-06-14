import streamlit as st
st.title("금융 문서 RAG 챗봇")          # 제목을 맨 먼저 그림
st.caption("앱 불러오는 중...")

from document_loader import load_pdf, chunk_text
from embedder import get_model, build_index, search
from generator import answer

@st.cache_resource
def load_model():
    with st.spinner("임베딩 모델 준비 중... (처음 한 번만 시간이 걸려요)"):
        return get_model()

model = load_model()
st.caption("준비 완료 아래에서 PDF를 올리세요")

uploaded = st.file_uploader("PDF 업로드", type="pdf")
if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())
    text = load_pdf("temp.pdf")
    chunks = chunk_text(text)
    index = build_index(chunks, model)
    st.success(f"문서 준비 완료 (청크 {len(chunks)}개)")

    query = st.text_input("질문을 입력하세요")
    if query:
        found = search(query, index, chunks, model)
        result = answer(query, found)
        st.write("### 답변")
        st.write(result)
        with st.expander("참고한 문서 부분"):
            for c in found:
                st.write(c[:300] + "...")
