import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. 환경 설정 (API 키는 런타임에 제공됨)
os.environ["OPENAI_API_KEY"] = ""

# 2. 기초 RAG 파이프라인 구축 (Retriever + Generator)
# 실제 프로젝트에서는 여기서 'Hybrid Search'나 'Chunking 전략'이 가미됩니다.
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """다음 제공된 문맥(Context)만을 사용하여 질문에 답하세요:
Context: {context}
Question: {question}
Answer: """
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4o-mini")

# RAG 체인 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

# 3. 평가용 데이터셋 준비 (질문, 답변, 문맥, 정답)
# 실무에서는 실제 유저의 질문과 전문가가 작성한 정답(Ground Truth)을 모아야 합니다.
data_samples = {
    "question": ["RAG의 주요 장점은 무엇인가요?", "임베딩 모델이란 무엇인가요?"],
    "answer": [],      # 생성된 답변이 들어갈 자리
    "contexts": [],    # 검색된 문맥이 들어갈 자리
    "ground_truth": [
        "RAG는 외부 지식을 활용해 환각 현상을 줄이고 최신 정보를 제공할 수 있다는 장점이 있습니다.",
        "텍스트를 고차원 벡터로 변환하여 의미적 유사도를 계산할 수 있게 하는 모델입니다."
    ]
}

# 파이프라인 실행 및 결과 수집
for q in data_samples["question"]:
    # 답변 생성
    response = rag_chain.invoke(q)
    data_samples["answer"].append(response.content)
    
    # 검색된 문맥 수집 (평가를 위해 필수)
    docs = retriever.get_relevant_documents(q)
    data_samples["contexts"].append([doc.page_content for doc in docs])

dataset = Dataset.from_dict(data_samples)

# 4. RAGAS 평가 수행
# 여기서 'LLM-as-a-Judge' 원리에 따라 수치화된 점수가 나옵니다.
result = evaluate(
    dataset,
    metrics=[
        faithfulness,       # 충실도: 문서에 있는 내용만 말했는가?
        answer_relevancy,  # 관련성: 질문에 맞는 답인가?
        context_precision, # 검색 정밀도: 필요한 정보가 상위에 있는가?
        context_recall     # 검색 재현율: 정답에 필요한 내용이 다 있는가?
    ],
)

print("--- RAG Evaluation Results ---")
print(result)

