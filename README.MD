# RAG 실무 시스템 아키텍처 (Enterprise Grade)

본 아키텍처는 rag_evaluation_guide.md에서 다룬 지연 시간(Latency), 보안(Security), 평가(Evaluation)를 모두 고려한 설계입니다.

## 1. 전체 시스템 흐름도 (Workflow)

사용자의 요청부터 답변 생성, 그리고 사후 평가까지의 전체 흐름입니다.

User Request: React 프론트엔드에서 질문 전달.

API Gateway (FastAPI): 요청 수신 및 사용자 인증(Auth) 확인.

Hybrid Retrieval Layer:

Vector DB (Chroma/Pinecone): 시맨틱 검색 수행.

Elasticsearch/BM25: 키워드 기반 검색 동시 수행.

RRF(Reciprocal Rank Fusion): 두 검색 결과를 최적으로 병합.

Re-ranking Stage: 상위 후보군을 Cross-Encoder 모델로 재정렬 (정확도 극대화).

LLM Generation: 최적화된 Context를 포함한 프롬프트로 답변 생성 (Streaming 적용).

Observability & Eval: 모든 로그를 DB에 저장하고, 주기적으로 RAGAS 평가 수행.

## 2. 상세 컴포넌트 설계

1. 데이터 인베스티션 파이프라인 (Data Ingestion - Async)

Trigger: 관리자가 새 매뉴얼(PDF) 업로드.

Worker (Celery/Redis): 업로드된 파일을 백그라운드에서 처리.

Unstructured 라이브러리로 텍스트/표 추출.

RecursiveCharacterTextSplitter로 전략적 청킹 (Overlap 15% 적용).

벡터 DB 증분 업데이트 (Incremental Update).

2. 서빙 레이어 (Serving Layer - Backend)

Streaming API: LangChain의 astream 기능을 사용하여 첫 토큰 지연 시간(TTFT) 최소화.

Prompt Management: 프롬프트 버전을 관리하여 특정 버전에서 RAGAS 점수가 어떻게 변하는지 추적.

Cache (Redis): 동일하거나 유사한 질문에 대해 모델 호출 없이 즉시 답변 (비용 절감).

3. 평가 및 모니터링 레이어 (MLOps)

Evaluation Loop: 릴리즈 전 혹은 주기적으로 RAGAS를 가동하여 성능 지표 리포트 생성.

Feedback Loop: 사용자가 누른 '좋아요/싫어요' 데이터를 Ground Truth 데이터셋 구축에 활용.

Guardrails: Llama Guard 등을 배치하여 부적절한 입력/출력 차단 (보안).

## 3. 엔지니어링 의사결정 (Trade-offs)

Latency vs Accuracy: 리랭커(Re-ranker)를 사용하면 정확도는 올라가지만 응답 속도가 0.5~1초 늦어집니다. 본 아키텍처는 정확도가 중요한 '기술 문서 상담'에 최적화하여 리랭커를 포함합니다.

Cost Efficiency: 모든 질문에 GPT-4를 쓰지 않고, 난이도가 낮은 질문은 GPT-4o-mini나 로컬 모델로 라우팅하여 비용을 최적화합니다.

## 4. 기술 스택 (Proposed Stack)

Frontend: React, Tailwind CSS (출처 하이라이팅 및 스트리밍 UI)

Backend: FastAPI (비동기 처리 최적화)

AI Framework: LangChain, RAGAS

Vector DB: ChromaDB (로컬 개발) -> Pinecone (운영 배포)

Model: OpenAI gpt-4o, BGE-Reranker (리랭킹용 오픈소스 모델)