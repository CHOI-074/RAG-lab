# 금융 문서 RAG 챗봇 (Financial Document RAG Chatbot)

> PDF로 된 금융 문서를 업로드하면, 문서 내용을 근거로 질문에 답해주는 RAG 기반 챗봇입니다.
> 사업보고서·약관·리포트처럼 길고 복잡한 금융 문서에서 원하는 정보를 빠르게 찾는 것을 목표로 합니다.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-009688)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-4285F4?logo=google&logoColor=white)

<br>

## 데모

> 실행 화면을 캡처해서 아래에 넣어주세요. (예: `assets/demo.png` 또는 GIF)

```
여기에 데모 스크린샷 / GIF 삽입
```

<br>

## 주요 기능

- **PDF 업로드 & 파싱** — 금융 문서 PDF를 업로드하면 텍스트를 추출해 검색 가능한 형태로 변환합니다.
- **의미 기반 검색 (Semantic Search)** — 단순 키워드가 아니라 문장의 의미로 관련 내용을 찾아옵니다.
- **근거 기반 답변 생성** — 검색된 문서 청크를 컨텍스트로 넣어, 환각(hallucination)을 줄인 답변을 생성합니다.
- **청크 사이즈 실험** — 청크 크기(300 / 500 / 800)에 따른 검색 품질 차이를 비교·분석합니다.

<br>

## 기술 스택

|구분          |사용 기술                                                     |
|------------|----------------------------------------------------------|
|**Language**|Python 3.10+                                              |
|**문서 처리**   |pypdf                                                     |
|**임베딩**     |sentence-transformers (`all-MiniLM-L6-v2`)                |
|**벡터 검색**   |FAISS                                                     |
|**LLM**     |Google Gemini API (`gemini-2.5-flash`, `google-genai` SDK)|
|**UI**      |Streamlit                                                 |

<br>

## 프로젝트 구조

```
financial-rag-chatbot/
├── app.py                # Streamlit 챗봇 UI (메인 실행 파일)
├── document_loader.py    # PDF 로딩 및 텍스트 추출 (pypdf)
├── embedder.py           # 문서 청크 임베딩 + FAISS 인덱싱
├── generator.py          # 검색 컨텍스트 기반 답변 생성 (Gemini)
├── chunk_experiment.py   # 청크 사이즈(300/500/800) 비교 실험
├── requirements.txt      # 의존성 목록
└── sample.pdf            # 데모용 샘플 PDF
```

> 파일별 역할이 위 설명과 다르면 본인 코드에 맞게 한 줄씩 수정해주세요.

<br>

## 동작 흐름 (Pipeline)

```
PDF 업로드
   │
   ▼
[document_loader] 텍스트 추출 → 청크 분할
   │
   ▼
[embedder] 청크 임베딩 → FAISS 인덱스 저장
   │
   ▼
사용자 질문 입력
   │
   ▼
질문 임베딩 → FAISS 유사도 검색 (Top-k 청크)
   │
   ▼
[generator] 검색된 청크 + 질문 → Gemini → 답변 생성
```

<br>

## 실행 방법

### 1. 레포 클론

```bash
git clone https://github.com/CHOI-074/RAG-lab.git
cd RAG-lab
```

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 활성화 (Windows / PowerShell)
venv\Scripts\Activate.ps1

# 활성화 (Mac / Linux)
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정

프로젝트 루트에 `.env` 파일을 만들고 본인의 Gemini API 키를 입력하세요.

```env
GEMINI_API_KEY=your_api_key_here
```

> Gemini API 키는 [Google AI Studio](https://aistudio.google.com/apikey)에서 발급받을 수 있습니다.

### 5. 실행

```bash
streamlit run app.py
```

<br>

## 청크 사이즈 실험

RAG 성능에 큰 영향을 주는 변수인 **청크 사이즈**를 직접 비교 실험했습니다.
동일한 PDF를 `chunk_size`만 바꾸고(overlap=50 고정), **같은 질문**으로 상위 3개 청크를 검색해 비교했습니다.

|chunk_size|청크 개수 |평균 길이  |검색 품질                        |
|:--------:|:----:|:-----:|:----------------------------|
|300       |34    |294    |잘게 쪼개져 상위에 서로 다른 주제가 섞임 · 단편적|
|**500**   |**19**|**487**|**1순위에 핵심 본문이 잡힘 · 균형 최적**   |
|800       |12    |742    |상위가 목차에 치우치고 노이즈 증가          |

같은 질문·같은 문서인데 300과 800은 1순위로 **목차 헤더**가 잡힌 반면, 500만 유일하게 1순위에 **실제 핵심 본문**이 검색됐습니다.

> **결론 — 이 문서에서는 `chunk_size = 500`이 최적.**
> 세 조건 중 유일하게 가장 관련 높은 본문을 검색 1순위로 잡았습니다.
> (단, 문서 밀도·구조에 따라 최적값은 달라질 수 있습니다.)

### 예상 못 한 인사이트: 청크보다 PDF 전처리

검색 결과에 목차 번호·전화번호·아이콘 텍스트 같은 **레이아웃 노이즈**가 계속 섞여 들어왔습니다.
디자인된 발간물 PDF라 본문 외 텍스트가 그대로 추출된 것인데, 청크 사이즈를 아무리 튜닝해도 추출 텍스트 자체에 노이즈가 많으면 검색 품질에 한계가 있었습니다.

> **청크 사이즈만큼이나 PDF 전처리(노이즈 제거)도 중요하다** — 이번 실험의 가장 큰 배움이었습니다.

> 자세한 분석 과정은 [블로그 글](#)에 정리했습니다. *(Velog 링크 추가)*

<br>

## 향후 개선 방향

- [ ] 검색 방식 개선 (Hybrid Search: 키워드 + 의미 검색 결합)
- [ ] Re-ranking 단계 추가로 검색 정확도 향상
- [ ] 멀티 문서 동시 검색 지원
- [ ] 답변에 출처(페이지·청크) 표시 기능

<br>

## 작성자

**최창연 (CHOI-074)**

- GitHub: [@CHOI-074](https://github.com/CHOI-074)
- 도메인 관심사: 금융 · 데이터 분석 · AI
