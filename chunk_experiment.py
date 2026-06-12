from document_loader import load_pdf, chunk_text
from embedder import get_model, build_index, search

text = load_pdf("sample.pdf")
model = get_model()

query = "이 문서의 주요 내용은?"   # ← 실제로 궁금한 질문으로 바꾸세요

for size in [300, 500, 800]:
    chunks = chunk_text(text, chunk_size=size, overlap=50)
    avg = sum(len(c) for c in chunks) // len(chunks)
    index = build_index(chunks, model)
    found = search(query, index, chunks, model, k=3)

    print(f"\n===== chunk_size = {size} =====")
    print(f"청크 개수: {len(chunks)} / 평균 길이: {avg}")
    print("검색된 상위 청크:")
    for i, c in enumerate(found, 1):
        print(f"  [{i}] {c[:120]}...")