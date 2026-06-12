from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# 확인용
if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    chunks = chunk_text(text)
    print(f"총 글자 수: {len(text)}")
    print(f"청크 개수: {len(chunks)}")
    print("첫 청크 미리보기:", chunks[0][:100])