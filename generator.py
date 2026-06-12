import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""아래 [문서] 내용만 근거로 [질문]에 답하세요.
문서에 근거가 없으면 "문서에서 찾을 수 없습니다"라고 답하세요.

[문서]
{context}

[질문]
{query}
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return resp.text