# server.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from model import TranslationModel
import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# 모델 인스턴스 초기화
translator = TranslationModel(
    model_name="gpt-3.5-turbo",  # 또는 "gpt-4o"
    db_connection=None  # .env에서 자동으로 가져감
)

# 요청 스키마 정의
class TranslationRequest(BaseModel):
    text: str
    session_id: str = "default_thread"
    language: str = "korean"

# POST 요청 처리
@app.post("/translate")
def translate(req: TranslationRequest):
    response = translator.translate(
        text=req.text,
        session_id=req.session_id,
        language=req.language
    )
    return {"response": response}

# 헬스체크
@app.get("/")
def root():
    return {"message": "Translation API is running"}
