from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from model import TranslationModel
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
translator = TranslationModel(model_name="gpt-4o")

class ProfileRequest(BaseModel):
    user_id: str
    birth_year: Optional[str] = None
    gender: Optional[str] = None
    education_level: Optional[str] = None
    major: Optional[str] = None
    career: Optional[str] = None
    interests: Optional[str] = None
    certifications: Optional[str] = None
    preferred_work: Optional[str] = None
    self_description: Optional[str] = None
    profiles: Optional[str] = None
    session_id: Optional[str] = "default_thread"
    language: Optional[str] = "korean"

def convert_to_prompt(req: ProfileRequest) -> str:
    if req.profiles:
        return req.profiles
    return (
        f"출생년도:{req.birth_year or '모름'}, 성별:{req.gender or '모름'}, "
        f"최종학력:{req.education_level or '모름'}, 전공:{req.major or '모름'}, "
        f"경력:{req.career or '모름'}, 관심 분야:{req.interests or '모름'}, "
        f"자격증:{req.certifications or '모름'}, 희망 근무형태:{req.preferred_work or '모름'}, "
        f"자기소개:{req.self_description or '모름'}"
    )

@app.post("/recommend")
def recommend_job(req: ProfileRequest):
    user_input = convert_to_prompt(req)
    response = translator.translate(
        text=user_input,
        session_id=req.session_id,
        language=req.language
    )
    return {"user_id": req.user_id, "response": response}

# 헬스체크
@app.get("/")
def root():
    return {"message": "Translation API is running"}
