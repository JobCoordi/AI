from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import TranslationModel
from last import FinalRecommendation
import os
from dotenv import load_dotenv
import logging

# 기본 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 환경변수 로드
load_dotenv()
db_conn = os.getenv("DB_CONNECTION_STRING")

# 모델 초기화 예외 처리
try:
    translator = TranslationModel(model_name="gpt-4o")
except Exception as e:
    logger.error(f"TranslationModel 로드 실패: {e}")
    translator = None

try:
    last = FinalRecommendation(model_name="gpt-4o", db_connection=db_conn)
except Exception as e:
    logger.error(f"FinalRecommendation 로드 실패: {e}")
    last = None

# 요청 모델
class MessageRequest(BaseModel):
    user_id: str
    content: str
    is_last: bool

# 카테고리 정의
category = [
    "사업관리", "경영·회계·사무", "금융·보험", "교육·자연·사회과학", "법률·경찰·소방·교도·국방",
    "보건·의료", "사회복지·종교", "문화·예술·디자인·방송", "운전·운송", "영업판매",
    "경비·청소", "이용·숙박·여행·오락·스포츠", "음식서비스", "건설", "기계",
    "재료", "화학·바이오(구.화학)", "섬유·의복", "전기·전자", "정보통신",
    "식품가공", "인쇄·목재·가구·공예", "환경·에너지·안전", "농림어업"
]

@app.post("/chat")
def chat(req: MessageRequest):
    if not translator or not last:
        raise HTTPException(status_code=500, detail="모델 초기화 실패")

    try:
        if not req.is_last:
            response = translator.translate(user_id=req.user_id, content=req.content)
            return {"user_id": req.user_id, "response": response}
        else:
            result = last.get_recommendation(user_id=req.user_id)
            logger.info(f"추천 결과: {result}")
            parsed = result.strip("[]").split(":")
            if len(parsed) != 3:
                raise ValueError("추천 결과 형식 오류")
            
            job = parsed[0].strip()
            reason = parsed[1].strip()
            cat = parsed[2].strip()

            if cat not in category:
                raise ValueError(f"존재하지 않는 카테고리: {cat}")

            category_index = category.index(cat) + 1

            return {
                "job": job,
                "reason": reason,
                "category": category_index
            }
    except Exception as e:
        logger.error(f"/chat 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

@app.get("/")
def root():
    return {"message": "Chat API is running"}
