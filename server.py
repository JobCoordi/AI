from fastapi import FastAPI
from pydantic import BaseModel
from model import TranslationModel
from last import FinalRecommendation
import os
from dotenv import load_dotenv
app = FastAPI()
translator = TranslationModel(model_name="gpt-4o")
load_dotenv()
db_conn = os.getenv("DB_CONNECTION_STRING")

last = FinalRecommendation(model_name="gpt-4o", db_connection=db_conn)

class MessageRequest(BaseModel):
    user_id: str
    content: str
    is_last: bool

@app.post("/chat")
def chat(req: MessageRequest):
    if not req.is_last :
        response = translator.translate(user_id=req.user_id, content=req.content)
        return {"user_id": req.user_id, "response": response}
    else :
        category=[
    "사업관리",
    "경영·회계·사무",
    "금융·보험",
    "교육·자연·사회과학",
    "법률·경찰·소방·교도·국방",
    "보건·의료",
    "사회복지·종교",
    "문화·예술·디자인·방송",
    "운전·운송",
    "영업판매",
    "경비·청소",
    "이용·숙박·여행·오락·스포츠",
    "음식서비스",
    "건설",
    "기계",
    "재료",
    "화학·바이오(구.화학)",
    "섬유·의복",
    "전기·전자",
    "정보통신",
    "식품가공",
    "인쇄·목재·가구·공예",
    "환경·에너지·안전",
    "농림어업"
]
        result = last.get_recommendation(user_id=req.user_id)
        print("결과",result)
        parsed = result.strip("[]").split(",")
        # job, reason , category= result.split(":")
        return {'job':parsed[0].replace(" ", ""),"reason":parsed[1],"category":category.index(parsed[2].replace(" ", ""))+1}




@app.get("/")
def root():
    return {"message": "Chat API is running"}



