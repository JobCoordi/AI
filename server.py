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
        result = last.get_recommendation(user_id=req.user_id)
        job, reason = result.split(":")
        return {'job':job,"reason":reason}




@app.get("/")
def root():
    return {"message": "Chat API is running"}
