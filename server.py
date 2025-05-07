from fastapi import FastAPI
from pydantic import BaseModel
from model import TranslationModel

app = FastAPI()
translator = TranslationModel(model_name="gpt-4o")

class MessageRequest(BaseModel):
    user_id: str
    content: str

@app.post("/chat")
def chat(req: MessageRequest):
    response = translator.translate(user_id=req.user_id, content=req.content)
    return {"user_id": req.user_id, "response": response}

@app.get("/")
def root():
    return {"message": "Chat API is running"}
