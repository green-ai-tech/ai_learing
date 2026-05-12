from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/main/index.html")
def main_service(request: Request):
    now = datetime.now()
    date_time = f"{now.year}/{now.month}/{now.day}"

    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"date_time": date_time},
    )


class PromptModel(BaseModel):
    query: str = "你好"


@app.post("/chat.service")
def ajax_service(query: PromptModel):
    return {
        "ai_messages": f"这是来自智能体的智能答复：{query.query}",
    }


if __name__ == "__main__":
    uvicorn.run("templates:app", host="127.0.0.1", port=8848)
