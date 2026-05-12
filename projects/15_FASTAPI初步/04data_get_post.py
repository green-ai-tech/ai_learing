from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def get_service(name: str = "默认"):
    return F"""
<body>
    <h1>{name}</h1>
</body>
"""

@app.post("/post/index.html", response_class=HTMLResponse)
def post_service(name: str = Form("默认"), age: int = Form(0)):
    
    return F"""
    <body>
        {name} <br>
        {age}
    </body>
"""



if __name__ == "__main__":
    uvicorn.run("04data_get_post:app", reload=True)