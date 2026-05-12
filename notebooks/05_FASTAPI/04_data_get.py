from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/pages/data_get",response_class=HTMLResponse)
def get_service(name:str = "默认"):
    return f"""
<body>
    <h1>{name}</h1>
</body>
"""


@app.post("/post/index.html",response_class=HTMLResponse)
def post_service(name:str = "默认", age: int = 0):
    return f"""
    <body>
        {name}<br>
        {age}
    </body>
"""


if __name__ == "__main__":
    uvicorn.run("04_data_get:app",host="0.0.0.0",port=7777,reload=True)

    
    