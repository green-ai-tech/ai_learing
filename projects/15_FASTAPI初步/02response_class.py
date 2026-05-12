"""
1. 服务提供HTML页面服务
    1.1. JSON
    1.2. HTML
2. from fastapi.responses import (
    FileResponse,    # 文件下载服务
    HTMLResponse,    # 网页服务
    JSONResponse,    # 默认
    PlainTextResponse,   # 文本服务
    RedirectResponse,   # 重定向
    StreamingResponse,   # 字节流
)
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn 

app = FastAPI()

@app.get("/pages/index.do", response_class=HTMLResponse)
def html_service():
    page = """
    <html>
        <head>
            <title>我的页面服务</title>
            <meta charset="utf-8"> <!--单标记-->
        </head>
        <body>
            <h1 style="color:red;">我的网页</h1>
            <input type="text" value="靓仔">
        </body>
    </html>
    """
    return page

if __name__ == "__main__":
    uvicorn.run("02response_class:app", host="0.0.0.0", port=7777, reload=True)

"""
网页的设计与专门的工具：
    dreamwaver
"""