"""
1. 服务提供HTML 页面服务
    1.1 JSON
    1.2 HTML

2. from fastapi.responses import(
    FileResponse,           # 文件下载服务
    HTMLResponse,           # 网页服务
    JSONResponse,           # 默认
    PlainTextResponse,      # 文件


)

"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse,JSONResponse
import uvicorn

app = FastAPI()

@app.get("/pages/index.do",response_class=HTMLResponse)
def html_service():
    page = """
    <html>
        <head>
            <title>我的页面服务</title>
            <meta charset = "utf-8">
        </head>

        <body>
            <h1 style="color:red">我的网页</h1>
            <input type="text" value ="靓仔">
        </body>
    </html>
    """
    return page


if __name__=="__main__":
    uvicorn.run("02_response_class:app",host="0.0.0.0",port=7777,reload=True)
