# 1. 引入模块

from fastapi import FastAPI
import uvicorn

# 2. 创建 FastAPI 对象（负责底层的HTTP 协议的处理，而且实现ASGI 的协议细节）

app = FastAPI()

# 3. 定义用户访问的URL（get,post）
@app.get("/main/index.html")

# 4. 提供URL的服务
def url_service():
    """
    返回:
        1. 返回页面
            1.1 直接HTML 内容
            1.2 Jinjia 模版

        2. 返回json字典格式（数据） 
            2.1 字典格式
    """

    return {
        "messages":" 喂 靓仔"
    }

# 5. 启动服务
"""
uvicorn module : FastAPI的名字 -- host 0.0.0.0 -- port 7777 -- reload

uvicorn 01_env_model:app --host 0.0.0.0 --port 7777 --reload

APP = module_name:app_name
-- host = 0.0.0.0 万有地址 （表示本机的所有IP地址）
-- reload ：开发模式（不需要重启服务）

"""
# 6. 使用浏览器访问
 #http://127.0.0.1:7777/main/index.html

if __name__ == "__main__":
    uvicorn.run("01_env_model:app",host="0.0.0.0",port=7777,reload=True)