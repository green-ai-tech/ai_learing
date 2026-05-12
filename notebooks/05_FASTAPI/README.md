# 1. 项目架构
    业务模块
    1.1 HTML (CSS + javascript) 页面文件
            |
            |
    1.2 后段的服务模块 （FastAPI）
            |
            |
    1.3 业务模块（oop,面向对象）
            |
            |
    1.4 智能体（面向对象）
            |
            |
    1.5 数据库,知识库，Skill模块（Tool，MCP，...）





# 2. 后段（CGI【python】 + WSGI【jonggo,Flask】 + ASGI 【FastAPI】）

    2.1. 环境与编程模式
        pip install "fastapi[all]"
        验证：执行uvicorn工具

    2.2 第一个FastAPI程序
        ASGI 的程序 = web 服务器程序

        使用浏览器访问
            浏览器 ----url------ web 服务器（调用FastAPI程序）
                                    ｜
                                容器CGI（Java，C++，Python，Go ，Swift ,c# ...）
                                                    |
                                                    uvicorn(Nginx[支持ASGI协议：模块])

        uvicorn有两种使用方法：
            1. API
            2. 指令








># 3. 前端（web = Html + css + JavaScript [DHTML])
># 4. 前端（wps（Excel：VBA宏： 大模型做数据分析） + 通信工具（微信 + 钉钉）


# 与项目相关
    1. 每个页面的命名规则， 使用excel 表格设计
        1. 目录设计
        2. 扩展名
        3. 页面名

"""
GET / HTTP/1.1
Host: 127.0.0.1:9999
Connection: keep-alive
sec-ch-ua: "Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "macOS"
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Sec-Fetch-Site: none
Sec-Fetch-Mode: navigate
Sec-Fetch-User: ?1
Sec-Fetch-Dest: document
Accept-Encoding: gzip, deflate, br, zstd
Accept-Language: zh-CN,zh;q=0.9,en;q=0.8
Cookie: ajs_anonymous_id=6cba5c02-64db-4b2b-a0d8-6d85ac13e860; _streamlit_xsrf=2|55983433|97bf8b87f53f219874faaf626c32f6ec|1777815147; csrf_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzgyOTg4NzksInN1YiI6ImRlYmY0MjhjLWVmNGQtNDkxNy04YzJhLTA2OWQyMzM3ZTcwNCJ9.h6wyJ1hMfwmSz0OvEaw9TwSAQjrTEPcrQt6Y6p067k8; access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZGViZjQyOGMtZWY0ZC00OTE3LThjMmEtMDY5ZDIzMzdlNzA0IiwiZXhwIjoxNzc4Mjk4ODc5LCJpc3MiOiJTRUxGX0hPU1RFRCIsInN1YiI6IkNvbnNvbGUgQVBJIFBhc3Nwb3J0In0.atiJmYYHE5ASTjuCFFWGpnNX58ZWJQFJrV-VDu7oq68; refresh_token=fbe2f15ef9b94a168ce2947ec1e76bd62abe500caa0e5a23c9eaf2b0412a4e12acc4c1f4a2d02f432ffbb9670bb9b678a5952962449bbd4ee97cfb033e8d73a3


"""