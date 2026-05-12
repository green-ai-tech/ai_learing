# 1. 项目架构
    业务模块
        1.1. HTML(CSS+javascrip)页面文件。
                    | 
        1.2. 后端的服务模块（FastAPI）
                    |
        1.3. 业务模块（面向对象）
                    |
        1.4. 智能体（面向对象）
                    |
        1.5. 数据库，知识库，Skill模块（Tool， MCP， ...）



# 2. 后端-FastAPI（CGI[python] + WSGI[Djongo, Flask] + ASGI[FastAPI]）
    2.1. 环境与编程模式
        pip install "fastapi[all]" 
        验证：执行uvicorn工具
            不能执行原因：工具的目录不在path环境变量。
                C:\Program Files\Python313\Scripts
                C:\Users\ThinkPad\AppData\Roaming\Python\Python313\Scripts
    2.2. 第一个FastAPI程序
        ASGI的程序 == Web 服务器程序
        使用浏览器访问
            浏览器 ---url--->  Web服务器（调用FastAPI程序）
                                    |
                                   容器CGI（Java， C++， Python， Go， Swift, C#）
                                                      |
                                                   uvicorn(Ngnix[支持ASGI协议：模块])
        uvicorn有两种使用方法：
            1. API
            2. 指令
    2.3. 实现Web页面服务
    
    2.4. 理解HTTP协议工作原理
        get与post的区别

```shell
GET /main/index.html HTTP/1.1    # 请求行（请求头）：请求方法 请求资源URL 协议HTTP1.1

Host: 127.0.0.1:9999             # 协议头 ，每一行一个头， ：分成两个部分
Connection: keep-alive
sec-ch-ua: "Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Sec-Fetch-Site: none
Sec-Fetch-Mode: navigate
Sec-Fetch-User: ?1
Sec-Fetch-Dest: document
Accept-Encoding: gzip, deflate, br, zstd
Accept-Language: zh-CN,zh;q=0.9
        # 一个空行  （空行）
        # 最后的空行（协议体Body）： get请求方法，没有body
```
----
```shell
POST /main/index.html HTTP/1.1
Host: 127.0.0.1:9999
Connection: keep-alive
Content-Length: 23
Cache-Control: max-age=0
sec-ch-ua: "Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
Upgrade-Insecure-Requests: 1
Content-Type: application/x-www-form-urlencoded
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36
Origin: null
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: navigate
Sec-Fetch-User: ?1
Sec-Fetch-Dest: document
Accept-Encoding: gzip, deflate, br, zstd
Accept-Language: zh-CN,zh;q=0.9
      # 空行
name=%E9%9D%93%E4%BB%94     # HTTP 请求体（Request Body）
```
    - 响应码：
      - 1: 请求中 
      - 2：请求成功
      - 3：继续请求中（二次请求）
      - 4：客户端错误
      - 5：服务器错误

    2.5. get方法的数据传递
        http://127.0.0.1:8000/?name=靓仔
        querystring：查询字符串：?name=value&age=20   # 因为get没有http body只能使用querystring。
    

># 3. 前端（Web = HTML + CSS + Javascript(DHTML)）
># 4. 前端（WPS（Excel：VBA宏：大模型数据分析） + 通信工具（微信 + 钉钉））



# 与项目相关的
    1. 每个页面的命名规则，使用excel表格设计。
       1. 目录设计
       2. 扩展名
       3. 页面名