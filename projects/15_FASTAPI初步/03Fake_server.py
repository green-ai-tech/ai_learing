"""
写一个Web服务器，然后使用浏览器访问我们的服务器，观察浏览器是怎么使用HTTP访问网页。
    物理层/链路层 ETHE
    网络/传输/协议层：IP / TCP | UDP
    应用/会话层(FTP， POP3，SMTP， SSH， TELNET， 【HTTP/HTTPS】)
"""

import socket

# 1. 创建socket：套接字编程（插座：net socket， file socket， 匿名sokcet）：双工管道
server = socket.socket(
    socket.AF_INET,   # 协议族PF = Protocol Family
    socket.SOCK_STREAM   # 使用字节流
)

# 2. 设置网络选项
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 地址与端口可以冲突, 其中的1表示True

# 3. 绑定IP（识别主机：4字节整数：IP地址简记四字节整数）与端口（识别程序）
try:
    server.bind(("127.0.0.1", 9999))
    # 4. 监听是否有人连接
    server.listen(5)   # 排队的数量
    
    # 5. 抓取连接的客户端（阻塞：有人连接，马上返回，没有人连接，就是一直等待）
    client_socket, client_addr = server.accept()   # 很多人连接，应该使用训练
    print("有人连接：", client_addr)
    
    # 6. 抓取浏览器请求的数据（观察数据，了解HTTP协议是怎么通信）
    recv_data = client_socket.recv(4096).decode("utf-8")  # 4096可以是任何整数，表示接收的数据大小。并且使用decode直接解码成字符串
    print("浏览器请求的数据：")
    print(recv_data)    
except Exception as e:
    print("服务错误：", str(e))
finally:
    server.close()  # 正常执行与异常发生都会执行。