import socket

def get_baidu_homepage():
    # 1. 创建 socket 对象
    # socket.AF_INET: IPv4 地址族
    # socket.SOCK_STREAM: TCP 协议
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # 2. 连接到百度服务器
        # 百度域名 www.baidu.com，HTTP 端口为 80
        server_address = ('www.baidu.com', 80)
        print(f"正在连接到 {server_address}...")
        client_socket.connect(server_address)
        print("连接成功！")
        
        # 3. 构造 HTTP 请求
        # 发送 GET 请求获取首页
        request = (
            "GET / HTTP/1.1\r\n"
            "Host: www.baidu.com\r\n"
            "User-Agent: Python Socket Client\r\n"
            "Accept: text/html,application/xhtml+xml\r\n"
            "Connection: close\r\n"  # 请求完成后关闭连接
            "\r\n"
        )
        
        print("发送 HTTP 请求...")
        client_socket.send(request.encode('utf-8'))
        
        # 4. 接收响应数据
        print("接收响应数据...")
        response_data = b''
        while True:
            # 每次接收 4096 字节
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        # 5. 解码响应内容
        response_text = response_data.decode('utf-8')
        
        # 6. 分离响应头和响应体
        # 找到第一个空行（响应头结束标记）
        header_end = response_text.find('\r\n\r\n')
        if header_end != -1:
            headers = response_text[:header_end]
            body = response_text[header_end + 4:]
            print("\n" + "="*50)
            print("响应头:")
            print("="*50)
            print(headers)
            print("\n" + "="*50)
            print("响应体前500个字符:")
            print("="*50)
            print(body[:500])  # 只显示前500个字符
        else:
            print("响应数据:")
            print(response_text[:1000])
        
        return response_text
        
    except socket.error as e:
        print(f"socket 错误: {e}")
        return None
    finally:
        # 5. 关闭连接
        client_socket.close()
        print("\n连接已关闭")

if __name__ == "__main__":
    get_baidu_homepage()