import socket
import ssl

def get_baidu_homepage_https():
    """
    使用 SSL 连接百度 HTTPS 首页
    """
    # 创建普通 socket
    plain_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # 连接到百度的 HTTPS 端口
        server_address = ('www.baidu.com', 443)
        print(f"正在连接到 {server_address}...")
        plain_socket.connect(server_address)
        print("TCP 连接成功！")
        
        # 包装为 SSL socket
        context = ssl.create_default_context()
        ssl_socket = context.wrap_socket(plain_socket, server_hostname='www.baidu.com')
        print("SSL/TLS 握手成功！")
    
        # 构造 HTTPS 请求
        request = (
            "GET / HTTP/1.1\r\n"
            "Host: www.baidu.com\r\n"
            "User-Agent: Python Socket Client/1.0\r\n"
            "Accept: text/html,application/xhtml+xml\r\n"
            "Accept-Language: zh-CN,zh;q=0.9\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        
        print("发送 HTTPS 请求...")
        ssl_socket.send(request.encode('utf-8'))
        
        # 接收响应
        print("接收响应数据...")
        response_data = b''
        while True:
            chunk = ssl_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        # 解码响应
        response_text = response_data.decode('utf-8')
        
        # 分离响应头和响应体
        header_end = response_text.find('\r\n\r\n')
        if header_end != -1:
            headers = response_text[:header_end]
            body = response_text[header_end + 4:]  # 跳过\r\n\r\n四个字节。
            
            print("\n" + "="*60)
            print("HTTP 响应头:")
            print("="*60)
            print(headers)
            
            print("\n" + "="*60)
            print("网页内容预览（前800个字符）:")
            print("="*60)
            print(body[:800])
            
            # 保存完整响应到文件
            with open('baidu_homepage.html', 'w', encoding='utf-8') as f:
                f.write(body)
            print("\n完整网页已保存到 baidu_homepage.html")
        else:
            print(response_text[:1000])
        
        return response_text
        
    except socket.error as e:
        print(f"socket 错误: {e}")
        return None
    except ssl.SSLError as e:
        print(f"SSL 错误: {e}")
        return None
    finally:
        ssl_socket.close()
        print("\n连接已关闭")

if __name__ == "__main__":
    get_baidu_homepage_https()
    