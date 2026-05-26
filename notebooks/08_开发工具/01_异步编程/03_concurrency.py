import time     # 计算时间
import threading   # 多任务
import requests    # 爬网页：HTTP/HTTPS协议

def crawl_page():
    response = requests.get("https://baidu.com")
    if response.status_code == 200:
        print(F"爬取成功：{len(response.content)}字符", )
    else:
        print("爬取失败")

NUMS = 10

if __name__ == "__main__":
    # 1. 同步爬取
    start_time = time.time()
    for i in range(NUMS):
        crawl_page()
    
    end_time = time.time()
    print(F"爬取{NUMS}次的时间：{end_time - start_time}")
    
    
    # 2. 异步爬取
    start_time = time.time()
    ths = []
    for i in range(NUMS):
        th = threading.Thread(target=crawl_page)
        ths.append(th)
        th.start()
    
    for t in  ths:
        t.join()
    
    end_time = time.time()
    print(F"爬取{NUMS}次的时间：{end_time - start_time}")