import time
import threading

a = 0 
b = 0 

def multi_task():
    global a,b
    a = a+1
    time.sleep(0)   #当前任务 放弃执行的时间片
    b = b+1

    if a!=b:
        print(f"{a} != {b}")
        a = b= 0

if __name__ == "__main__":
    ths =[]
    th = threading.Thread(target=multi_task) 
    ths.append(th)
    th.start()

    for t in ths:
        t.join()

