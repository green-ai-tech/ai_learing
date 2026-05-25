import time
import threading

def task(name:int,ts:str):
    print(f"线程={name}")
    time.sleep(1)

if  __name__ == "__main__":
    #创建多个任务
    ths = []    #存放线程
    for i in range(5):
        th = threading.Thread(target=task,args=[i+1,"线程任务"]) #创建线程
        ths.append(th)
        th.start()

    #等待线程结束
    for t in ths:
            t.join()   