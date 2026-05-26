import time
import threading


a = 0
b = 0
THREAD_COUNT = 1000
locker = threading.Lock()


def multi_task():
    global a, b

    a = a + 1
    time.sleep(0)   # 当前任务放弃执行的时间片，让其他线程有机会插入执行
    b = b + 1

    if a != b:
        print(f"未加锁: a={a}, b={b}")


def multi_task_locker():
    global a, b

    with locker:
        a = a + 1
        time.sleep(0)   # 即使这里让出时间片，其他线程也拿不到锁
        b = b + 1

        if a != b:
            print(f"已加锁: a={a}, b={b}")


def run_task(name, target):
    global a, b

    a = 0
    b = 0
    ths = []

    for _ in range(THREAD_COUNT):
        th = threading.Thread(target=target)
        ths.append(th)
        th.start()

    for t in ths:
        t.join()

    print(f"{name}最终结果: a={a}, b={b}")


if __name__ == "__main__":
    run_task("未加锁", multi_task)
    run_task("已加锁", multi_task_locker)
