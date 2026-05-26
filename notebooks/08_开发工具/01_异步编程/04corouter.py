import asyncio
import time 

# 1. 使用函数实现多任务

async def complex_task(delay): # 协程函数 多任务
    time.sleep(delay)
    return "完成协程"

#2. 事件循环

if 