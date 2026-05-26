import asyncio
import requests

async def get_page(nums):   # 协程
    for i in range(nums):
        requests.get("https://baidu.com")


async def call_task(nums):
    print("任务调用前...")
    await get_page(nums)        # 协程调用 asyncio.run  //  await: 阻塞等待协程执行，返回结果
    print("任务调用后")
    
    return F"任务完成，调用{nums}次"

async def main():
    # 无并发
    result1 = await call_task(5)
    result2 = await call_task(3)
    print(result1, result2)
    # 并发调用
    task3 = asyncio.create_task(call_task(5))
    task4 = asyncio.create_task(call_task(3))
    results = await asyncio.gather(task3, task4)  #  并发

if __name__ == "__main__":
    # 无并发
    # task1 = call_task(5)   # 创建协程对象
    # task2 = call_task(10)  # 

    # result1 = asyncio.run(task1)
    # result2 = asyncio.run(task2)
    # print(result1, result2)
    
    asyncio.run(main())