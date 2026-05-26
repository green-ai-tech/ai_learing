from langchain.chat_models import init_chat_model
import asyncio

model = init_chat_model(model="ollama:qwen3:4b")

async def  model_task_1():
    print("任务1开始=====")
    result = await model.ainvoke("使用C++写一个冒泡算法。")
    print("=====任务1完成")
    return "任务1完成"

async def  model_task_2():
    print("任务2开始=====")
    result = await model.ainvoke("使用python写一个冒泡算法。")
    print("=====任务2完成")
    return "任务2完成"

async def concurrency_task():
    task1 = asyncio.create_task(model_task_1())
    task2 = asyncio.create_task(model_task_2())
    
    results = await asyncio.gather(task1, task2)
    print("所有任务执行完毕：", results)
    

asyncio.run(concurrency_task())

# ollama: 对并发性能差。（开发）
# vllm：对并发效果好。（生产）