from langchain.chat_models import init_chat_model
import asyncio

model = init_chat_model(model="ollama:qwen3:4b")

async def  model_task_1():
    print("任务1开始=====")
    result = await model.ainvoke("使用C++写一个冒泡算法。")
    print("=====任务1完成")
    return result

async def  model_task_2():
    print("任务2开始=====")
    result = await model.ainvoke("使用python写一个冒泡算法。")
    print("=====任务2完成")
    return result

async def concurrency_task():
    task1 = asyncio.create_task(model_task_1())
    task2 = asyncio.create_task(model_task_2())
    
    pending = [task1, task2]
    
    # 并发的异常处理，及时处理，延时处理
    while pending: 
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)  # 第一个调用完成，则结束
        # done返回完成的任务，pending返回没有完成任务
        print("完成任务：", done)
        print("没有完成任务：", pending)  # 继续使用wait处理
        # 怎么处理完成的任务
        for task in done:
            print(await task) # 取值
        # 基础等待其余没有完成的任务
        # done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)  # 第一个调用完成，则结束

asyncio.run(concurrency_task())

# ollama: 对并发性能差。（开发）
# vllm：对并发效果好。（生产）