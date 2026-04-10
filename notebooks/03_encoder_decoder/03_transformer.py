# 通过transformer 来认识大模型
#1. 通过transformers 认识大模型的结构

from transformers import pipeline
import torch


pipe = pipeline(
    task="text-generation",
    model="/Volumes/AI/models/local/Qwen3.5-2B",
    trust_remote_code=True
)

message = [{"role":"user","content":"你好！"}]

#  直接使用pipe 推理
outputs = pipe(message)
print(outputs)
print(type)
print("="*80)


from transformers.pipelines.text_generation import Chat   
chat = Chat(message)
print(chat)
print(chat.messages)

input = pipe.preprocess(chat)
print("="*80)
print(input)

outputs = pipe.forward(input)
print("="*80)
print(outputs)


results = pipe.postprocess(outputs)
print("="*80)
print(results)
