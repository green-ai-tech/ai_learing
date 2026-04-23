import asyncio   # 异步io
from langchain.agents import create_agent

from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "math":{
        "transport": "stdio",
        "command": "python",
        "args": ["/Users/logicye/Code/ai_learning/notebooks/04_LLM_agent/MCP/mcpserver.py", ]   # 路径
    },
})

# 查询可用工具
tools = asyncio.run(client.get_tools())    # 异步操作，需要同步等待
# tools = client.get_tools()
print(tools)

# 使用工具创建代理

from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="qwen3.6:latest",
    base_url="http://192.168.8.21:11434"  
)
agent = create_agent(
    model=llm,
    tools=tools
)

math_response = asyncio.run(agent.ainvoke({
    "messages":[
        {"role": "user", "content": "3加5，再乘以10，等于多少"}
    ]
}))

# print(math_response["messages"][-1].content)
print(math_response)