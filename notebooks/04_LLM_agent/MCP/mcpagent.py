import asyncio    #异步io
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "math":{
        "transport":"stdio",
        "command":"python",
        "args":["/Users/logicye/Code/ai_learning/notebooks/04_大模型agent/MCP/mcpserver.py",]
    }
})

#查询可用工具
tools = asyncio.run(client.get_tools())

print(tools)


#使用工具创建代理·

agent = create_agent.ai({
    "messages":[
        "role":"user",
        "content":"3加5，再乘以10，等于多少？"
    ]
    
})