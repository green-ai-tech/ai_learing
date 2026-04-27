#=======================1. 实现中间件==========================
from typing import Any, Callable
from langchain.agents.middleware import (AgentMiddleware,AgentState,ModelRequest,ModelResponse,ToolCallRequest,)
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain_core.tools import BaseTool

class Middleware_my(AgentMiddleware):
    state_schema: AgentState
    tools: list[BaseTool]
    
    def before_agent(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        print("正在执行before_agent()函数")
        return super().before_agent(state, runtime)
    
    def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        print("正在执行before_model()函数")
        return super().before_model(state, runtime)
    
    def wrap_model_call(self, request: ModelRequest[None], handler: Callable[[ModelRequest[None]], ModelResponse[Any]]) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        print("正在执行wrap_model()函数")
        return handler(request)
    
    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
        print("正在执行wrap_tool()函数")        
        return handler(request)
    
    def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        print("正在执行after_model函数")        
        return super().after_model(state, runtime)
    
    def after_agent(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        print("正在执行after_agent()函数")        
        return super().after_agent(state, runtime)
    

# ========================2. 创建工具==========================
from langchain.tools import tool
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气信息

    Args:
        city: 要查询天气的城市名
    返回：
        指定城市的天气
    """
    print("###", "工具被调用")
    return  F"{city}的天气良好，温度零下50°"

# ========================3. 创建agent==========================
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="ollama:qwen3.6:latest",
    temperature=0.5,
    base_url="http://192.168.8.21:11434"
)

agent = create_agent(
    model=model,
    middleware=[Middleware_my()],
    tools=[get_weather],
    system_prompt="你是AI助手，给用户提供查询与帮助。"
)

#========================4. 观察使用============================
from langchain_core.messages import HumanMessage


if __name__=="__main__":

    print("密切观察 👀")
    response = agent.invoke({
    "messages": [HumanMessage(content="今天杭州天气如何？")]
    })
    for msg in response["messages"]:
        msg.pretty_print()


