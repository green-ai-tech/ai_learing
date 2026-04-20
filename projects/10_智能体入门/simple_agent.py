from langchain.tools import tool
from langchain.chat_models import init_chat_model

#=================1. 定义一个模型==================
model = init_chat_model("ollama:gemma4:e4b", temperature=0.3)

#=================2. 定义一个工具===================
@tool
def add(a: int, b: int) -> int:
    """
    将 'a' 和 'b' 相加
    
    参数：
        a: 第一个整型
        b: 第二个整型
    """
    return a + b

@tool
def mutly(a: int, b: int) -> int:
    """
    将 'a' 和 'b' 相乘
    
    参数：
        a: 第一个整型
        b: 第二个整型
    """
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """
    将 'a' 和 'b' 相除
    
    参数：
        a: 第一个整型
        b: 第二个整型
    """
    return a / b

# 工具绑定
tools = [add, mutly, divide]
model_with_tools = model.bind_tools(tools)

# 工具名字映射（必须加！否则找不到工具）
tool_by_name = {tool.name: tool for tool in tools}

#=================3. 定义状态======================
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

#=================4. 定义节点======================
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
# 大模型调用节点
system_prompt = "你是一位计算师，负责对一组输入执行算术运算"
def llm_call(state: MessagesState):
    """LLM决定是否调用工具"""
    print(">>>> LLM 调用中")
    outputs = model_with_tools.invoke([SystemMessage(system_prompt)] + state["messages"])

    return {
        "messages": [outputs],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# 工具调用节点
def tool_node(state: MessagesState):
    """执行工具调用"""
    result = []
    print("\t 执行工具...")
    
    # 获取最后一条AI消息
    last_msg = state["messages"][-1]
    
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool = tool_by_name[tool_name]
        observation = tool.invoke(tool_call["args"])
        
        # 把工具结果包装成 ToolMessage（必须！）
        from langchain_core.messages import ToolMessage
        result.append(ToolMessage(
            content=str(ToolMessage(observation,tool_call_id = tool_call["id"])),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": result}

#=================5. 定义边 编译===================
from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END

# 条件分支：是否需要调用工具
def branch(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # 判断是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

# 构建图
agent_b = StateGraph(MessagesState)

# 添加节点
agent_b.add_node("llm_call", llm_call)
agent_b.add_node("tool_node", tool_node)

# 构建流程
agent_b.add_edge(START, "llm_call")
agent_b.add_conditional_edges(
    "llm_call",
    branch,  # 分支函数
    {
        "tool_node": "tool_node",
        END: END
    }
)
agent_b.add_edge("tool_node", "llm_call")  # 工具执行完回到LLM

agent = agent_b.compile()

#=================6. 调用智能体===================
# 修正导入 & 修正调用
from langchain_core.messages import HumanMessage  # 修复导入

# 输入问题
messages = [HumanMessage(content="3 和 4 相加，再乘以5,除以4")]
result = agent.invoke({"messages": messages, "llm_calls": 0})


for m in result["messages"]:  
    m.pretty_print()        