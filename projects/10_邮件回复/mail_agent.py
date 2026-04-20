# ================ 1. 状态 =================
from typing import TypedDict, Literal      # 状态使用字典格式:basemodel

# 定义邮件分类的状态
class EmailClassification(TypedDict):
    intent : Literal["question", "bug", "billing", "feature", "complex"]
    urgency : Literal["low", "medium", "high", "critical"]
    topic: str
    summary : str
# 定义邮件的状态(包括分类)
class EmailAgentState(TypedDict):
    # 邮件的原始数据
    email_content : str
    sender_email : str
    email_id : str

    # 分类结果
    classification : EmailClassification | None
    #搜索的数据/API接口数据
    search_results: list[str] | None
    customer_history : dict | None
    #生成内容
    draft_response : str | None
    messages : list[str] | None

# ================ 2. 节点 =================
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="ollama:gemma4:e4b",
    temperature = 0,

)
# 读取邮件
def read_email(state: EmailAgentState) -> dict:
    # 假设省略实际的读取过程
    # 直接返回读取的内容
    return {
        "messages":[
            HumanMessage(content=f"正在处理邮件:{state["email_content"]}")
        ]
    }
# 意图分类
def classify_intent(state:EmailAgentState) -> Command[
    Literal["search_document","human_review", "draft_response", "bug_tracking"]
    ]:
    """使用LLM对邮件意图和紧急程度进行分类，然后根据分类结果进行路由"""
    structed_llm = llm.with_structured_output(EmailClassification)


    classification_prompt = F"""
    分析此客户邮件并进行分类：
    邮件内容：{state["email_content"]}
    发件人：{state["sender_email"]}
    请提供分类结果，包括意图，紧急程度，主题和摘要
    """
    classification = structed_llm.invoke(classification_prompt)

    # 根据分类结果
    if classification["intent"] == "billing" or classification["urgency"] == "critical":
        goto = "human_review"
    if classification["intent"] in ["question", "feature"]:
        goto = "search_document"
    if classification["intent"] == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(
        update={"classification": classification},
        goto = goto,          # 跳转的节点（节点名）
    )
# 文档搜索
def search_document(state:EmailAgentState)->Command[Literal["draft_response"]]:
    """在知识库中搜索相关信息"""
    classification = state["classification"]
    query = f"{classification.get("intent","")} {classification.get("topic","")}"
    
    # 直接模拟搜索结果
    try:
        search_results = [
            "通过【设置】->【安全】->【更改密码】",
            "密码长度至少12字符",
            "包含大小写字母，数字和下划线",
        ]
    except Exception as e:
        search_results = [f"搜索服务暂时不可用:{str(e)}"]
    return Command(
        update={
            "search_results": search_results,
        },
        goto="draft_response"
    )
# bug跟踪
def bug_tracking(state:EmailAgentState)->Command[Literal["draft_response"]]:
    """创建或更新缺陷追踪工单"""
    
    ticket_id = "bug_12345"
    return Command(
        update={
            "search_results": [F"缺陷工单{ticket_id}已创建"],
        },
        goto="draft_response"
    )
# 邮件草稿
def draft_response(state:EmailAgentState)->Command[Literal["human_review","send_reply"]]:
    """利用上下文生成响应，并根据质量进行路由"""
    classification = state.get("classification",{})
    context_sections = []
    # 获取搜索结果
    if state.get("search_results"):

        # 获取历史咨询信息
        formatted_doc = [F"- {doc}" for doc in state["search_results"]]
        context_sections.append(F"相关文档: \n{formatted_doc}")
    if state.get("customer_history"):    
        context_sections.append(F"客户等级:{state["customer_history"].get("tier", "standard")}")    
    
    # 根据前面两个信息生成提示词，交给大模型
    draft_prompt = F"""
    请草拟一封回复此客户邮件的信函:
    {state["email_content"]}

    邮件意图：{classification.get("intent", "unknown")}
    邮件紧急程度：{classification.get("urgency", "medium")}

    {chr(10).join(context_sections)}       # 10的ASCII码是换行
    指南：
    - 保持专业且乐于助人
    """

    # 调用大模型生成邮件回复内容
    response = llm.invoke(draft_prompt)
    # 根据状态自动回复邮件
    needs_review = (
        classification.get("urgency") in ["high", "critical"] or 
        classification.get("intent") == "complex"
    )
    goto = "human_review" if needs_review else "send_reply"
    return Command(
        update={
            "draft_response": response.content
        },
        goto=goto
    )
# 人工审核
def human_review(state:EmailAgentState)->Command[Literal["send_reply", END]]:
    """使用中断暂停以进行人工审核，并根据决策进行路由"""
    classification = state.get("classification",{})
    # 调用中断interrupt(继续时回归到原来状态)
    human_decision = interrupt({
        "email_id":state.get("email_id", ""),
        "originalemial":state.get("email_content", ""),
        "draft_response":state.get("draft_response", "") ,
        "urgency":classification.get("urgency"),
        "intent":classification.get("intent"),
        "action":"请审核并批准/编辑并回复"
    })

    if human_decision.get("approved"):
        return Command(
            update={
                "draft_response": human_decision.get("edited_response", state.get("draft_response", ""))
            },
            goto="send_reply"
        )
    else:
        return Command(
            update={},
            goto=END
        )
# 发送邮件
def send_reply(state:EmailAgentState)->dict:
    """发送邮件回复"""
    print(f"正在发送邮件回复：{state["draft_response"][:100]}...")
    return {}       # 最后返回的内容

# ================ 3. 链接 =================
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

# 创建图
workflow = StateGraph(EmailAgentState)

# 添加节点
workflow.add_node("read_mail", read_email)
workflow.add_node("classify_intent", classify_intent)

workflow.add_node("search_document", search_document)

workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge( START, "read_mail")
workflow.add_edge( "read_mail", "classify_intent")

workflow.add_edge("send_reply" , END)

memory = MemorySaver()

agent = workflow.compile(checkpointer=memory)



# ================ 4. 测试 =================
state_init ={
    "email_content": "我的订阅被重复扣费，情况紧急",
    "sender_email" : "caiwei@emial.com",
    "email_id"     : "email_123",
    "messages"     : []
} 
config = {
    "configurable" : {"thread_id":"customer_Fu"}
}
result = agent.invoke(state_init, config=config)
# 打印流程终端的状态
print(f"人工审核中断：{result["__interrupt__"]}")

# 模拟人工审核
human_response = Command(
    resume={
        "approved":True,
        "edited_response":"哈哈哈 你受着先"
    }
)

last_result = agent.invoke(human_response, config)