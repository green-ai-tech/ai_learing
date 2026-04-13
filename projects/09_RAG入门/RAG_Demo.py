from langchain.chat_models import init_chat_model              #模型（可选择）
from langchain.agents import create_agent
from langchain.tools import tool
#from langchian.cores import 弃用
from langchain_chroma import Chroma
from langchain.embeddings import init_embeddings


emb = init_embeddings("ollama:qwen3-embedding:0.6b")
db = Chroma(embedding_function=emb,persist_directory="/Users/logicye/Code/ai_learning/notebooks/04_大模型agent/vdb")
#=======================工具========================
@tool
def search_rag(query:str) ->str :
    """
    检索知识库
    参数：
        query： 检索的内容
    返回：
        检索增强的内容
    """
    #RAG操作
    print("正在做检索...")

    docs = db.max_marginal_relevance_search(query=query,k=3,fetch_k=20)
    return "\n\n".join([doc.page_content for  doc in docs])




#======================智能体=======================
model  = init_chat_model(
    model="ollama:qwen3.5:9b",
    max_tokens = 512,
    temperature = 0.7,
    base_url = "http://127.0.0.1:11434"
)


# 创建一个智能体
system_prompt = ("你是一个助手，回答问题前完成下面操作：\n"
                 "1. 调用 search_rag 工具进行检索增强")

agent = create_agent(
    model           =  model,
    system_prompt   =  system_prompt,        #虚拟角色
    tools           = [search_rag],        #功能

    middleware      =[],
    response_format =   None,
    checkpointer    =   None,
    state_schema    =   None,
    context_schema  =   None,
    store           =   None
)

#RAG 查询
response = agent.invoke(
    {"messages":[
        {
            "role":"user",
            "content":"什么是Tool"
        }
    ]})

# 打印回复
print(response["messages"][-1].content)
