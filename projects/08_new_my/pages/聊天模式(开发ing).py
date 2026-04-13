# 屏蔽 transformers 库的 __path__ 警告（必须在所有 import 之前）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.suppress_warnings import apply, restore
apply()

#=====================1.0 设置基础界面===============================
import streamlit as st
from langchain.tools import tool
from datetime import datetime                       #获取时间的库
import requests
from utils.stock_query import query_stock

# 关键 import 完成后恢复 stderr
restore()

st.set_page_config(
    page_title="聊天模型示例",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.write("这是侧边栏，可以放一些说明或者工具")

st.title(body="智能体应用示例",width="stretch",text_alignment="center")
st.caption(body="使用的是免费本地部署的大模型",width="stretch",text_alignment="center")


prompt=st.chat_input("请输入你的问题：")
#=====================2.0 定义工具函数===========================


def get_coordinates(city_name: str):
    """通过城市名获取经纬度（使用Open-Meteo地理编码API）"""
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=zh&format=json"
    try:
        response = requests.get(geo_url, timeout=10)
        data = response.json()
        if data.get('results'):
            location = data['results'][0]
            return location['latitude'], location['longitude'], location['name']
        else:
            return None, None, None
    except Exception as e:
        print(f"地理编码查询失败: {e}")
        return None, None, None

@tool
def get_weather(location: str) -> str:
    """
    获取指定位置的天气信息。
    
    参数:
    location (str): 需要查询天气的地理位置，例如 "北京" 或 "New York"。
    
    返回:
    str: 指定位置的天气信息。
    """
    # 1. 获取城市坐标
    lat, lon, city_name = get_coordinates(location)
    if lat is None:
        return f"未找到城市: {location}，请检查城市名称是否正确。"
    
    # 2. 构建天气API请求
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m"
        f"&timezone=auto"
    )
    
    try:
        # 3. 发送请求
        response = requests.get(weather_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 4. 解析天气数据
        current = data.get('current_weather', {})
        temp = current.get('temperature')
        wind_speed = current.get('windspeed')
        
        # 获取湿度数据
        hourly = data.get('hourly', {})
        humidity = None
        if hourly.get('relativehumidity_2m'):
            humidity = hourly['relativehumidity_2m'][0]
        
        # 5. 格式化返回结果
        weather_desc = "晴朗" if current.get('weathercode') == 0 else "多云或有雨"
        result = f"{city_name} 当前天气：{weather_desc}，温度 {temp}°C，风速 {wind_speed} km/h"
        if humidity:
            result += f"，湿度 {humidity}%"
        
        return result + "。"
    except requests.exceptions.RequestException as e:
        return f"网络请求错误：{e}"
    except Exception as e:
        return f"处理数据时出错：{e}"

@tool
def get_datetime() -> str:
    """
    获取当前的日期和时间。
    
    返回:
    str: 当前的日期和时间，格式为 "YYYY-MM-DD HH:MM:SS"。
    """

    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_ip_info() -> str:
    """
    获取当前设备的IP地址信息。
    
    返回:
    str: 当前设备的IP地址信息。
    """
    try:
        ip = requests.get("https://api.ip.sb/ip").text.strip()
        return f"当前公网IP：{ip}"
    except:
        return "获取IP失败"
    
@tool
def get_stock(stock_name: str) -> str:
    """查询股票实时行情，支持 A 股、港股、美股"""
    return query_stock(stock_name)

#=====================3.0 模型与代理 初始化======================
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain.embeddings import init_embeddings

# RAG 知识库设置
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'vector_database')
DB_PATH = os.path.abspath(DB_PATH)
emb = init_embeddings("ollama:qwen3-embedding:4b")
db = Chroma(embedding_function=emb, persist_directory=DB_PATH)

@tool
def search_rag(query: str) -> str:
    """检索知识库，对查询进行检索增强"""
    print(f"🔍 正在检索: {query}")
    try:
        docs = db.similarity_search(query=query, k=3)
        if not docs:
            return "知识库中未找到相关内容。"
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知")
            results.append(f"[{i}] {doc.page_content}\n   — {source}")
        return "\n\n".join(results)
    except Exception as e:
        return f"检索失败: {e}"

system_prompt = (
    "当用户询问如下关键词以及相近的词的时候，使用调用 search_rag 工具进行检索增强：\n"
    "LLM Agent、AI Agent、大语言模型智能体、LangChain、检索增强生成 RAG、增强查询、"
    "工具调用 Agent、多智能体系统、自主智能体、智能体规划、智能体记忆机制、大模型推理、"
    "知识库问答、向量数据库、文档检索、上下文增强、智能体框架、大模型应用开发、"
    "智能体工作流、生成式检索增强"
)

#常规 静态模型
model = init_chat_model(model="ollama:qwen3.5:9b ",temperature=0.7)

agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[get_weather, get_datetime, get_ip_info, get_stock, search_rag]
)
#===================4.0 逻辑实现=================================

# 初始化对话历史
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# 显示历史对话
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f'<span style="color: #2ecc71;">{message["content"]}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color: #3498db;">{message["content"]}</span>', unsafe_allow_html=True)

# 处理新消息
if prompt:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(f'<span style="color: #2ecc71;">{prompt}</span>', unsafe_allow_html=True)

    # 存入用户消息
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    # 构建完整对话上下文
    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation_history]

    # AI 回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("思考中..."):
            for chunk, metadata in agent.stream(
                {"messages": messages},
                stream_mode="messages"
            ):
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(f'<span style="color: #3498db;">{full_response}</span>▌', unsafe_allow_html=True)
        message_placeholder.markdown(f'<span style="color: #3498db;">{full_response}</span>', unsafe_allow_html=True)

    # 存入 AI 回复
    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
