# 屏蔽 transformers 库的 __path__ 警告（必须在所有 import 之前）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.suppress_warnings import apply, restore
apply()

#=====================1.0 设置基础界面===============================
import streamlit as st
from langchain.tools import tool
from datetime import datetime
import requests
from utils.stock_query import query_stock

# 关键 import 完成后恢复 stderr
restore()

st.set_page_config(
    page_title="智能体助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 macOS 毛玻璃主题
from themes.macos_theme import render_theme, welcome_card
render_theme()

# 侧边栏 - 工具面板
with st.sidebar:
    st.markdown("### 🛠️ 工具集")
    st.markdown("""
    | 工具 | 说明 |
    |------|------|
    | 🌤️ 天气 | 查询城市实时天气 |
    | 📅 时间 | 获取当前日期时间 |
    | 🌐 IP | 查询公网 IP 地址 |
    | 📈 股票 | A股/港股/美股行情 |
    | 📚 RAG | 知识库检索增强 |
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("使用本地 Ollama 大模型驱动")

# 标题
st.markdown("### 🤖 智能体助手", unsafe_allow_html=True)
st.markdown('<span style="color: #6e6e73; font-size: 13px;">本地部署大模型 · 多工具智能体</span>', unsafe_allow_html=True)


prompt = st.chat_input("请输入你的问题：")

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
    """获取指定位置的天气信息"""
    lat, lon, city_name = get_coordinates(location)
    if lat is None:
        return f"未找到城市: {location}，请检查城市名称。"

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m"
        f"&timezone=auto"
    )

    try:
        response = requests.get(weather_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data.get('current_weather', {})
        temp = current.get('temperature')
        wind_speed = current.get('windspeed')

        hourly = data.get('hourly', {})
        humidity = hourly.get('relativehumidity_2m', [None])[0]

        weather_desc = "晴朗" if current.get('weathercode') == 0 else "多云或有雨"
        result = f"{city_name} 当前天气：{weather_desc}，温度 {temp}°C，风速 {wind_speed} km/h"
        if humidity:
            result += f"，湿度 {humidity}%"

        return result + "。"
    except Exception as e:
        return f"查询失败：{e}"

@tool
def get_datetime() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_ip_info() -> str:
    """获取当前设备的公网 IP"""
    try:
        ip = requests.get("https://api.ip.sb/ip").text.strip()
        return f"当前公网 IP：{ip}"
    except:
        return "获取 IP 失败"

@tool
def get_stock(stock_name: str) -> str:
    """查询股票实时行情，支持 A 股、港股、美股"""
    return query_stock(stock_name)

#=====================3.0 模型与代理 初始化======================
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain.embeddings import init_embeddings

# RAG 知识库
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'vector_database')
DB_PATH = os.path.abspath(DB_PATH)
emb = init_embeddings("ollama:qwen3-embedding:4b")
db = Chroma(embedding_function=emb, persist_directory=DB_PATH)

@tool
def search_rag(query: str) -> str:
    """检索知识库，对查询进行检索增强"""
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

model = init_chat_model(model="ollama:qwen3.5:9b", temperature=0.7)

agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[get_weather, get_datetime, get_ip_info, get_stock, search_rag]
)

#===================4.0 逻辑实现=================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# 首次进入显示欢迎卡片
if len(st.session_state.conversation_history) == 0:
    st.markdown(welcome_card(), unsafe_allow_html=True)

# 渲染历史消息
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# 处理新消息
if prompt:
    # 用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    # AI 回复（带上下文）
    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation_history]

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
                    message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
