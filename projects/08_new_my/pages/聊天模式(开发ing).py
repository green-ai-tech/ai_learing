#=====================1.0 设置基础界面===============================
import streamlit as st

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
from langchain.tools import tool
from datetime import datetime                       #获取时间的库
import requests


import requests
from langchain.tools import tool

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
#=====================3.0 模型与代理 初始化======================
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent



#尝试使用动态模型
# model = init_chat_model()
# if prompt :
#     response = model.invoke(
#         input=prompt,
#         tools=[get_weather, get_datetime, get_ip_info],
#         config={
#             "configurable": {"model": "ollama:qwen3.5:9b", "temperature": 0.9}
#         }
#     )


#常规 静态模型
model = init_chat_model(model="ollama:qwen3.5:9b",tempreature=0.7)


agent = create_agent(
    model=model,
    tools=[get_weather,get_datetime,get_ip_info]
)
#===================4.0 逻辑实现=================================

    #保留对话
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

    #显示历史对话
for message in st.session_state.conversation_history:
    if message["role"] !="ai":
       st.markdown(f'<div style="color: green; text-align: right">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color: blue; text-align="left">{message["content"]}</div>',unsafe_allow_html=True)



if prompt:
    #显示用户的输入
    with st.chat_message("user"):
        st.markdown(f'<div style="color: green; text-align="right">{prompt}</div>',unsafe_allow_html=True)

    #存放用户消息
    st.session_state.conversation_history.append({"role":"user","content":prompt})


    #显示AI的推理结果
    # with st.chat_message("ai"):
    #     with st.spinner("思考中..."):

    #         response = agent.invoke({"messages":[{"role":"user","content":prompt}]})

    #         st.markdown(f'<div style="color: blue; text-align="right">{response['messages'][-1].content}</div>',unsafe_allow_html=True)
    # #存放AI的消息        
    # st.session_state.conversation_history.append({"role":"ai","content":response['messages'][-1].content})



    with st.chat_message("ai"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("思考中..."):
            # stream_mode="messages" 返回 (message_chunk, metadata)
            for chunk, metadata in agent.stream(
                {"messages": [("user", prompt)]},
                stream_mode="messages"
            ):
                # chunk 是 AIMessageChunk 对象，包含 content 属性
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(
                        full_response + '<span style="color: blue;">▌</span>',unsafe_allow_html=True)
            message_placeholder.markdown(f'<span style="color: blue;">{full_response}</span>',unsafe_allow_html=True)

    st.session_state.conversation_history.append({"role": "ai", "content": full_response})