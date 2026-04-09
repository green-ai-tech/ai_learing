import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
import datetime
import requests



#--------------------------工具实现----------------------------------
@tool
def get_datetime()->str:
    """
    获取当前的日期和时间
    Returns:
        str: 当前的日期和时间，格式为 "YYYY-MM-DD HH:MM:SS"
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_weather(location:str)->str:
    """
    获取指定位置的天气信息
    Args:
        location (str): 位置名称，例如 "北京"、"上海" 等
    Returns:
        str: 指定位置的天气信息，格式为 "Location: Weather Info"
    """
    api_key = "7630410ef5ff49bd82eaa9cafea22638"        # 替换为你的和风天气 API key
    weather_url = "https://devapi.qweather.com/v7/weather/now"
    city_url = "https://geoapi.qweather.com/v2/city/lookup"
    city_params = {
        "location": location,          
        "key": api_key
    }
    try:
        city_resp = requests.get(city_url, params=city_params)
        city_resp.raise_for_status()
        city_data = city_resp.json()

        if city_data.get("code") != "200" or not city_data.get("location"):
            return f"查询失败，未找到城市：{location}"
        
        # 获取ID
        city_id = city_data["location"][0]["id"]
        city_name = city_data["location"][0]["name"]


        weather_params = {
            "location": city_id,
            "key": api_key
        }

        weathr_resp = requests.get(weather_url, params=weather_params)
        weathr_resp.raise_for_status()
        weather_data = weathr_resp.json()

        if weather_data.get("code") == "200":
            now = weather_data["now"]
            weather_info = f"{city_name}天气：{now['text']}，温度：{now['temp']}°C，湿度：{now['humidity']}%"
            return weather_info
        else:
            return f"查询天气失败，错误信息：{weather_data.get('code')}"
        
    except requests.RequestException as e:
        return f"网络请求失败：{str(e)}"

@tool
def get_news(category:str)->str:
    """
    获取指定类别的新闻信息
    Args:
        category (str): 新闻类别，例如 "科技"、"体育" 等
    Returns:
        str: 指定类别的新闻信息，格式为 "Category: News Info"
    """
    # 这里可以调用一个新闻API来获取新闻信息，以下是一个示例实现
    # 由于没有实际的API，这里返回一个模拟的新闻信息
    return f"{category}新闻：这是一些关于{category}的最新新闻。"

#--------------------------模型与代理的初始化--------------------------
model = init_chat_model(
    model="ollama:qwen3.5:9b",
    temperature=0.5
)


agent = create_agent(
    model=model,
    tools=[get_datetime, get_weather, get_news]
)

#--------------------------Streamlit界面------------------------------
#1. 页面配置
st.set_page_config(
    page_title="智能体应用",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

#2. 侧边栏
with st.sidebar:
    st.write("这是侧边栏，可以放一些说明或者工具")

#3. 内容区域的处理

# 状态
if "messages" not in st.session_state:      #判定message变量是否已经创建
    st.session_state.messages = []          #在回话对象中创建一个列表，用来保存我们的对话记录

st.title(body="智能体应用示例",width="stretch",text_alignment="center")
st.caption(body="使用的是免费本地部署的大模型",width="stretch",text_alignment="center")


#显示历史聊天信息
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"],text_alignment="right")
    else:
        with st.chat_message("ai"):
            st.markdown(f"<font color='blue'>{message['content']}</font>", unsafe_allow_html=True)




prompt = st.chat_input("请输入你的问题：")
if prompt:

    #显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt,text_alignment="right")

    #存放用户消息
    st.session_state.messages.append({"role":"user","content":prompt})

    #输出AI答复
    with st.chat_message("ai"):
        with st.spinner("AI正在思考..."):

            #调用模型(只调用一次)
            response = agent.invoke({"messages":[{"role":"user","content":prompt}]})

            st.markdown(f"<font color='blue'>{response['messages'][-1].content}</font>", unsafe_allow_html=True)

    st.session_state.messages.append({"role":"ai","content":response['messages'][-1].content})



