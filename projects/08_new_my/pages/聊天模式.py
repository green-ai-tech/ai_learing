# 关闭transformers警告
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from langchain.tools import tool  # 这个必须在最前面！
import re
import pyautogui
import os
import subprocess
import requests


# ===================== 天气工具 =====================
def get_coordinates(city_name: str):
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
    参数: location (str): 城市名，例如 北京、上海
    返回: 天气详情字符串
    """
    lat, lon, city_name = get_coordinates(location)
    if lat is None:
        return f"未找到城市: {location}，请检查城市名称是否正确。"
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
        humidity = None
        if hourly.get('relativehumidity_2m'):
            humidity = hourly['relativehumidity_2m'][0]
        weather_desc = "晴朗" if current.get('weathercode') == 0 else "多云/有雨"
        result = f"主人>.< {city_name} 当前天气：\n🌤 天气：{weather_desc}\n🌡 温度：{temp}°C\n🌬 风速：{wind_speed} km/h"
        if humidity:
            result += f"\n💧 湿度：{humidity}%"
        return result + "\n小喵温馨提醒主人注意穿衣哦～"
    except requests.exceptions.RequestException as e:
        return f"网络请求错误：{e}"
    except Exception as e:
        return f"处理数据时出错：{e}"

# ===================== 系统控制工具 =====================
@tool
def control_system(oper: str) -> str:
    """
    系统控制工具：打开CMD、记事本、控制鼠标移动
    """
    pass

# ===================== 计算器 =====================
@tool
def calculator(op: str, a: float, b: float) -> str:
    """加法计算器"""
    try:
        return f"{a} 加 {b} 等于 {a + b}"
    except:
        return "计算出错啦>"

# ===================== 股票查询工具 =====================
@tool
def get_stock(stock_name: str) -> str:
    """
    查询股票实时数据（新浪免费接口，无需Token）
    输入：股票名称 或 代码，如：贵州茅台、000001、600519
    """
    import requests
    import re

    def code_from_name(name):
        # 简单映射（常用股）
        mapping = {
            "贵州茅台": "sh600519", "平安银行": "sz000001",
            "招商银行": "sh600036", "阿里巴巴": "hk09988",
            "腾讯": "hk00700", "苹果": "aapl"
        }
        if name in mapping:
            return mapping[name]
        # 6位数字自动加前缀
        if re.match(r"^\d{6}$", name):
            return f"sh{name}" if name.startswith(("6","5","9")) else f"sz{name}"
        return None

    code = code_from_name(stock_name.strip())
    if not code:
        return f"未找到股票：{stock_name}"

    url = f"http://hq.sinajs.cn/list={code}"
    headers = {"Referer": "https://finance.sina.com.cn/"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        data = res.text.strip()
        # 解析格式：var hq_str_sh600519="名称,开盘,昨收,现价,最高,最低,..."
        match = re.search(r'="([^"]+)"', data)
        if not match:
            return "数据解析失败"
        arr = match.group(1).split(",")
        if len(arr) < 32:
            return "数据不完整"

        name = arr[0]
        open_p = float(arr[1])
        pre_close = float(arr[2])
        price = float(arr[3])
        high = float(arr[4])
        low = float(arr[5])
        change = price - pre_close
        change_pct = (change / pre_close) * 100

        return (
            f"📈 {name}（{code}）\n"
            f"当前价：{price:.2f} 元\n"
            f"开盘：{open_p:.2f} | 最高：{high:.2f} | 最低：{low:.2f}\n"
            f"涨跌：{change:.2f} 元 ({change_pct:.2f}%)\n"
        )
    except Exception as e:
        return f"查询失败：{str(e)}"

# ===================== 意图识别 =====================
def run_tool(query: str) -> str:
    q = query.strip()
    if any(k in q for k in ["cmd", "记事本", "鼠标"]):
        return control_system.invoke({"oper": q})
    elif any(k in q for k in ["加", "+"]) and any(c.isdigit() for c in q):
        nums = re.findall(r"\d+\.?\d*", q)
        if len(nums) >= 2:
            return calculator.invoke({"op": "+", "a": float(nums[0]), "b": float(nums[1])})
        return "小喵没找到两个数字哦>"
    elif "天气" in q:
        city = re.sub(r"天气|\s", "", q).strip() or "北京"
        return get_weather.invoke({"location": city})
    elif "股票" in q:
        stock_key = re.sub(r"股票|\s", "", q).strip()
        if not stock_key:
            return "请告诉小喵你想查哪只股票~"
        return get_stock.invoke({"stock_name": stock_key})
    elif "你是谁" in q:
        return "主人你好>.<，我是快读小喵酱，有什么问题都可以问我哦"
    else:
        return "我可以帮你打开CMD、记事本、控制鼠标、算加法、查天气、查股票~"

# ===================== 页面 =====================
st.set_page_config(page_title="小喵初代", page_icon="🐾", layout="wide")
st.title("小喵初代 🐾")
st.caption("本地免费小喵模型")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("小喵在等你先说>.<")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("ai"):
        with st.spinner("小喵思考中..."):
            ans = run_tool(prompt)
            st.markdown(ans)
    st.session_state.messages.append({"role": "ai", "content": ans})