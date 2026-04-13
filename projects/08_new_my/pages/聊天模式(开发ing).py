#=====================1.0 设置基础界面===============================
import streamlit as st
import re
from urllib.parse import quote

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

SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
}

COMMON_STOCK_MAPPING = {
    "贵州茅台": "sh600519",
    "平安银行": "sz000001",
    "招商银行": "sh600036",
    "阿里巴巴": "hk09988",
    "阿里巴巴美股": "gb_baba",
    "腾讯": "hk00700",
    "腾讯控股": "hk00700",
    "苹果": "gb_aapl",
    "特斯拉": "gb_tsla",
    "英伟达": "gb_nvda",
    "微软": "gb_msft",
}


def _safe_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _format_number(value, unit=""):
    if value is None:
        return "--"

    abs_value = abs(value)
    if abs_value >= 100000000:
        return f"{value / 100000000:.2f}亿{unit}"
    if abs_value >= 10000:
        return f"{value / 10000:.2f}万{unit}"
    return f"{value:.2f}{unit}" if isinstance(value, float) else f"{value}{unit}"


def _search_stock_code(keyword: str):
    """通过新浪搜索接口，将股票名称解析为行情代码。"""
    search_url = (
        "https://suggest3.sinajs.cn/suggest/"
        f"type=11,12,13,14,15&key={quote(keyword)}&name=suggestdata"
    )
    response = requests.get(search_url, headers=SINA_HEADERS, timeout=5)
    response.encoding = "gbk"

    match = re.search(r'="([^"]*)"', response.text)
    if not match or not match.group(1):
        return None

    for item in match.group(1).split(";"):
        parts = item.split(",")
        if len(parts) < 4:
            continue

        code = parts[3].strip().lower()
        if re.fullmatch(r"(sh|sz)\d{6}", code):
            return code
        if re.fullmatch(r"hk\d{5}", code):
            return code
        if re.fullmatch(r"gb_[a-z.]+", code):
            return code
    return None


def _normalize_stock_code(query: str):
    text = query.strip()
    lowered = text.lower()

    if not text:
        return None

    if text in COMMON_STOCK_MAPPING:
        return COMMON_STOCK_MAPPING[text]

    if re.fullmatch(r"(sh|sz)\d{6}", lowered):
        return lowered
    if re.fullmatch(r"hk\d{5}", lowered):
        return lowered
    if re.fullmatch(r"gb_[a-z.]+", lowered):
        return lowered

    if re.fullmatch(r"\d{6}", text):
        return f"sh{text}" if text.startswith(("5", "6", "9")) else f"sz{text}"
    if re.fullmatch(r"\d{5}", text):
        return f"hk{text}"
    if re.fullmatch(r"[A-Za-z.]{1,10}", text):
        return f"gb_{lowered}"

    return _search_stock_code(text)


def _fetch_stock_fields(code: str):
    quote_url = f"https://hq.sinajs.cn/list={code}"
    response = requests.get(quote_url, headers=SINA_HEADERS, timeout=5)
    response.encoding = "gbk"
    response.raise_for_status()

    match = re.search(r'="([^"]*)"', response.text.strip())
    if not match or not match.group(1):
        return None

    fields = [item.strip() for item in match.group(1).split(",")]
    if not any(fields):
        return None
    return fields


def _parse_a_stock(code: str, fields):
    if len(fields) < 32:
        return None

    pre_close = _safe_float(fields[2])
    price = _safe_float(fields[3])
    change = None if price is None or pre_close is None else price - pre_close
    change_pct = None if change is None or not pre_close else change / pre_close * 100

    return {
        "market": "A股",
        "name": fields[0],
        "code": code,
        "price": price,
        "open": _safe_float(fields[1]),
        "pre_close": pre_close,
        "high": _safe_float(fields[4]),
        "low": _safe_float(fields[5]),
        "change": change,
        "change_pct": change_pct,
        "volume": _safe_float(fields[8]),
        "turnover": _safe_float(fields[9]),
        "currency": "CNY",
        "updated_at": f"{fields[30]} {fields[31]}".strip(),
    }


def _parse_hk_stock(code: str, fields):
    if len(fields) < 18:
        return None

    updated_at = fields[17]
    if len(fields) > 18 and fields[18]:
        updated_at = f"{fields[17]} {fields[18]}"

    return {
        "market": "港股",
        "name": fields[1] or fields[0],
        "code": code,
        "price": _safe_float(fields[6]),
        "open": _safe_float(fields[2]),
        "pre_close": _safe_float(fields[3]),
        "high": _safe_float(fields[4]),
        "low": _safe_float(fields[5]),
        "change": _safe_float(fields[7]),
        "change_pct": _safe_float(fields[8]),
        "volume": _safe_float(fields[12]) if len(fields) > 12 else None,
        "turnover": _safe_float(fields[11]) if len(fields) > 11 else None,
        "currency": "HKD",
        "updated_at": updated_at.strip(),
    }


def _parse_us_stock(code: str, fields):
    if len(fields) < 11:
        return None

    pre_close = _safe_float(fields[26]) if len(fields) > 26 else None
    price = _safe_float(fields[1])
    change = _safe_float(fields[4]) if len(fields) > 4 else None
    if change is None and price is not None and pre_close is not None:
        change = price - pre_close

    change_pct = _safe_float(fields[2]) if len(fields) > 2 else None
    if change_pct is None and change is not None and pre_close:
        change_pct = change / pre_close * 100

    return {
        "market": "美股",
        "name": fields[0],
        "code": code,
        "price": price,
        "open": _safe_float(fields[5]) if len(fields) > 5 else None,
        "pre_close": pre_close,
        "high": _safe_float(fields[6]) if len(fields) > 6 else None,
        "low": _safe_float(fields[7]) if len(fields) > 7 else None,
        "change": change,
        "change_pct": change_pct,
        "volume": _safe_float(fields[10]) if len(fields) > 10 else None,
        "turnover": None,
        "market_cap": _safe_float(fields[12]) if len(fields) > 12 else None,
        "currency": "USD",
        "updated_at": fields[3] if len(fields) > 3 else "",
    }


def _parse_stock_quote(code: str, fields):
    if code.startswith(("sh", "sz")):
        return _parse_a_stock(code, fields)
    if code.startswith("hk"):
        return _parse_hk_stock(code, fields)
    if code.startswith("gb_"):
        return _parse_us_stock(code, fields)
    return None


def _render_stock_quote(stock):
    if not stock:
        return "数据解析失败"

    price_text = "--" if stock["price"] is None else f'{stock["price"]:.2f} {stock["currency"]}'
    open_text = "--" if stock["open"] is None else f'{stock["open"]:.2f}'
    high_text = "--" if stock["high"] is None else f'{stock["high"]:.2f}'
    low_text = "--" if stock["low"] is None else f'{stock["low"]:.2f}'

    change_text = "--"
    if stock["change"] is not None:
        sign = "+" if stock["change"] > 0 else ""
        change_text = f"{sign}{stock['change']:.2f}"

    pct_text = "--"
    if stock["change_pct"] is not None:
        sign = "+" if stock["change_pct"] > 0 else ""
        pct_text = f"{sign}{stock['change_pct']:.2f}%"

    lines = [
        f"📈 {stock['name']}（{stock['market']} / {stock['code']}）",
        f"当前价：{price_text}",
        f"开盘：{open_text} | 最高：{high_text} | 最低：{low_text}",
        f"涨跌：{change_text} ({pct_text})",
    ]

    if stock["volume"] is not None:
        lines.append(f"成交量：{_format_number(stock['volume'], '股')}")
    if stock["turnover"] is not None:
        lines.append(f"成交额：{_format_number(stock['turnover'])} {stock['currency']}")
    if stock.get("market_cap") is not None:
        lines.append(f"总市值：{_format_number(stock['market_cap'])} {stock['currency']}")
    if stock["updated_at"]:
        lines.append(f"更新时间：{stock['updated_at']}")

    return "\n".join(lines)

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
    """
    查询股票实时行情（新浪免费接口，无需 Token）。
    支持 A 股、港股、美股。
    输入示例：贵州茅台、600519、000001、00700、hk00700、AAPL
    """
    query = stock_name.strip()
    if not query:
        return "请输入股票名称或代码。"

    try:
        code = _normalize_stock_code(query)
        if not code:
            return (
                f"未识别股票：{stock_name}。\n"
                "可以尝试输入更完整的名称，或直接输入代码，例如：600519、hk00700、AAPL。"
            )

        fields = _fetch_stock_fields(code)
        if not fields:
            return f"未查询到 {stock_name} 的行情数据。"

        stock = _parse_stock_quote(code, fields)
        if not stock:
            return f"{stock_name} 的数据格式暂不支持解析。"

        return _render_stock_quote(stock)
    except requests.exceptions.RequestException as e:
        return f"查询失败：网络请求错误：{e}"
    except Exception as e:
        return f"查询失败：{e}"

#=====================3.0 模型与代理 初始化======================
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


#常规 静态模型
model = init_chat_model(model="ollama:qwen3-vl:4b ",tempreature=0.7)

agent = create_agent(
    model=model,
    tools=[get_weather,get_datetime,get_ip_info,get_stock]
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
