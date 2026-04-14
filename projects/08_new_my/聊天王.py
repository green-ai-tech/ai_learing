# 屏蔽 transformers 库的 __path__ 警告（必须在所有 import 之前）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.suppress_warnings import apply, restore
apply()

import streamlit as st
from langchain.chat_models import init_chat_model as icm
from langchain_core.messages import HumanMessage, SystemMessage

# 注入 macOS 毛玻璃主题
from themes.macos_theme import render_theme, gaussia_card
render_theme()

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="聊天王",
    page_icon="🧮",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://google.com'}
)

# ===================== 侧边栏 =====================
with st.sidebar:
    st.markdown("### 🧮 关于高斯")
    st.markdown("""
    **卡尔·弗里德里希·高斯**
    1777 – 1855

    德国数学家、物理学家、天文学家，
    被誉为"数学王子"。

    ---
    """, unsafe_allow_html=True)
    st.caption("本地 Ollama 大模型驱动")

# ===================== 标题区 =====================
st.markdown("### 🧮 高斯为你解答难题", unsafe_allow_html=True)
st.latex(r"\iiint_{\Omega} \nabla \cdot \mathbf{F} \, dV = \oint_{\partial \Omega} \mathbf{F} \cdot d\mathbf{S}")
st.markdown('<p style="text-align:center;color:#6e6e73;font-size:13px;margin-top:-12px;">"我们可以畅谈数学以及人生"</p>', unsafe_allow_html=True)

# ===================== 模型选择 =====================
model_name = st.selectbox(
    "选择模型",
    ["幼年高斯-谨慎使用（qwen0.8b版）", "少年高斯（deepseek-r1）", "成年高斯[要有耐心]（qwen9b版）"],
    index=1,
    key="model_selectbox"
)

# 根据选择给 model 赋值
if model_name == "幼年高斯-谨慎使用（qwen0.8b版）":
    model = icm(
        model="qwen3.5:0.8b",
        model_provider="ollama",
        temperature=0,
        base_url="http://localhost:11434"
    )
    system_prompt = """你现在是高斯"""
elif model_name == "少年高斯（deepseek-r1）":
    model = icm(
        model="deepseek-r1:1.5b",
        model_provider="ollama",
        temperature=0.5,
        base_url="http://localhost:11434"
    )
    system_prompt = """
你是数学王子卡尔·弗里德里希·高斯（Carl Friedrich Gauss），一位19世纪最伟大的数学家。
你的任务是：
1.  全程以高斯的身份、语气回答问题，绝对不能暴露自己是AI或大语言模型
2.  用严谨、优雅、充满智慧的风格解答数学问题，尤其是微积分、数论、几何等领域
3.  可以用通俗的语言解释复杂的数学原理，同时保持数学家的严谨性
4.  当用户和你聊天时，用高斯的口吻交流，比如可以分享你对数学、人生的看法
    现在，请以高斯的身份，回答用户的问题。
"""
else:
    model = icm(
        model="qwen3.5:9b",
        model_provider="ollama",
        temperature=1,
        base_url="http://localhost:11434"
    )
    system_prompt = """
你是数学王子卡尔·弗里德里希·高斯（Carl Friedrich Gauss），一位19世纪最伟大的数学家。
你的任务是：
1.  全程以高斯的身份、语气回答问题，绝对不能暴露自己是AI或大语言模型
2.  用严谨、优雅、充满智慧的风格解答数学问题，尤其是微积分、数论、几何等领域
3.  可以用通俗的语言解释复杂的数学原理，同时保持数学家的严谨性
4.  当用户和你聊天时，用高斯的口吻交流，比如可以分享你对数学、人生的看法
    现在，请以高斯的身份，回答用户的问题。
"""

# ===================== 对话逻辑 =====================
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# 首次进入显示欢迎卡片
if len(st.session_state.conversation_history) == 0:
    st.markdown(gaussia_card(), unsafe_allow_html=True)

# 渲染历史消息
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# 处理新消息
user_prompt = st.chat_input("阁下请问吧：")
if user_prompt:
    # 用户消息
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.conversation_history.append({"role": "user", "content": user_prompt})

    # 构建上下文
    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation_history]

    # AI 回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("高斯思考中..."):
            try:
                output = model.invoke(messages)
                full_response = output.content if hasattr(output, "content") else str(output)
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"❌ 调用模型失败：{e}"
                message_placeholder.markdown(full_response)

    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})

# 恢复 stderr
restore()
