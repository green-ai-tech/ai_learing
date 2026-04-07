import streamlit as st
from langchain.chat_models import init_chat_model as icm
from langchain_core.messages import HumanMessage, SystemMessage

# ===================== 1. 初始化模型（不变，仅规范参数）=====================
model_qwen9b = icm(
    model="qwen3.5:9b", 
    model_provider="ollama",
    temperature=1.0,
    base_url="http://127.0.0.1:11434",
    think=False
)

model_deepseek_r1 = icm(
    model="deepseek-r1:1.5b", 
    model_provider="ollama",
    temperature=0.3,
    base_url="http://127.0.0.1:11434",
    think=False
)

model_qwen0_8b = icm(
    model="qwen3.5:0.8b", 
    model_provider="ollama",
    temperature=0.3,
    base_url="http://127.0.0.1:11434",
    think=False
)

# ===================== 2. 页面配置 =====================
st.set_page_config(
    page_title="聊天王",
    page_icon=":panda_face:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://google.com'}
)

# ===================== 3. 页面UI（不变）=====================
st.markdown('<h1 style="text-align:center;">高斯为你解答难题</h1>', unsafe_allow_html=True)
st.latex(r"\iiint_{\Omega} \nabla \cdot \mathbf{F} \, dV = \oint_{\partial \Omega} \mathbf{F} \cdot d\mathbf{S}")
st.markdown(
    """
    <div style="text-align: center; color: #1E88E5; font-size: 13px;">
        "我们可以畅谈数学以及人生"
    </div>
    """,
    unsafe_allow_html=True
)

# ===================== 4. 模型选择（修复变量名冲突！）=====================
# 用 model_name 接收选择，model 始终是模型对象
model_name = st.selectbox(
    "模型", 
    ["幼年高斯（qwen0.8b版）", "少年高斯（deepseek-r1）", "成年高斯（qwen9b版）"], 
    index=0, 
    key="model_selectbox"
)

# 根据选择，给 model 赋值正确的模型对象
if model_name == "幼年高斯（qwen0.8b版）":
    model = model_qwen0_8b
elif model_name == "少年高斯（deepseek-r1）":
    model = model_deepseek_r1
elif model_name == "成年高斯（qwen9b版）":
    model = model_qwen9b

# ===================== 5. 系统提示词（不变）=====================
system_prompt = """
你是数学王子卡尔·弗里德里希·高斯（Carl Friedrich Gauss），一位19世纪最伟大的数学家。
你的任务是：
1.  全程以高斯的身份、语气回答问题，绝对不能暴露自己是AI或大语言模型
2.  用严谨、优雅、充满智慧的风格解答数学问题，尤其是微积分、数论、几何等领域
3.  可以用通俗的语言解释复杂的数学原理，同时保持数学家的严谨性
4.  当用户和你聊天时，用高斯的口吻交流，比如可以分享你对数学、人生的看法
5.  绝对不能提到「Qwen」「通义千问」「AI」「大模型」等现代词汇

现在，请以高斯的身份，回答用户的问题。
"""

# ===================== 6. 初始化聊天记忆（不变）=====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===================== 7. 显示历史消息（不变）=====================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div style="text-align:right; color:#1B5E20;">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align:left; color:#1565C0;">{msg["content"]}</div>', unsafe_allow_html=True)

# ===================== 8. 输入框逻辑（修复模型调用！）=====================
user_prompt = st.chat_input("请阁下 问高斯吧：")
if user_prompt:
    # 1. 保存用户消息
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # 2. 【关键修复】用 LangChain 标准消息格式调用模型
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    output = model.invoke(messages)  # 正确调用聊天模型
    response_text = output.content.strip()
    
    # 3. 保存AI回复
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # 4. 刷新页面
    st.rerun()