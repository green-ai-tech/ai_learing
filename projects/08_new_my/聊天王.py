# 屏蔽 transformers 库的 __path__ 警告（必须在所有 import 之前）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils.suppress_warnings import apply, restore
apply()

import streamlit as st
from langchain.chat_models import init_chat_model as icm
from langchain_core.messages import HumanMessage, SystemMessage

# 关键 import 完成后恢复 stderr
restore()

# ===================== 2. 页面配置 =====================
st.set_page_config(
    page_title="聊天王",
    page_icon=":panda_face:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://google.com'}
)

# ===================== 3. 页面UI=====================
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

# ===================== 4. 模型选择=====================
model_name = st.selectbox(
    "模型", 
    ["幼年高斯-谨慎使用（qwen0.8b版）", "少年高斯（deepseek-r1）", "成年高斯[要有耐心]（qwen9b版）"], 
    index=0, 
    key="model_selectbox"
)

cout =0;

# 根据选择，给 model 赋值正确的模型对象
if model_name == "幼年高斯-谨慎使用（qwen0.8b版）":
    cout =0;
    model = icm(
        model="qwen3.5:0.8b ",
        model_provider="ollama",
        temperature=0,          
        base_url="http://localhost:11434"
    )
elif model_name == "少年高斯（deepseek-r1）":
    cout =1;
    model = icm(
        model="deepseek-r1:1.5b",
        model_provider="ollama",
        temperature=0.5,
        base_url="http://localhost:11434"
    )
else:
    cout =2;
    model = icm(
        model="qwen3.5:9b",
        model_provider="ollama",
        temperature=1,
        base_url="http://localhost:11434"
    )

# ===================== 5. 系统提示词=====================
system_prompt_brf="""你现在是高斯"""

system_prompt = """
你是数学王子卡尔·弗里德里希·高斯（Carl Friedrich Gauss），一位19世纪最伟大的数学家。
你的任务是：
1.  全程以高斯的身份、语气回答问题，绝对不能暴露自己是AI或大语言模型
2.  用严谨、优雅、充满智慧的风格解答数学问题，尤其是微积分、数论、几何等领域
3.  可以用通俗的语言解释复杂的数学原理，同时保持数学家的严谨性
4.  当用户和你聊天时，用高斯的口吻交流，比如可以分享你对数学、人生的看法
    现在，请以高斯的身份，回答用户的问题。
"""



user_prompt = st.chat_input("阁下请问吧：")    #输入组件
if user_prompt:
    #调用模型，完成文档， 输出
    st.markdown(f"<p style='text-align: right;'>{user_prompt}</p>", unsafe_allow_html=True)

    if cout==0:
        all_prompts = system_prompt_brf + "\n\n" + user_prompt
    else:     
        all_prompts = system_prompt + "\n\n" + user_prompt
    
   
    #调用模型
    output = model.invoke(all_prompts)

    #显示输出
    st.markdown(f"<p style='text-align: left; color: blue;'>{output.content}</p>", unsafe_allow_html=True)
