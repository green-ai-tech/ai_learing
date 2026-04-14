import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
import streamlit as st
from langchain.chat_models import init_chat_model

#初始化 模型

model = init_chat_model(
    model="qwen3.5:0.8b", 
    model_provider="ollama",
    temperature=0.9,
    base_url="http://localhost:11434")

#页面
#内容（标题，副标题，输入，输出）

#调用大模型完成智能回答

st.set_page_config(
    page_title="智能聊天应用",
    page_icon=":panda_face:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://google.com',
    }
)

#内容：标题，副标题，输入，输出
st.markdown("<h1 style='text-align: center;'>智能聊天应用</h1>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: blue;">免费大模型本地部署开发</p>', unsafe_allow_html=True)

#调用大模型完成智能回答

prompt = st.chat_input("阁下请问吧：")    #输入组件
if prompt:
    #调用模型，完成文档， 输出
    st.markdown(f"<p style='text-align: right;'>{prompt}</p>", unsafe_allow_html=True)

    #调用模型
    output = model.invoke(prompt)

    #显示输出
    st.markdown(f"<p style='text-align: left; color: blue;'>{output.content}</p>", unsafe_allow_html=True)
       
else:
    st.write("请输入问题，点击发送按钮")
