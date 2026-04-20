import streamlit as st
from utils import setup_logging


setup_logging()         #完成日志的初始化
pg = st.navigation([
    st.Page("pages/home.py", title="主页", icon="🔥"),
    st.Page("pages/login.py", title="登录", icon=":material/favorite:"),
], position="hidden")
pg.run()