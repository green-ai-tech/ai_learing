# macOS 毛玻璃暗色主题 - 全局 CSS
# 所有 Streamlit 页面通用

THEME_CSS = """
<style>
:root {
    --bg-primary: #1a1a1a;
    --bg-secondary: #2c2c2e;
    --bg-glass: rgba(44, 44, 46, 0.72);
    --border-glass: rgba(255, 255, 255, 0.08);
    --accent-blue: #007AFF;
    --accent-green: #34c759;
    --text-primary: #f5f5f7;
    --text-secondary: #a1a1a6;
    --text-muted: #6e6e73;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
}

.main { background: var(--bg-primary) !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }

/* 侧边栏毛玻璃 */
section[data-testid="stSidebar"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border-right: 1px solid var(--border-glass) !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* 输入框 */
[data-testid="stChatInput"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stBaseButton-secondary"] {
    background: var(--accent-blue) !important;
    border-radius: 50% !important;
}

/* 聊天消息 */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-lg) !important;
    margin-bottom: 12px !important;
}
[data-testid="stAvatar"] {
    border-radius: 50% !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent-blue) !important; }

/* 表格 */
table {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-glass) !important;
}
th { background: rgba(255,255,255,0.05) !important; color: var(--text-primary) !important; }
td { color: var(--text-secondary) !important; border-color: var(--border-glass) !important; }

/* 代码块 */
code, pre {
    background: rgba(0,0,0,0.3) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-glass) !important;
}

/* 滚动条 */
::-webkit-scrollbar { width: 6px !important; }
::-webkit-scrollbar-track { background: transparent !important; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15) !important; border-radius: 3px !important; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25) !important; }

/* 用户消息 - 系统蓝 */
.user-msg { color: var(--accent-green); }
.ai-msg { color: var(--accent-blue); }
</style>
"""


def render_theme():
    """在 Streamlit 页面中注入主题样式"""
    import streamlit as st
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ============ 组件 HTML ============

def welcome_card(title="🤖 智能助手", desc="我是你的本地 AI 智能体", tools=None):
    """生成毛玻璃欢迎卡片"""
    if tools is None:
        tools = [
            ("🌤️", "天气查询"),
            ("📅", "时间查询"),
            ("📈", "股票行情"),
            ("🌐", "IP 信息"),
            ("📚", "知识库检索 (RAG)"),
        ]
    items = ""
    span = "1" if len(tools) % 2 != 0 else "2"
    for icon, name in tools:
        if name == tools[-1][1] and len(tools) % 2 != 0:
            items += f'''<div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:10px 12px;border:1px solid rgba(255,255,255,0.05);grid-column:span 2;">
                <span style="font-size:16px;">{icon}</span>
                <span style="color:#f5f5f7;font-size:13px;margin-left:6px;">{name}</span>
            </div>'''
        else:
            items += f'''<div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:10px 12px;border:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:16px;">{icon}</span>
                <span style="color:#f5f5f7;font-size:13px;margin-left:6px;">{name}</span>
            </div>'''
    return f'''
    <div style="background:rgba(44,44,46,0.6);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
        border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;margin:16px 0 24px 0;
        box-shadow:0 4px 16px rgba(0,0,0,0.3);">
        <h3 style="color:#f5f5f7;margin:0 0 12px 0;font-size:20px;font-weight:600;">{title}</h3>
        <p style="color:#a1a1a6;margin:0 0 16px 0;font-size:14px;line-height:1.5;">{desc}</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">{items}</div>
    </div>'''


def gaussia_card():
    """高斯主题欢迎卡片"""
    return f'''
    <div style="background:rgba(44,44,46,0.6);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
        border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;margin:16px 0 24px 0;
        box-shadow:0 4px 16px rgba(0,0,0,0.3);text-align:center;">
        <h3 style="color:#f5f5f7;margin:0 0 8px 0;font-size:22px;font-weight:600;">🧮 数学王子 · 高斯</h3>
        <p style="color:#a1a1a6;margin:0;font-size:14px;line-height:1.6;">
            "我们可以畅谈数学以及人生"<br>
            <span style="color:#6e6e73;font-size:12px;">选择不同模型，体验高斯的智慧</span>
        </p>
    </div>'''
