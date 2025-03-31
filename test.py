# 请先安装必要库：pip install streamlit openai
import streamlit as st
from openai import OpenAI

# 设置页面标题和图标
st.set_page_config(page_title="DeepSeek Chat", page_icon="🤖")

# 初始化OpenAI客户端
client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],  # 推荐使用Streamlit Secrets管理密钥
    base_url=st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/beta")
)

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个小助手"}]

# 处理模型切换状态
if "model" not in st.session_state:
    st.session_state.model = "deepseek-chat"

# 使用selectbox选择模式
model_option = st.selectbox(
    "选择模式：",
    ["deepseek-v3", "deepseek-r1"],
    index=0,  # 初始值为“普通聊天”
    help="选择是否启用深度思考模型"
)

# 根据选择的选项切换模型
if model_option == "deepseek-r1":
    st.session_state.model = "deepseek-reasoner"
else:
    st.session_state.model = "deepseek-chat"

# 展示历史对话
for message in st.session_state.messages[1:]:  # 跳过系统消息
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入处理
if prompt := st.chat_input("请输入询问:"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 展示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 展示机器人回复
    with st.chat_message("assistant"):
        # 调用API
        response = client.chat.completions.create(
            model=st.session_state.model,  # 动态选择模型
            messages=st.session_state.messages,
            max_tokens=1024,
            stream=False
        )

        # 获取回复内容
        reply = response.choices[0].message.content

        # 展示Markdown格式回复
        st.markdown(reply)

        # 添加助手消息到历史
        st.session_state.messages.append({"role": "assistant", "content": reply})