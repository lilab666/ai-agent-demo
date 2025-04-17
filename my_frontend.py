# my_frontend.py
import streamlit as st
import requests
import json
import time
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("frontend_debug.log"), logging.StreamHandler()]
)

# API配置
BACKEND_URL = "http://localhost:8000"

# 初始化界面
st.set_page_config(
    page_title="智能知识库助手",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()


init_session_state()


def show_sidebar():
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "上传文档（支持PDF/TXT/DOCX）",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="单个文件大小不超过200MB",
            key="file_uploader"
        )

        if uploaded_files:
            current_files = {f.name for f in uploaded_files}
            if current_files != st.session_state.uploaded_files:
                st.session_state.uploaded_files = current_files

                if st.session_state.get("uploading"):
                    return

                try:
                    st.session_state.uploading = True
                    with st.spinner("📤 正在上传文档..."):
                        files = [("files", (file.name, file)) for file in uploaded_files]
                        response = requests.post(
                            f"{BACKEND_URL}/upload",
                            files=files,
                            timeout=30
                        )
                    logging.debug(f"上传响应: {response.status_code} {response.text}")

                    if response.status_code == 200:
                        data = response.json()
                        if data["status"] == "success":
                            st.success(f"✅ 知识库更新成功！知识片段数：{data['chunk_count']}")
                            st.session_state.kb_ready = True
                            st.rerun()
                    else:
                        st.error(f"❌ 上传失败：{response.text}")

                except requests.exceptions.Timeout:
                    st.error("⏳ 上传超时，请检查网络连接")
                except Exception as e:
                    st.error(f"⚠️ 上传异常：{str(e)}")
                finally:
                    st.session_state.uploading = False

        st.divider()
        try:
            with st.spinner("🔄 获取知识库状态..."):
                status = requests.get(f"{BACKEND_URL}/status", timeout=5).json()
                logging.debug(f"知识库状态: {status}")

            status_icon = "✅" if status["ready"] else "❌"
            status_color = "green" if status["ready"] else "red"

            st.markdown(
                f"<h3 style='color:{status_color};'>知识库状态 {status_icon}</h3>",
                unsafe_allow_html=True
            )
            st.caption(f"🔖 版本号：{status['version']}")
            st.caption(f"📑 知识片段数：{status['chunk_count']}")

        except requests.exceptions.RequestException:
            st.error("⚠️ 无法连接知识库服务")
        except Exception as e:
            st.error(f"❌ 状态获取失败：{str(e)}")

        # 展示上传文件列表
        st.divider()
        st.markdown("### 📄 已上传文件")
        try:
            response = requests.get(f"{BACKEND_URL}/files", timeout=5)
            if response.status_code == 200:
                file_list = response.json()
                if file_list:
                    for file in file_list:
                        col1, col2, col3, col4 = st.columns([4, 3, 2, 1])
                        with col1:
                            st.write(file.get("file_name", "未知文件名"))
                        with col2:
                            st.write(file.get("upload_time", ""))
                        with col3:
                            st.write(file.get("file_size", ""))
                        with col4:
                            if st.button("🗑️ 删除", key=file["file_name"] + file["upload_time"]):
                                try:
                                    delete_response = requests.delete(
                                        f"{BACKEND_URL}/files",
                                        json={
                                            "file_name": file["file_name"],
                                            "upload_time": file["upload_time"]
                                        },
                                        timeout=10
                                    )
                                    if delete_response.status_code == 200:
                                        st.success(f"✅ 删除成功：{file['file_name']}")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ 删除失败：{delete_response.text}")
                                except Exception as e:
                                    st.error(f"⚠️ 删除异常：{str(e)}")
                else:
                    st.info("📭 当前知识库中还没有上传的文件")
            else:
                st.error("⚠️ 无法获取文件列表")
        except Exception as e:
            st.error(f"❌ 获取文件失败：{str(e)}")


def chat_interface():
    """主聊天界面"""
    st.title("💬 智能知识库助手")

    # 模式切换
    mode = st.radio(
        "模式选择",
        ["💬 普通聊天", "📚 知识库问答"],
        horizontal=True,
        index=1 if st.session_state.get("kb_ready", False) else 0,
        key="mode_selector"
    )

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"📖 参考来源：{', '.join(msg['sources'])}")

    # 用户输入处理
    if prompt := st.chat_input("请输入您的问题："):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 准备响应
        with st.chat_message("assistant"):
            full_response = ""
            sources = []
            error_occurred = False
            placeholder = st.empty()

            # 调试信息面板
            debug_expander = None
            if st.session_state.debug_mode:
                debug_expander = st.expander("🐛 原始响应数据", expanded=False)
                debug_expander.write(
                    f"请求体：```json\n{json.dumps({'prompt': prompt, 'mode': 'knowledge' if mode == '📚 知识库问答' else 'general', 'k': 3}, indent=2)}\n```")

            try:
                with requests.post(
                        f"{BACKEND_URL}/chat",
                        json={
                            "prompt": prompt,
                            "mode": "knowledge" if mode == "📚 知识库问答" else "general",
                            "k": 3
                        },
                        stream=True,
                        timeout=(5, 120)  # 连接5秒，读取120秒超时
                ) as response:
                    # 记录原始响应头
                    logging.debug(f"响应头: {response.headers}")

                    if debug_expander:
                        debug_expander.code(f"HTTP状态码: {response.status_code}")
                        debug_expander.code(f"响应头:\n{json.dumps(dict(response.headers), indent=2)}")

                    response.raise_for_status()

                    # 流式处理数据
                    for line in response.iter_lines():
                        if line:
                            try:
                                # 增强解码容错
                                line_str = line.decode('utf-8', errors='replace').strip()
                                logging.debug(f"原始响应行: {line_str}")  # 控制台日志

                                if debug_expander:
                                    debug_expander.code(f"← {line_str}")  # 网页调试

                                if not line_str:
                                    continue

                                data = json.loads(line_str)

                                # 验证响应格式
                                if "type" not in data:
                                    raise ValueError(f"无效响应格式: {line_str}")

                                # 处理不同数据类型
                                if data["type"] == "content":
                                    content = data.get("data", "")
                                    full_response += content
                                    placeholder.markdown(full_response + "▌")
                                elif data["type"] == "sources":
                                    sources = data.get("data", [])
                                elif data["type"] == "error":
                                    raise Exception(f"后端错误: {data.get('data', '未知错误')}")

                            except json.JSONDecodeError as e:
                                error_msg = f"JSON解析失败: {e}\n原始数据: {line_str}"
                                logging.error(error_msg)
                                if debug_expander:
                                    debug_expander.error(error_msg)
                                error_occurred = True
                            except KeyError as e:
                                error_msg = f"缺少必要字段: {e}\n数据: {data}"
                                logging.error(error_msg)
                                if debug_expander:
                                    debug_expander.error(error_msg)
                                error_occurred = True

                            time.sleep(0.02)  # 控制流式显示速度

            except requests.exceptions.Timeout:
                error_msg = "⏳ 请求超时，请稍后重试"
                st.error(error_msg)
                logging.error(error_msg)
                error_occurred = True
            except requests.exceptions.RequestException as e:
                error_msg = f"⚠️ 网络错误：{str(e)}"
                st.error(error_msg)
                logging.error(error_msg)
                error_occurred = True
            except Exception as e:
                error_msg = f"❌ 处理失败：{str(e)}"
                st.error(error_msg)
                logging.error(error_msg)
                error_occurred = True

            # 最终显示处理
            if not error_occurred:
                placeholder.markdown(full_response)
                if sources:
                    st.caption(f"📖 参考来源：{', '.join(sources)}")

                # 更新消息历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
            else:
                # 移除未完成的用户消息
                if st.session_state.messages and st.session_state.messages[-1]["content"] == prompt:
                    st.session_state.messages.pop()


# 启动界面
show_sidebar()
chat_interface()
