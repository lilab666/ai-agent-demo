import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

# 初始化消息上下文
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]


def trim_messages(messages, max_length=12):
    # 如果超过最大长度，就删除最早的一对 user 和 assistant 消息（下标 1 和 2）
    while len(messages) > max_length:
        if len(messages) >= 3:
            # 删除最早的 user 和 assistant（保留 system）
            del messages[1:3]
        else:
            break


while True:
    user_input = input("\n你：")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("结束对话。")
        break

    # 加入 user 消息
    messages.append({"role": "user", "content": user_input})

    trim_messages(messages)
    # 发起流式请求
    response = client.chat.completions.create(
        model="qwen-turbo-2025-04-28",
        messages=messages,
        stream=True,
        # Qwen3 开源模型时使用：
        # extra_body={"enable_thinking": False},
    )

    full_reply = ""
    print("助手：", end="", flush=True)
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            full_reply += delta
            print(delta, end="", flush=True)

    # 加入 assistant 回复到上下文
    messages.append({"role": "assistant", "content": full_reply})
