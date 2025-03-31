import os

from dotenv import load_dotenv
load_dotenv()

# 调试输出
print(os.getenv("QWEN_API_KEY"))
print(os.getenv("DEEPSEEK_API_KEY"))
