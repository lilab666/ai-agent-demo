import os
from openai import OpenAI
from docx import Document  # 新增依赖

client = OpenAI(
    api_key="sk-68fd531c27e74f83ac79a81a3eb67e45",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 读取 Word 文档内容
doc = Document('产品信息资料库.docx')
text_content = [paragraph.text for paragraph in doc.paragraphs]
combined_text = '\n'.join(text_content)  # 将所有段落合并为一个字符串

completion = client.embeddings.create(
    model="text-embedding-v3",
    input=combined_text,  # 直接传递文本内容
    dimensions=1024,
    encoding_format="float"
)

print(completion.model_dump_json())