from langchain.schema import Document

docs = [Document(page_content="hello :     world\nfuck\tfuck", metadata={})]

for doc in docs:
    doc.page_content = doc.page_content.replace(" ", "")

print(docs[0].page_content)  # 输出：helloworld ✅
