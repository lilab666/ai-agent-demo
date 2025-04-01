import sqlite3

# 连接到 SQLite 数据库
db_path = 'resources/metadata.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询 documents 表的所有内容
cursor.execute("SELECT * FROM documents")

# 获取并打印所有记录
rows = cursor.fetchall()
for row in rows:
    print(row)

# 关闭连接
conn.close()
