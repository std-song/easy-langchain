# 配置文件
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文嵌入模型（首次自动下载约130MB）
OLLAMA_MODEL = "qwen2:1.5b"                  # 本地LLM
VECTOR_DB_PATH = "./vector_db"               # 向量库保存路径
DOCS_PATH = "./docs"                         # 知识文档目录
CHUNK_SIZE = 500                             # 文本分块大小
CHUNK_OVERLAP = 50                           # 分块重叠