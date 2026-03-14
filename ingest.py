"""
运行此脚本将 docs/ 目录中的文档向量化并存储
"""
import os
from config import *
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents():
    """加载 docs 目录下所有支持的文档"""
    loaders = {
        "*.txt": TextLoader,
        "*.pdf": PyPDFLoader,
        "*.docx": Docx2txtLoader,
    }
    docs = []
    for pattern, loader_cls in loaders.items():
        loader = DirectoryLoader(
            DOCS_PATH,
            glob=pattern,
            loader_cls=loader_cls,
            loader_kwargs={"encoding": "utf-8"} if loader_cls == TextLoader else {},
            silent_errors=True
        )
        docs.extend(loader.load())
    print(f"✅ 共加载 {len(docs)} 个文档")
    return docs


def build_vector_db():
    """构建并保存向量数据库"""
    print("📚 正在加载文档...")
    documents = load_documents()

    if not documents:
        print("❌ docs/ 目录为空，请先放入文档！")
        return

    print("✂️  正在分割文本...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ 共切分为 {len(chunks)} 个文本块")

    print("🔢 正在生成向量（首次需下载模型，请耐心等待）...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("💾 正在保存向量数据库...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)
    print(f"✅ 向量数据库已保存至 {VECTOR_DB_PATH}/")


if __name__ == "__main__":
    build_vector_db()