"""
基于本地知识库的问答应用 - 主程序
启动后访问 http://127.0.0.1:7860
"""
import os
import gradio as gr
from config import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ── 提示词模板 ──────────────────────────────────────────
PROMPT_TEMPLATE = """你是一个专业的问答助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请直接说"知识库中未找到相关内容"，不要编造答案。

参考资料：
{context}

用户问题：{question}

请用中文给出简洁、准确的回答："""


def load_qa_chain():
    """初始化问答链"""
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError("向量数据库不存在，请先运行 ingest.py！")

    # 加载嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 加载向量库
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 初始化本地 LLM（Ollama）
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.1,       # 低温度 = 更确定的回答
        num_predict=512,       # 最大输出token
    )

    # 创建提示词
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # 构建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),  # 检索最相关的3段
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


# ── 初始化 ──────────────────────────────────────────────
print("🚀 正在初始化问答系统...")
try:
    qa_chain = load_qa_chain()
    print("✅ 系统初始化成功！")
    init_error = None
except Exception as e:
    qa_chain = None
    init_error = str(e)
    print(f"❌ 初始化失败：{e}")


# ── 问答函数 ─────────────────────────────────────────────
def answer_question(question, history):
    if not question.strip():
        return history, ""

    if qa_chain is None:
        history = history or []
        history.append([question, f"❌ 系统未初始化：{init_error}"])
        return history, ""

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        sources = result.get("source_documents", [])
        if sources:
            source_info = "\n\n📎 参考来源：\n"
            for i, doc in enumerate(sources, 1):
                source = doc.metadata.get("source", "未知来源")
                preview = doc.page_content[:80].replace("\n", " ")
                source_info += f"{i}. {os.path.basename(source)} - {preview}...\n"
            answer += source_info

    except Exception as e:
        answer = f"❌ 查询出错：{str(e)}\n请确认 Ollama 正在运行（命令：ollama serve）"

    history = history or []
    history.append([question, answer])
    return history, ""


# ── Gradio 界面 ──────────────────────────────────────────
with gr.Blocks(title="📚 本地知识库问答", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 本地知识库问答系统\n基于 LangChain + Ollama + FAISS | 完全本地运行")

    chatbot = gr.Chatbot(height=450, label="对话记录")

    with gr.Row():
        question_box = gr.Textbox(
            placeholder="请输入你的问题，按 Enter 发送...",
            label="",
            scale=5
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    clear_btn = gr.Button("🗑️ 清空对话")

    status_text = "✅ 系统就绪" if qa_chain else f"❌ 未就绪：{init_error}"
    gr.Markdown(f"**状态：** {status_text} | **模型：** `{OLLAMA_MODEL}` | **嵌入：** `bge-small-zh`")

    state = gr.State([])
    submit_btn.click(answer_question, [question_box, state], [chatbot, question_box])
    question_box.submit(answer_question, [question_box, state], [chatbot, question_box])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
## 第五步：放入知识文档
"""
在 `docs/` 目录放入任意 `.txt` / `.pdf` / `.docx` 文件，例如 `sample.txt`：

"""