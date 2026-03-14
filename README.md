这是一个适合 AMD Ryzen 7 / Radeon 780M 本地运行的知识库问答项目，全程用 CPU 推理，无需 GPU，简单易部署。

📁 项目结构预览
local_qa/
├── app.py              # 主程序（Gradio界面）
├── ingest.py           # 文档入库脚本
├── config.py           # 配置文件
├── requirements.txt    # 依赖
└── docs/               # 放你的知识文档（pdf\word\txt）
    └── sample.txt

第一步：安装 PyCharm 和 Python 环境
确保已安装：
  Python 3.10+（推荐 3.11）
  PyCharm Community Edition（免费）

第二步：创建项目并安装依赖
在 PyCharm Terminal 中执行
  # 创建虚拟环境
  python -m venv venv
  venv\Scripts\activate   # Windows
  
  # 安装依赖
  pip install -r requirements.txt

第三步：安装 Ollama 并下载模型
Ollama 是最简单的本地 LLM 运行工具：
  # 1. 下载安装 Ollama（Windows）
  # 访问：https://ollama.com/download
  
  # 2. 安装后，在命令行拉取轻量中文模型（约 2GB）
  ollama pull qwen2:1.5b

第四步：创建所有代码文件
  config.py
  ingest.py（文档入库）
  app.py（主程序）

第五步：放入知识文档
  在 `docs/` 目录放入任意 `.txt` / `.pdf` / `.docx` 文件，例如 `sample.txt`：

第六步：运行项目
  # 终端1：启动 Ollama
  ollama serve
  
  # 终端2（PyCharm Terminal）：先建库
  python ingest.py
  
  # 然后启动应用
  python app.py

  # 然后浏览器访问
  http://127.0.0.1:7860 
  
  ## 整体架构图
  ```
  你的问题
     ↓
  [嵌入模型 bge-small-zh]  →  问题向量
     ↓
  [FAISS 向量库]  →  检索最相关的3段文本
     ↓
  [Prompt 拼装] = 问题 + 检索到的文本
     ↓
  [Ollama qwen2:1.5b]  →  生成回答
     ↓
  Gradio 界面显示
