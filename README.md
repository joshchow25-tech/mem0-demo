# mem0-demo：智能客服系统

> 基于 **mem0ai + 通义千问(Qwen) + ChromaDB + RAG** 的智能客服系统

---

## 技术架构

```
用户输入
   ↓
┌─────────────────────────────────────────┐
│           CustomerServiceAgent          │
│                                         │
│  1. 检索用户记忆 (mem0ai + ChromaDB)    │
│  2. 检索知识库 (RAG + ChromaDB)         │
│  3. 构建增强 Prompt                     │
│  4. 调用 Qwen 生成回答                  │
│  5. 更新用户记忆 (mem0ai)               │
└─────────────────────────────────────────┘
   ↓
返回回答 + 引用来源 + 使用记忆
```

**核心组件：**
| 组件 | 用途 |
|------|------|
| `mem0ai` | 跨会话用户记忆管理（自动提取、存储、检索） |
| `ChromaDB` | 向量数据库（知识库 + 记忆双重存储） |
| `Qwen (通义千问)` | 大语言模型 + Embedding 生成 |
| `FastAPI` | RESTful API 后端 |
| `RAG` | 知识库检索增强生成 |

---

## 项目结构

```
mem0-demo/
├── main.py                  # FastAPI 入口
├── agent.py                 # 智能客服 Agent 核心
├── rag_knowledge_base.py    # RAG 知识库（ChromaDB + Qwen Embedding）
├── memory_manager.py        # mem0ai 记忆管理
├── config.py                # 全局配置
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量示例
├── static/
│   └── index.html           # 前端聊天界面
└── data/
    ├── chroma_db/           # ChromaDB 持久化目录（自动创建）
    └── knowledge/           # 可放置知识库文件
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DashScope API Key：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
QWEN_MODEL=qwen-plus
QWEN_EMBEDDING_MODEL=text-embedding-v3
```

> 获取 API Key：https://dashscope.aliyun.com/

### 3. 启动服务

```bash
python main.py
```

服务启动后访问：
- **前端界面**：http://localhost:8000
- **API 文档**：http://localhost:8000/docs

---

## API 接口

### 💬 对话

```bash
# 普通对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "如何退款？", "user_id": "user_001"}'

# 流式对话 (SSE)
curl "http://localhost:8000/chat/stream?message=退款政策&user_id=user_001"
```

### 📚 知识库管理

```bash
# 添加文本
curl -X POST http://localhost:8000/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{"text": "退款政策：7天无理由退款...", "source": "退款政策.txt"}'

# 上传文件 (.txt/.pdf/.docx)
curl -X POST http://localhost:8000/knowledge/file \
  -F "file=@your_document.pdf"

# 查看知识库
curl http://localhost:8000/knowledge/list

# 删除来源
curl -X DELETE http://localhost:8000/knowledge/退款政策.txt
```

### 🧠 记忆管理

```bash
# 查看用户记忆
curl http://localhost:8000/memory/user_001

# 清除用户记忆
curl -X DELETE http://localhost:8000/memory/user_001
```

---

## 工作原理

### mem0ai 记忆机制

每次对话后，`mem0ai` 会自动从对话中提取关键信息并存储：
- 用户的问题偏好
- 用户提到的个人信息
- 历史问题和解决情况

下次对话时，自动检索相关记忆，提供个性化回答。

### RAG 检索流程

1. 用户问题 → Qwen Embedding 生成向量
2. ChromaDB cosine 相似度检索 Top-K 文档片段
3. 将检索结果注入 Prompt，增强回答准确性

---

## 自定义知识库

在 `data/knowledge/` 放置文档文件，或通过 API 上传。支持格式：
- `.txt` / `.md` — 纯文本
- `.pdf` — PDF 文档
- `.docx` — Word 文档

---

## 环境变量说明

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | 必填 |
| `QWEN_MODEL` | 对话模型 | `qwen-plus` |
| `QWEN_EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-v3` |
| `CHROMA_PERSIST_PATH` | ChromaDB 存储路径 | `./data/chroma_db` |
| `APP_PORT` | 服务端口 | `8000` |
