# mem0-demo：智能客服系统

> 基于 **mem0ai + 通义千问(Qwen) + ChromaDB + RAG** 的智能客服系统，采用多 Agent 架构

---

## 架构总览

```
用户输入
   │
   ▼
┌─────────────────────────────────────┐
│         Router Agent                 │  ← 统一入口，意图识别
│  · 关键词快速匹配（0ms）            │
│  · LLM 兜底路由                    │
│  · 支持串行组合（一句话多意图）     │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
KnowledgeAgent    MerchantAgent        ← 子 Agent
(知识库 Agent)    (商户数据 Agent)
    │                 │
    ▼                 ▼
RAG 知识库       业务接口调用
+ 用户记忆       (订单/物流/用户/商品)
```

**核心组件：**
| 组件 | 用途 |
|------|------|
| **Router Agent** | 意图识别，路由到子 Agent，支持串行组合多 Agent |
| **KnowledgeAgent** | 知识库问答（RAG + mem0ai 记忆） |
| **MerchantAgent** | 商户数据查询（订单/物流/用户/商品，6 个工具函数） |
| `mem0ai` | 跨会话用户记忆管理 |
| `ChromaDB` | 向量数据库（知识库 + 记忆双重存储） |
| `Qwen (通义千问)` | 大语言模型 + Embedding 生成 |
| `FastAPI` | RESTful API 后端 |

---

## 项目结构

```
mem0-demo/
├── main.py                  # FastAPI 入口
├── router_agent.py          # 路由 Agent（意图识别 + 多 Agent 分发）
├── knowledge_agent.py       # 知识库 Agent（RAG + 记忆增强）
├── merchant_agent.py        # 商户数据 Agent（业务接口调用 + Mock）
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
# 普通对话（多 Agent 串行组合）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查订单顺便问退货政策", "user_id": "user_001"}'

# 返回示例（路由到两个 Agent）
{
  "answer": "您的订单已于今天上午派送\n\n退货政策：7天内可申请...",
  "rag_sources": [...],
  "memories_used": [...],
  "model": "qwen-plus",
  "routed_to": ["merchant", "knowledge"],   // 路由目标列表
  "tools_called": [{"tool": "get_order_info", "result": ...}]
}

# 流式对话 (SSE，多 Agent 串行输出)
curl "http://localhost:8000/chat/stream?message=查订单&user_id=user_001"
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
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | **必填** |
| `QWEN_MODEL` | 对话模型 | `qwen-plus` |
| `QWEN_EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-v3` |
| `CHROMA_PERSIST_PATH` | ChromaDB 存储路径 | `./data/chroma_db` |
| `APP_PORT` | 服务端口 | `8000` |
| `ROUTER_MODEL` | Router Agent 使用的模型（可换更轻量的） | `qwen-plus` |
| `MERCHANT_API_BASE_URL` | 商户业务接口地址 | `http://localhost:8080/api` |
| `MERCHANT_API_KEY` | API Key | 必填（沙箱环境） |
| `MERCHANT_API_SECRET` | API Secret | 必填（沙箱环境） |
| `MERCHANT_API_PASSPHRASE` | API Passphrase | 必填（沙箱环境） |
| `MERCHANT_API_TOKEN` | 商户接口 Bearer Token（兼容旧字段） | 可选 |
| `MERCHANT_API_TIMEOUT` | 商户接口请求超时（秒） | `10` |

> 未配置 `MERCHANT_API_BASE_URL` 时，MerchantAgent 自动返回 Mock 数据，方便本地开发。

---

## 多 Agent 工作流程

### Router Agent 路由策略

```
用户输入 → _route()
    │
    ├─ [1] 关键词快速匹配（无 LLM 调用，0ms）
    │     user_input 同时检测两类关键词，不短路
    │     → 同时命中：直接返回 ["merchant", "knowledge"]
    │     → 命中一个：直接返回 [target]
    │
    ├─ [2] LLM 路由兜底（轻量调用 qwen-plus）
    │     支持返回单目标或多目标（逗号分隔）
    │     → "merchant" / "knowledge" / "merchant,knowledge"
    │
    └─ [3] 默认 → ["knowledge"]
```

### 串行组合（多意图）

一句话同时涉及两类问题时，两个 Agent 按顺序串行调用，结果自动合并：

```
用户：「查一下我的订单，顺便问退货政策」
         ↓
  Router 识别 → ["merchant", "knowledge"]
         ↓
  [1] MerchantAgent.chat()  →  "您的订单已发货"
  [2] KnowledgeAgent.chat() →  "退货政策：7天无理由..."
         ↓
  _merge_answers() 去重 + 拼接
         ↓
  "您的订单已发货\n\n退货政策：7天无理由..."
```

### MerchantAgent 工具函数

| 工具 | 用途 | Mock 数据 |
|------|------|-----------|
| `get_order_info(user_id)` | 查询用户订单 | 自动生成 2 个示例订单 |
| `get_logistics(order_id)` | 查询物流状态 | 模拟快递轨迹 |
| `get_user_info(user_id)` | 查询用户账户 | 等级/积分/余额 |
| `get_product_info(product_id)` | 查询商品信息 | 价格/库存/规格 |
| `get_refund_status(order_id)` | 查询退款进度 | 退款状态/金额 |
| `search_knowledge_fallback(query)` | 知识库兜底 | 关键词匹配 FAQ |

> 配置真实业务接口后，工具函数会自动调用 `MERCHANT_API_BASE_URL` 下的对应接口。
