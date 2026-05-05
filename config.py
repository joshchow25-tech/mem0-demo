"""
全局配置模块
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ====================== Qwen / DashScope ======================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

# DashScope OpenAI 兼容接口
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ====================== ChromaDB ======================
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma_db")
CHROMA_COLLECTION_NAME = "customer_service_kb"  # 知识库 collection
CHROMA_MEMORY_COLLECTION = "mem0_memory"         # mem0 记忆 collection

# ====================== RAG ======================
RAG_TOP_K = 5           # 检索 Top-K 片段
RAG_CHUNK_SIZE = 500    # 文本分块大小
RAG_CHUNK_OVERLAP = 50  # 分块重叠
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))  # 最低相关度阈值

# ====================== mem0 ======================
MEM0_TOP_K = 5           # 每次检索记忆条数
MEM0_SCORE_THRESHOLD = float(os.getenv("MEM0_SCORE_THRESHOLD", "0.3"))

# ====================== mem0 Custom Instructions ======================
# 控制 LLM 如何从对话中提取记忆的指令
# 覆盖 config/init 时设置，影响所有 add() 调用
CUSTOM_INSTRUCTIONS = os.getenv("MEM0_CUSTOM_INSTRUCTIONS", "")

# 预设记忆提取策略（通过 MEM0_MEMORY_STRATEGY 环境变量选择）
# 可选：customer_service | general | strict
MEMORY_STRATEGY = os.getenv("MEM0_MEMORY_STRATEGY", "customer_service")

# ====================== App ======================
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ====================== Merchant Agent ======================
# 商户业务接口基础 URL（可配置为内网地址或 Mock 地址）
MERCHANT_API_BASE_URL = os.getenv("MERCHANT_API_BASE_URL", "http://localhost:8080/api")
# 商户接口鉴权参数
MERCHANT_API_KEY = os.getenv("MERCHANT_API_KEY", "")          # API Key
MERCHANT_API_SECRET = os.getenv("MERCHANT_API_SECRET", "")    # API Secret
MERCHANT_API_PASSPHRASE = os.getenv("MERCHANT_API_PASSPHRASE", "")  # Passphrase
# 兼容旧字段（优先级：独立字段 > TOKEN 字段）
MERCHANT_API_TOKEN = os.getenv("MERCHANT_API_TOKEN", "")
# 请求超时（秒）
MERCHANT_API_TIMEOUT = int(os.getenv("MERCHANT_API_TIMEOUT", "10"))

# ====================== Orchestrator ======================
# Router 使用的模型（可选更轻量的模型降低延迟）
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen-plus")

# ====================== Telegram Bot ======================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
