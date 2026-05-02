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

# ====================== mem0 ======================
MEM0_TOP_K = 5  # 每次检索记忆条数

# ====================== App ======================
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
