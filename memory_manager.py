"""
mem0ai 记忆管理模块
- 使用 mem0ai 为每个用户维护长期记忆
- 基于 ChromaDB 存储向量记忆
- 使用 Qwen 作为 LLM 提取记忆要点
- 对话存档使用独立的 ChromaDB collection（绕过 LLM 提取，直接存储原文）
"""
import logging
import time
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from mem0 import Memory

from config import (
    DASHSCOPE_API_KEY,
    QWEN_MODEL,
    QWEN_BASE_URL,
    QWEN_EMBEDDING_MODEL,
    CHROMA_PERSIST_PATH,
    CHROMA_MEMORY_COLLECTION,
    MEM0_TOP_K,
)

# 对话存档专用 collection 名称
CONV_COLLECTION = "conv_archive"

logger = logging.getLogger(__name__)


def build_mem0_config() -> dict:
    """构建 mem0 配置，使用 Qwen + ChromaDB"""
    return {
        # LLM：通义千问（OpenAI 兼容模式）
        "llm": {
            "provider": "openai",
            "config": {
                "model": QWEN_MODEL,
                "openai_base_url": QWEN_BASE_URL,
                "api_key": DASHSCOPE_API_KEY,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        # Embedder：通义千问 text-embedding
        "embedder": {
            "provider": "openai",
            "config": {
                "model": QWEN_EMBEDDING_MODEL,
                "openai_base_url": QWEN_BASE_URL,
                "api_key": DASHSCOPE_API_KEY,
            },
        },
        # 向量存储：ChromaDB 本地持久化
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": CHROMA_MEMORY_COLLECTION,
                "path": CHROMA_PERSIST_PATH,
            },
        },
        # 版本
        "version": "v1.1",
    }


class MemoryManager:
    """
    用户记忆管理器
    - 自动从对话中提取关键记忆
    - 按 user_id 隔离记忆
    - 支持记忆检索、更新、删除
    """

    def __init__(self):
        config = build_mem0_config()
        self.memory = Memory.from_config(config)
        # 对话存档用独立的 ChromaDB client（绕过 LLM 提取，直接存原文）
        self._conv_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self._conv_col = self._conv_client.get_or_create_collection(
                name=CONV_COLLECTION,
                metadata={"description": "对话存档"},
            )
        except Exception:
            self._conv_col = self._conv_client.get_collection(name=CONV_COLLECTION)
        logger.info("mem0 记忆管理器已初始化（Qwen + ChromaDB）")

    def add_conversation(
        self,
        messages: list[dict],
        user_id: str,
        metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        从对话中提取并存储记忆
        :param messages: [{"role": "user"/"assistant", "content": "..."}]
        :param user_id: 用户唯一标识
        :param metadata: 附加元数据（可包含 category 字段）
        :return: 提取到的记忆列表
        """
        md = metadata.copy() if metadata else {}
        # 统一 metadata 格式，确保 category 字段存在
        if "category" not in md:
            md["category"] = "通用"
        result = self.memory.add(
            messages=messages,
            user_id=user_id,
            metadata=md,
        )
        memories = result.get("results", [])
        logger.info(f"[记忆] 用户 {user_id} 新增/更新记忆 {len(memories)} 条，分类：{md['category']}")
        return memories

    def add_memory_direct(
        self,
        content: str,
        user_id: str,
        category: str = "通用",
    ) -> list[dict]:
        """
        直接写入一条记忆（供工具调用使用）
        注意：category="对话记录" 时会写入独立的存档 collection，不走 LLM 提取
        """
        if category == "对话记录":
            return self.add_conversation_archive(user_id, content)
        result = self.memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=user_id,
            metadata={"category": category, "source": "tool_call"},
        )
        memories = result.get("results", [])
        logger.info(f"[记忆·工具] 用户 {user_id} 写入记忆：{content[:50]}... 分类：{category}")
        return memories

    def add_conversation_archive(
        self,
        user_id: str,
        conversation_text: str,
    ) -> list[dict]:
        """
        直接将对话存档写入 ChromaDB（绕过 LLM 提取）
        conversation_text 格式：已拼接好的 "【时间】\n用户：...\n助手：..." 字符串
        """
        record_id = str(uuid.uuid4())
        ts = int(time.time())
        # 截取前 500 字做 embedding（超长内容截断）
        embed_text = conversation_text[:500]
        metadata = {
            "user_id": user_id,
            "category": "对话记录",
            "source": "conversation_archive",
            "created_at": ts,
            "timestamp": conversation_text.split("】")[0].replace("【", "") if "】" in conversation_text else "",
        }
        try:
            self._conv_col.add(
                ids=[record_id],
                documents=[conversation_text],
                embeddings=[self._embed(embed_text)],
                metadatas=[metadata],
            )
            logger.info(f"[对话存档] 用户 {user_id} 已保存 (id={record_id})")
            return [{"id": record_id, "memory": conversation_text, "metadata": metadata}]
        except Exception as e:
            logger.error(f"[对话存档] 保存失败：{e}")
            return []

    def _embed(self, text: str) -> list[float]:
        """调用 embedding 模型获取向量"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)
            resp = client.embeddings.create(model=QWEN_EMBEDDING_MODEL, input=text)
            return resp.data[0].embedding
        except Exception as e:
            logger.warning(f"[Embedding] 失败：{e}，返回零向量")
            return [0.0] * 1536

    def search_memory(self, query: str, user_id: str, top_k: int = MEM0_TOP_K) -> list[dict]:
        """
        根据查询检索用户记忆（同时搜 mem0 用户画像和对话存档）
        :return: [{"id": ..., "memory": ..., "score": ...}, ...]
        """
        results = self.memory.search(query=query, user_id=user_id, limit=top_k)
        memories = results.get("results", [])
        # 追加对话存档的向量检索
        try:
            q_embed = self._embed(query[:500])
            conv_results = self._conv_col.query(
                query_embeddings=[q_embed],
                where={"user_id": user_id},
                n_results=min(top_k, 5),
            )
            for doc, mid, meta, dist in zip(
                conv_results.get("documents", [[]])[0],
                conv_results.get("ids", [[]])[0],
                conv_results.get("metadatas", [[]])[0],
                conv_results.get("distances", [[]])[0],
            ):
                memories.append({
                    "id": mid,
                    "memory": doc,
                    "score": 1.0 - (dist or 0),
                    "metadata": meta,
                })
        except Exception as e:
            logger.warning(f"[对话存档检索] 失败：{e}")
        logger.info(f"[记忆] 为用户 {user_id} 检索到 {len(memories)} 条相关记忆")
        return memories

    def get_all_memory(self, user_id: str) -> list[dict]:
        """获取用户所有记忆（mem0 用户画像 + 对话存档）"""
        results = self.memory.get_all(user_id=user_id)
        memories = results.get("results", [])
        # 追加对话存档
        try:
            conv_results = self._conv_col.get(where={"user_id": user_id})
            for doc, mid, meta in zip(
                conv_results.get("documents", []),
                conv_results.get("ids", []),
                conv_results.get("metadatas", []),
            ):
                # 构造与 mem0 格式一致的记忆对象
                memories.append({
                    "id": mid,
                    "memory": doc,
                    "metadata": meta,
                    "created_at": meta.get("created_at"),
                    "updated_at": None,
                })
        except Exception as e:
            logger.warning(f"[对话存档] 读取失败：{e}")
        return memories

    def format_memory_context(self, query: str, user_id: str) -> str:
        """检索记忆并格式化为 Prompt 上下文"""
        memories = self.search_memory(query, user_id)
        if not memories:
            return ""
        lines = []
        for m in memories:
            lines.append(f"- {m.get('memory', '')}")
        return "\n".join(lines)

    def delete_user_memory(self, user_id: str) -> bool:
        """清除某用户所有记忆（含对话存档）"""
        try:
            self.memory.delete_all(user_id=user_id)
            # 删除对话存档
            try:
                conv_results = self._conv_col.get(where={"user_id": user_id})
                if conv_results and conv_results.get("ids"):
                    self._conv_col.delete(ids=conv_results["ids"])
                    logger.info(f"[对话存档] 已清除用户 {user_id} 的 {len(conv_results['ids'])} 条存档")
            except Exception as e:
                logger.warning(f"[对话存档] 清除失败：{e}")
            logger.info(f"[记忆] 已清除用户 {user_id} 的所有记忆")
            return True
        except Exception as e:
            logger.error(f"[记忆] 清除失败：{e}")
            return False

    def get_memory_by_id(self, memory_id: str) -> Optional[dict]:
        """
        按 ID 获取单条记忆详情
        :return: 记忆字典，含 id/memory/created_at/updated_at/metadata
        """
        try:
            result = self.memory.get(memory_id=memory_id)
            return result
        except Exception as e:
            logger.error(f"[记忆] 获取单条记忆失败：{e}")
            return None

    def get_memory_history(self, memory_id: str) -> list[dict]:
        """
        获取单条记忆的变更历史
        :return: [{"id", "memory_id", "old_memory", "new_memory", "event", "created_at", "updated_at"}, ...]
        """
        try:
            return self.memory.history(memory_id=memory_id) or []
        except Exception as e:
            logger.error(f"[记忆] 获取记忆历史失败：{e}")
            return []

    def update_memory(self, memory_id: str, data: str) -> bool:
        """
        按 ID 更新记忆内容
        :param memory_id: 记忆 ID
        :param data: 新的记忆内容
        """
        try:
            self.memory.update(memory_id=memory_id, data=data)
            return True
        except Exception as e:
            logger.error(f"[记忆] 更新记忆失败：{e}")
            return False

    def delete_memory_by_id(self, memory_id: str) -> bool:
        """按 ID 删除单条记忆（同时支持 mem0 和对话存档）"""
        try:
            self.memory.delete(memory_id=memory_id)
            return True
        except Exception:
            pass
        # 可能是对话存档 ID，尝试从 conv_archive 删除
        try:
            self._conv_col.delete(ids=[memory_id])
            logger.info(f"[对话存档] 已删除记忆 {memory_id}")
            return True
        except Exception as e:
            logger.error(f"[记忆] 删除单条记忆失败：{e}")
            return False
