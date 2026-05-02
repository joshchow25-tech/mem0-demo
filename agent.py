"""
智能客服 Agent 核心逻辑
整合：mem0 记忆 + ChromaDB RAG + Qwen 大模型

处理流程：
1. 检索用户记忆（了解用户背景偏好）
2. 检索 RAG 知识库（获取相关文档）
3. 构建增强 Prompt
4. 调用 Qwen 生成回答
5. 异步更新用户记忆
"""
import json
import logging
from typing import Optional, AsyncIterator
from openai import OpenAI, AsyncOpenAI

from config import (
    DASHSCOPE_API_KEY,
    QWEN_MODEL,
    QWEN_BASE_URL,
    RAG_TOP_K,
    MEM0_TOP_K,
)
from rag_knowledge_base import KnowledgeBase
from memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# ====================== System Prompt ======================
SYSTEM_PROMPT = """你是一位专业、友善的智能客服助手。

你的能力：
1. 根据知识库内容准确回答用户问题
2. 记住用户的偏好和历史问题，提供个性化服务
3. 对于知识库没有覆盖的问题，诚实告知并提供通用建议

回答原则：
- 简洁清晰，优先使用知识库中的内容
- 如有用户历史记忆，结合记忆提供个性化回复
- 不确定时主动说明，避免误导用户
- 语气亲切专业，使用中文回答
- **输出格式**：用 Markdown 格式输出回答，合理使用以下元素：
  - 标题（##、###）用于分节
  - **加粗** 用于强调重点
  - 列表（- 或 1.）用于多条信息
  - `代码` 或代码块用于技术内容
  - > 引用 用于提示或注意事项
  - 表格用于对比信息

{memory_section}

**记忆管理（重要）**：
当你从对话中了解到用户的重要信息时，必须调用 `save_memory` 工具保存。
需要保存的内容包括但不限于：
- 用户的偏好（如喜欢的颜色、风格、品牌）
- 订单或物流信息
- 售后问题或投诉记录
- 用户的身份信息（姓名、地址、电话等）
- 用户提到的产品需求
不要保存闲聊内容、无关信息或已经保存在记忆中的内容。
每次对话最多调用 2 次 `save_memory`。
"""

MEMORY_SECTION_TEMPLATE = """
【用户历史记忆】
{memory_content}
"""

RAG_CONTEXT_TEMPLATE = """
【相关知识库内容】
{rag_content}
"""


# ====================== 工具定义 ======================
SAVE_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "save_memory",
        "description": "保存一条重要信息到用户记忆库。"
                     "当用户透露偏好、需求、订单信息、身份信息等值得记住的内容时调用。"
                     "不要在闲聊或无关对话时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "要记住的内容，用简洁的中文描述，不超过50字",
                },
                "category": {
                    "type": "string",
                    "description": "记忆分类",
                    "enum": ["用户偏好", "订单信息", "售后记录", "产品咨询", "对话记录", "通用"],
                },
            },
            "required": ["content"],
        },
    },
}


class CustomerServiceAgent:
    """
    智能客服 Agent
    """

    def __init__(self, knowledge_base: KnowledgeBase, memory_manager: MemoryManager):
        self.kb = knowledge_base
        self.mm = memory_manager

        # 同步客户端（普通调用）
        self.llm = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL,
        )
        # 异步客户端（流式调用）
        self.async_llm = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL,
        )

    def _build_system_prompt(self, user_memory_context: str) -> str:
        """构建带记忆的 System Prompt"""
        if user_memory_context:
            memory_section = MEMORY_SECTION_TEMPLATE.format(memory_content=user_memory_context)
        else:
            memory_section = ""
        return SYSTEM_PROMPT.format(memory_section=memory_section)

    def _build_user_message(self, user_input: str, rag_context: str) -> str:
        """构建带 RAG 的用户消息"""
        if rag_context:
            return f"{RAG_CONTEXT_TEMPLATE.format(rag_content=rag_context)}\n\n用户问题：{user_input}"
        return user_input

    def chat(
        self,
        user_input: str,
        user_id: str,
        conversation_history: Optional[list[dict]] = None,
        update_memory: bool = True,
    ) -> dict:
        """
        单轮对话（同步）
        - 当 update_memory=True 时，LLM 自主调用 save_memory 工具保存重要信息
        """
        conversation_history = conversation_history or []

        # Step 1: 检索用户记忆
        memory_context = self.mm.format_memory_context(user_input, user_id)
        memories_used = self.mm.search_memory(user_input, user_id)

        # Step 2: 检索 RAG 知识库
        rag_results = self.kb.search(user_input, top_k=RAG_TOP_K)
        rag_context = self.kb.format_context(user_input, top_k=RAG_TOP_K)

        # Step 3: 构建 Prompt
        system_prompt = self._build_system_prompt(memory_context)
        user_message = self._build_user_message(user_input, rag_context)

        # Step 4: 构建消息列表（含历史）
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-6:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        # Step 5: 调用 Qwen（带工具）
        tools = [SAVE_MEMORY_TOOL] if update_memory else None
        response = self.llm.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=0.7,
            max_tokens=2000,
        )

        message = response.choices[0].message

        # Step 6: 处理工具调用
        if message.tool_calls and update_memory:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "save_memory":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        self.mm.add_memory_direct(
                            content=args["content"],
                            user_id=user_id,
                            category=args.get("category", "通用"),
                        )
                        logger.info(
                            f"[工具记忆] 已保存：{args['content'][:50]}... 分类：{args.get('category', '通用')}"
                        )
                    except Exception as e:
                        logger.warning(f"[工具记忆失败] {e}")

            # 将工具调用结果返回给 LLM，获取最终回复
            messages.append(message.model_dump())
            for tool_call in message.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "记忆已保存",
                })

            final_response = self.llm.chat.completions.create(
                model=QWEN_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            answer = final_response.choices[0].message.content
        else:
            answer = message.content

        # Step 7: 存档本轮对话（用户问 + 助手答）
        if update_memory:
            import time
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            conv_text = (
                f"【{ts}】\n"
                f"用户：{user_input}\n"
                f"助手：{answer[:300]}"
                + ("..." if len(answer) > 300 else "")
            )
            try:
                self.mm.add_memory_direct(
                    content=conv_text,
                    user_id=user_id,
                    category="对话记录",
                )
                logger.info(f"[对话存档] 已保存用户 {user_id} 的对话")
            except Exception as e:
                logger.warning(f"[对话存档失败] {e}")

        return {
            "answer": answer,
            "rag_sources": [
                {
                    "text": r["text"][:150] + "...",
                    "score": r["score"],
                    "source": r["metadata"].get("source", "知识库"),
                }
                for r in rag_results
            ],
            "memories_used": [m.get("memory", "") for m in memories_used],
            "model": QWEN_MODEL,
        }

    async def stream_chat(
        self,
        user_input: str,
        user_id: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        流式对话（异步，SSE 使用）
        - 流式模式暂不支持工具调用，记忆由 /chat 接口的 LLM 自主保存
        """
        conversation_history = conversation_history or []

        # 检索记忆 & RAG
        memory_context = self.mm.format_memory_context(user_input, user_id)
        rag_context = self.kb.format_context(user_input, top_k=RAG_TOP_K)

        system_prompt = self._build_system_prompt(memory_context)
        user_message = self._build_user_message(user_input, rag_context)

        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-6:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        # 流式调用
        full_answer = ""
        async with self.async_llm.chat.completions.stream(
            model=QWEN_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer += delta
                    yield delta
