"""
Router Agent（路由 Agent）
- 接收用户输入，识别意图，路由到正确的子 Agent
- 子 Agent：
    KnowledgeAgent  → 政策/FAQ/使用说明（RAG 知识库）
    MerchantAgent   → 订单/物流/用户/商品实时数据（业务接口）
- 支持串行组合：一句话涉及多个 Agent 时，依次调用并合并结果
- 对外提供与原 CustomerServiceAgent 相同的 chat() / stream_chat() 接口

路由策略：
  1. 快速规则匹配（关键词），同时命中两个 Agent 时都加入列表
  2. 规则未命中时，用轻量 LLM（qwen-plus）做二次判断
  3. 两者都不确定 → 默认走 KnowledgeAgent
  4. 结果按路由顺序自然合并（merchant → knowledge，顺序可调）
"""
import logging
import re
from typing import Optional, AsyncIterator

from openai import OpenAI

from config import (
    DASHSCOPE_API_KEY,
    QWEN_BASE_URL,
    ROUTER_MODEL,
)
from knowledge_agent import CustomerServiceAgent as KnowledgeAgent
from merchant_agent import MerchantAgent

logger = logging.getLogger(__name__)


# ====================== 路由规则 ======================

# 关键词命中 → 直接路由到 merchant，跳过 LLM 路由
MERCHANT_KEYWORDS = re.compile(
    r"业务|查询|余额|"
    r"订单[号码]?|快递[号码单]?|运单|物流|发货|到货|收货|签收|"
    r"退款|退货|换货|售后申请|维权|投诉|"
    r"库存|现货|缺货|补货|商品[状态信息]?|价格|规格|"
    r"我的[订单账户积分余额]|账户|积分|会员[等级]?|"
    r"ORDER|ORD[-_]?\d|[Ss][Ff]\d{8,}|[Zz][Tt][Oo]\d{8,}|"  # 订单号/快递号模式
    r"查[一下下下]?[订单物流]|帮我[看查]|卡类型|开卡",
    re.IGNORECASE,
)

# 关键词命中 → 直接路由到 knowledge
KNOWLEDGE_KEYWORDS = re.compile(
    r"政策|规定|说明|怎么[操作用申请]|如何[操作用申请]|"
    r"多少[天日]|几[天日]|工作日|时效|"
    r"支持[哪些什么]|可以[用吗]|能[不]?[用吗退]|"
    r"运费|免邮|配送[范围方式]|支付[方式]|付款[方式]|"
    r"会员[权益规则介绍]|积分规则|优惠券[规则使用]",
    re.IGNORECASE,
)

# LLM 路由的 System Prompt
ROUTER_SYSTEM_PROMPT = """你是一个意图分类器。用户发来一条客服消息，请判断应该由哪个 Agent 处理。

Agent 说明：
- knowledge：处理政策咨询、FAQ、使用说明、规则解释等静态内容（不需要实时数据）
- merchant：处理订单查询、物流追踪、用户账户、商品库存等需要实时查询业务系统的请求

规则：
- 如果只需要一个 Agent 处理，回复：knowledge 或 merchant
- 如果需要两个 Agent 处理（用户一句话涉及两类问题），回复：knowledge,merchant 或 merchant,knowledge
- 多个目标用英文逗号分隔，不要加空格
- 不要解释，不要加标点或换行。"""


class OrchestratorAgent:
    """
    路由 Agent，统一入口
    对外接口：chat() / stream_chat()，与原 CustomerServiceAgent 完全一致
    """

    def __init__(
        self,
        knowledge_agent: KnowledgeAgent,
        merchant_agent: MerchantAgent,
    ):
        self.knowledge = knowledge_agent
        self.merchant = merchant_agent

        self.router_llm = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL,
        )

    # ------------------------------------------------------------------
    # 路由逻辑
    # ------------------------------------------------------------------

    def _route(self, user_input: str) -> list[str]:
        """
        决定路由目标列表：['knowledge'] 或 ['merchant'] 或 ['merchant', 'knowledge']
        先跑快速规则（两者都检测），再由 LLM 二次确认/过滤。
        """
        targets: list[str] = []

        # 1. 快速规则（同时检测，不短路）
        has_merchant = bool(MERCHANT_KEYWORDS.search(user_input))
        has_knowledge = bool(KNOWLEDGE_KEYWORDS.search(user_input))

        if has_merchant:
            targets.append("merchant")
        if has_knowledge:
            targets.append("knowledge")

        # 两个规则都命中 → 直接返回，不需要 LLM
        if has_merchant and has_knowledge:
            logger.info(f"[Router] 规则双命中 → merchant+knowledge | input={user_input[:40]}")
            return targets

        # 只有一个命中 → 跳过 LLM，直接返回
        if targets:
            logger.info(f"[Router] 规则命中 → {targets[0]} | input={user_input[:40]}")
            return targets

        # 2. LLM 路由兜底
        try:
            resp = self.router_llm.chat.completions.create(
                model=ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                temperature=0,
                max_tokens=20,
            )
            raw = resp.choices[0].message.content.strip().lower()
            # 支持单目标和多目标，如 "merchant" 或 "knowledge,merchant"
            parts = [p.strip() for p in raw.split(",")]
            valid = [p for p in parts if p in ("merchant", "knowledge")]
            if valid:
                logger.info(f"[Router] LLM 路由 → {valid} | input={user_input[:40]}")
                return valid
        except Exception as e:
            logger.warning(f"[Router] LLM 路由失败，降级到 knowledge：{e}")

        # 3. 默认
        logger.info(f"[Router] 默认 → knowledge | input={user_input[:40]}")
        return ["knowledge"]

    # ------------------------------------------------------------------
    # 对外接口（与 KnowledgeAgent 签名一致）
    # ------------------------------------------------------------------

    def chat(
        self,
        user_input: str,
        user_id: str,
        conversation_history: Optional[list[dict]] = None,
        update_memory: bool = True,
    ) -> dict:
        """
        统一对话入口（同步）
        串行调用多个子 Agent，结果合并后返回
        """
        targets = self._route(user_input)

        # 串行调用（只在最后一个 Agent 更新记忆，避免重复写入）
        answers: list[str] = []
        all_rag_sources: list[dict] = []
        all_memories: list[str] = []
        all_tools: list[dict] = []

        for i, name in enumerate(targets):
            should_update_memory = update_memory and (i == len(targets) - 1)

            if name == "merchant":
                result = self.merchant.chat(
                    user_input=user_input,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    update_memory=should_update_memory,
                )
            else:
                result = self.knowledge.chat(
                    user_input=user_input,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    update_memory=should_update_memory,
                )

            answers.append(result.get("answer", ""))
            all_rag_sources.extend(result.get("rag_sources", []))
            all_memories.extend(result.get("memories_used", []))
            if result.get("tools_called"):
                all_tools.extend(result["tools_called"])

        # 合并答案（去重 + 自然段落拼接）
        final_answer = self._merge_answers(answers)

        return {
            "answer": final_answer,
            "rag_sources": all_rag_sources,
            "memories_used": all_memories,
            "model": "qwen-plus",
            "routed_to": targets,
            "tools_called": all_tools if all_tools else None,
        }

    @staticmethod
    def _merge_answers(answers: list[str]) -> str:
        """
        合并多个 Agent 的回答
        - 单条直接返回
        - 多条去重后用换行拼接，避免重复开场白
        """
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]

        merged: list[str] = []
        seen = set()
        for ans in answers:
            # 取第一句作为去重 key（去掉句号/问号）
            first = ans.strip().split("。")[0].split("？")[0].strip().lower()
            if first and first not in seen:
                seen.add(first)
                merged.append(ans.strip())
            elif not first and ans.strip():
                # 无法提取 key 的直接追加
                merged.append(ans.strip())

        return "\n\n".join(merged)

    async def stream_chat(
        self,
        user_input: str,
        user_id: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        流式对话入口（异步，SSE 使用）
        串行流式：先流完一个 Agent 的结果，再流下一个
        - merchant：同步调用后按 10 字一批模拟流式
        - knowledge：透传原生流式生成器
        """
        targets = self._route(user_input)

        for name in targets:
            # 插入 Agent 分隔标记（第二个及之后的 Agent）
            idx = targets.index(name)
            if idx > 0:
                yield "\n\n"

            if name == "merchant":
                result = self.merchant.chat(
                    user_input=user_input,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    update_memory=False,
                )
                answer = result.get("answer", "")
                # 按 10 字一批 yield，模拟打字效果
                for i in range(0, len(answer), 10):
                    yield answer[i: i + 10]
            else:
                async for chunk in self.knowledge.stream_chat(
                    user_input=user_input,
                    user_id=user_id,
                    conversation_history=conversation_history,
                ):
                    yield chunk
