"""
商户数据查询 Agent
- 有权限调用底层业务接口（订单、用户、库存、物流等）
- 工具以 Function Calling 方式暴露给 LLM
- 对外提供与 KnowledgeAgent 相同的 chat() 接口，方便 Orchestrator 统一调用

业务接口约定（可在 MERCHANT_API_BASE_URL 配置）：
  GET  /orders/{order_id}          - 查单个订单
  GET  /orders?user_id=&status=    - 查用户订单列表
  GET  /users/{user_id}            - 查用户信息
  GET  /products/{product_id}      - 查商品信息
  GET  /inventory/{product_id}     - 查库存
  GET  /logistics/{tracking_no}    - 查物流

所有接口通过 HTTP 调用真实业务系统。
"""
import base64
import hashlib
import hmac
import json
import logging
import httpx
from typing import Optional

from openai import OpenAI

from config import (
    DASHSCOPE_API_KEY,
    QWEN_MODEL,
    QWEN_BASE_URL,
    MERCHANT_API_BASE_URL,
    MERCHANT_API_KEY,
    MERCHANT_API_SECRET,
    MERCHANT_API_PASSPHRASE,
    MERCHANT_API_TOKEN,
    MERCHANT_API_TIMEOUT,
)

logger = logging.getLogger(__name__)


# ====================== 商户 API HTTP 客户端 ======================

class MerchantHttpClient:
    """
    商户业务接口 HTTP 客户端
    签名逻辑（与 paycrypto-sdk-java HttpUtil 保持一致）：
      1. 生成毫秒级时间戳
      2. 构造待签名字符串：timestamp + method + path + queryStr + body
      3. 用 apiSecret 对上述字符串做 HMAC-SHA256，Base64 编码得到签名
      4. Authorization 头格式：Railone:{apiKey}:{timestamp}:{signature}
    """

    AUTHORIZATION_PREFIX = "Railone"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        timeout: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 签名
    # ------------------------------------------------------------------

    @staticmethod
    def _sign(
        timestamp: str,
        method: str,
        path: str,
        query_str: str,
        body: Optional[str],
        api_key: str,
        api_secret: str,
    ) -> str:
        """
        生成签名字符串，与 Java SDK HmacSHA256Base64Util.sign() 一致：
        timestamp + method(大写) + apiKey + requestPath + queryString + body
        """
        # Java SDK 顺序：timestamp + method + appKey + requestPath + queryString + body
        parts = [timestamp, method.upper(), api_key, path]
        if query_str:
            parts.append("?" + query_str)
        if body:
            parts.append(body)
        sign_string = "".join(parts)
        sig = hmac.new(
            api_secret.encode("utf-8"),
            sign_string.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(sig).decode("utf-8")

    def _build_authorization(self, timestamp: str, signature: str) -> str:
        return f"{self.AUTHORIZATION_PREFIX}:{self.api_key}:{timestamp}:{signature}"

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        """
        GET 请求
        - path: API 路径（如 /orders/123）
        - params: 查询参数 dict，会拼接到 URL
        """
        timestamp = str(int(time.time() * 1000))
        query_str = self._build_query_str(params)
        signature = self._sign(timestamp, "GET", path, query_str, None, self.api_key, self.api_secret)

        headers = {
            "Content-Type": "application/json",
            "Authorization": self._build_authorization(timestamp, signature),
            "Access-Passphrase": self.api_passphrase,
        }

        url = f"{self.base_url}{path}"
        logger.info(f"[MerchantHttp] GET {url}?{query_str}")

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def post(self, path: str, params: Optional[dict] = None, json_body: Optional[dict] = None) -> dict:
        """
        POST 请求
        - path: API 路径
        - params: 查询参数 dict
        - json_body: JSON 请求体 dict
        """
        timestamp = str(int(time.time() * 1000))
        query_str = self._build_query_str(params)
        body_str = json.dumps(json_body) if json_body else ""
        signature = self._sign(timestamp, "POST", path, query_str, body_str, self.api_key, self.api_secret)

        headers = {
            "Content-Type": "application/json",
            "Authorization": self._build_authorization(timestamp, signature),
            "Access-Passphrase": self.api_passphrase,
        }

        url = f"{self.base_url}{path}"
        logger.info(f"[MerchantHttp] POST {url} body={body_str[:200]}")

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, params=params, json=json_body, headers=headers)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _build_query_str(params: Optional[dict]) -> str:
        """将 dict 转换为 key=value&key=value 形式的查询字符串"""
        if not params:
            return ""
        return "&".join(f"{k}={v}" for k, v in sorted(params.items()))


# ====================== 工具定义 ======================

MERCHANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单详情，包括商品、状态、金额、收货地址、支付信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，如 ORD20240101001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_user_orders",
            "description": "查询用户的订单列表，可按状态过滤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户 ID",
                    },
                    "status": {
                        "type": "string",
                        "description": "订单状态过滤：pending（待付款）/ paid（已付款）/ shipped（已发货）/ delivered（已收货）/ cancelled（已取消）",
                        "enum": ["pending", "paid", "shipped", "delivered", "cancelled", "all"],
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_user_info",
            "description": "查询用户账号信息，包括会员等级、积分、注册时间等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户 ID",
                    }
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": "根据物流运单号查询物流轨迹和当前状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_no": {
                        "type": "string",
                        "description": "物流运单号",
                    },
                    "carrier": {
                        "type": "string",
                        "description": "快递公司（可选）：SF / ZTO / YTO / YD / JD",
                    },
                },
                "required": ["tracking_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_product",
            "description": "查询商品详情，包括价格、规格、库存状态、上下架状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品 ID 或商品名称（模糊匹配）",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_refund",
            "description": "查询退款/售后申请状态和进度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "关联的订单号",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_card_types",
            "description": "查询支持的卡类型列表，返回卡类型 ID、名称、状态等基本信息。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_balance",
            "description": "查询机构账户余额信息，包括各币种可用余额、冻结金额等。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_customer_accounts",
            "description": "查询用户账户列表或指定用户的账户信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "acct_no": {
                        "type": "string",
                        "description": "用户账号（可选），不传则返回全部账户列表",
                    }
                },
                "required": [],
            },
        },
    },
]


# ====================== System Prompt ======================

MERCHANT_SYSTEM_PROMPT = """你是商户数据查询助手，专门负责查询订单、用户、物流、商品等业务数据。

你的能力：
1. 通过工具查询实时业务数据（订单状态、物流轨迹、用户账户、商品库存等）
2. 根据查询结果给出清晰、准确的回答
3. 如果用户提供的信息不完整，主动询问缺少的关键参数（如订单号、手机号等）

回答原则：
- 只回答与数据查询相关的问题，不做政策解释（那是知识库 Agent 的工作）
- 数据来自业务系统，如实呈现，不做猜测
- 涉及用户隐私数据（手机号、地址）展示时中间字符打码
- 用中文回答，格式清晰

{memory_section}
"""


# ====================== 业务接口客户端 ======================

class MerchantAPIClient:
    """
    商户业务接口客户端
    - 调用 MERCHANT_API_BASE_URL 配置的接口，使用签名鉴权
    - 接口不可达时抛出异常，由上层处理
    """

    def __init__(self):
        # 优先用新的签名鉴权客户端；其次降级到 Bearer Token（兼容旧接口）
        if MERCHANT_API_KEY and MERCHANT_API_SECRET:
            self._http = MerchantHttpClient(
                base_url=MERCHANT_API_BASE_URL,
                api_key=MERCHANT_API_KEY,
                api_secret=MERCHANT_API_SECRET,
                api_passphrase=MERCHANT_API_PASSPHRASE,
                timeout=MERCHANT_API_TIMEOUT,
            )
        else:
            self._http = None

        self.base_url = MERCHANT_API_BASE_URL.rstrip("/")
        self.timeout = MERCHANT_API_TIMEOUT

        # Bearer Token 降级模式下的请求头
        self._fallback_headers: dict[str, str] = {"Content-Type": "application/json"}
        if MERCHANT_API_TOKEN:
            self._fallback_headers["Authorization"] = f"Bearer {MERCHANT_API_TOKEN}"

    def _get(self, path: str, params: dict = None) -> dict:
        """发起 GET 请求"""
        if self._http:
            return self._http.get(path, params)
        # 降级：Bearer Token 模式
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params, headers=self._fallback_headers)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, params: dict = None, json_body: dict = None) -> dict:
        """发起 POST 请求"""
        if self._http:
            return self._http.post(path, params, json_body)
        # 降级：Bearer Token 模式
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, params=params, json=json_body, headers=self._fallback_headers)
            resp.raise_for_status()
            return resp.json()

    # ---------- 各工具对应的方法 ----------

    def query_order(self, order_id: str) -> dict:
        return self._get(f"/orders/{order_id}")

    def query_user_orders(self, user_id: str, status: str = "all") -> dict:
        params = {"user_id": user_id}
        if status and status != "all":
            params["status"] = status
        return self._get("/orders", params=params)

    def query_user_info(self, user_id: str) -> dict:
        return self._get(f"/users/{user_id}")

    def query_logistics(self, tracking_no: str, carrier: str = None) -> dict:
        params = {}
        if carrier:
            params["carrier"] = carrier
        return self._get(f"/logistics/{tracking_no}", params=params)

    def query_product(self, product_id: str) -> dict:
        return self._get(f"/products/{product_id}")

    def query_refund(self, order_id: str) -> dict:
        return self._get(f"/refund/{order_id}")

    def query_card_types(self) -> dict:
        return self._get("/api/v1/institution/card/type")

    def query_balance(self) -> dict:
        return self._get("/api/v1/institution/balance")

    def query_customer_accounts(self, acct_no: str = None) -> dict:
        params = {"acct_no": acct_no} if acct_no else None
        return self._get("/api/v1/customers/accounts", params=params)


# ====================== Merchant Agent ======================

class MerchantAgent:
    """
    商户数据查询 Agent
    - 接受用户查询，通过 Function Calling 调用业务接口，汇总后返回回答
    - chat() 接口签名与 KnowledgeAgent 保持一致，方便 Orchestrator 统一调用
    """

    def __init__(self, memory_manager=None):
        self.api = MerchantAPIClient()
        self.mm = memory_manager  # 可选，用于读取用户画像（如已知 user_id）

        self.llm = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=QWEN_BASE_URL,
        )

    def _build_system_prompt(self, user_id: str) -> str:
        memory_section = ""
        if self.mm and user_id:
            try:
                memories = self.mm.search_memory("订单 用户 购买记录", user_id)
                if memories:
                    lines = [m.get("memory", "") for m in memories if m.get("memory")]
                    if lines:
                        memory_section = "【用户历史记忆】\n" + "\n".join(f"- {l}" for l in lines[:5])
            except Exception:
                pass
        return MERCHANT_SYSTEM_PROMPT.format(memory_section=memory_section)

    def _dispatch_tool(self, tool_name: str, args: dict) -> str:
        """派发工具调用到对应的业务接口方法"""
        try:
            if tool_name == "query_order":
                result = self.api.query_order(args["order_id"])
            elif tool_name == "query_user_orders":
                result = self.api.query_user_orders(
                    args["user_id"], args.get("status", "all")
                )
            elif tool_name == "query_user_info":
                result = self.api.query_user_info(args["user_id"])
            elif tool_name == "query_logistics":
                result = self.api.query_logistics(
                    args["tracking_no"], args.get("carrier")
                )
            elif tool_name == "query_product":
                result = self.api.query_product(args["product_id"])
            elif tool_name == "query_refund":
                result = self.api.query_refund(args["order_id"])
            elif tool_name == "query_card_types":
                result = self.api.query_card_types()
            elif tool_name == "query_balance":
                result = self.api.query_balance()
            elif tool_name == "query_customer_accounts":
                result = self.api.query_customer_accounts(args.get("acct_no"))
            else:
                result = {"error": f"未知工具：{tool_name}"}

            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[MerchantAgent] 工具调用失败 {tool_name}: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def chat(
        self,
        user_input: str,
        user_id: str,
        conversation_history: Optional[list[dict]] = None,
        update_memory: bool = True,
    ) -> dict:
        """
        商户数据查询对话（同步）
        返回格式与 KnowledgeAgent.chat() 一致
        """
        conversation_history = conversation_history or []

        system_prompt = self._build_system_prompt(user_id)
        messages = [{"role": "system", "content": system_prompt}]

        # 带入历史（最近 6 条）
        for msg in conversation_history[-6:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        tools_called = []
        tool_results = []
        max_rounds = 5  # 防止无限 tool call 循环

        for round_i in range(max_rounds):
            response = self.llm.chat.completions.create(
                model=QWEN_MODEL,
                messages=messages,
                tools=MERCHANT_TOOLS,
                tool_choice="auto",
                temperature=0.3,   # 数据查询场景用低温，确保结果稳定
                max_tokens=2000,
            )
            message = response.choices[0].message

            # 没有工具调用 → 最终回复
            if not message.tool_calls:
                answer = message.content or "（无回复）"
                break

            # 有工具调用 → 执行并喂回
            messages.append(message.model_dump())
            for tc in message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info(f"[MerchantAgent] 调用工具：{fn_name}({fn_args})")
                tool_result = self._dispatch_tool(fn_name, fn_args)
                logger.info(f"[MerchantAgent] 接口返回数据：{tool_result}")
                tools_called.append({"tool": fn_name, "args": fn_args})
                tool_results.append(tool_result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
        else:
            answer = "抱歉，查询过程中遇到了问题，请稍后重试。"

        return {
            "answer": answer,
            "rag_sources": [],          # 商户 Agent 不使用 RAG
            "memories_used": [],
            "memory_archived": False,
            "model": QWEN_MODEL,
            "agent": "merchant",
            "tools_called": tools_called,
            "tools_results": [json.loads(tr) for tr in tool_results] if tool_results else [],
        }
