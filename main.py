"""
FastAPI 后端接口
提供：
- POST /chat           - 普通对话
- GET  /chat/stream    - SSE 流式对话
- POST /knowledge/add  - 上传文本到知识库
- POST /knowledge/file - 上传文件到知识库
- GET  /knowledge/list - 查看知识库来源列表
- DELETE /knowledge/{source} - 删除知识库来源
- GET  /memory/{user_id}     - 查看用户记忆
- DELETE /memory/{user_id}   - 清除用户记忆
- GET  /health         - 健康检查
"""
import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import APP_HOST, APP_PORT, DEBUG
from rag_knowledge_base import KnowledgeBase
from memory_manager import MemoryManager
from agent import CustomerServiceAgent

# ====================== 日志配置 ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局单例
kb: Optional[KnowledgeBase] = None
mm: Optional[MemoryManager] = None
agent: Optional[CustomerServiceAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global kb, mm, agent
    logger.info("初始化知识库...")
    kb = KnowledgeBase()
    logger.info("初始化记忆管理器...")
    mm = MemoryManager()
    logger.info("初始化客服 Agent...")
    agent = CustomerServiceAgent(kb, mm)
    logger.info("✅ 服务启动完成")
    # 后台异步写入演示知识（不阻塞启动）
    asyncio.create_task(_async_seed_demo_knowledge(kb))
    yield
    logger.info("服务关闭")


async def _async_seed_demo_knowledge(knowledge_base: KnowledgeBase):
    """异步后台写入演示知识"""
    try:
        await asyncio.get_event_loop().run_in_executor(None, _seed_demo_knowledge, knowledge_base)
    except Exception as e:
        logger.warning(f"演示知识写入失败（请检查 API Key）：{e}")


# ====================== 初始化组件 ======================
app = FastAPI(
    title="智能客服 API",
    description="基于 mem0ai + Qwen + ChromaDB 的智能客服系统",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _seed_demo_knowledge(knowledge_base: KnowledgeBase):
    """首次启动时写入演示知识内容"""
    if knowledge_base.count() > 0:
        return  # 已有内容，跳过

    demo_docs = [
        {
            "text": "退款政策：购买后7天内可申请无理由退款。退款申请通过后，款项将在3-5个工作日内原路退回。超过7天的退款申请需提供商品质量问题证明。",
            "metadata": {"source": "退款政策.txt", "category": "售后"},
        },
        {
            "text": "配送说明：普通配送3-5天到货，支持顺丰快递、圆通、中通等快递公司。购买满99元免运费，不足99元收取运费8元。偏远地区（新疆、西藏、内蒙古等）运费另计。",
            "metadata": {"source": "配送说明.txt", "category": "物流"},
        },
        {
            "text": "会员权益：注册会员可享受积分购物返现，每消费1元积1分，100分可兑换1元优惠券。会员等级分为普通会员、银卡会员、金卡会员、钻石会员，不同等级享受不同折扣。",
            "metadata": {"source": "会员权益.txt", "category": "会员"},
        },
        {
            "text": "售后服务：商品在收货后30天内出现非人为损坏，可申请免费维修或更换。联系售后客服请拨打400-800-1234，工作时间为周一至周日9:00-21:00。",
            "metadata": {"source": "售后服务.txt", "category": "售后"},
        },
        {
            "text": "支付方式：支持微信支付、支付宝、银行卡、花呗分期、京东白条等多种支付方式。下单后24小时内未付款，订单将自动取消。",
            "metadata": {"source": "支付方式.txt", "category": "支付"},
        },
        {
            "text": "常见问题：1.如何查看订单状态？在「我的订单」页面可查看所有订单的实时状态。2.如何修改收货地址？在订单发货前可联系客服修改地址。3.如何使用优惠券？在结算页面的「优惠券」选项中选择可用券。",
            "metadata": {"source": "FAQ.txt", "category": "FAQ"},
        },
    ]

    for doc in demo_docs:
        knowledge_base.add_text(doc["text"], metadata=doc["metadata"])

    logger.info(f"✅ 已写入 {len(demo_docs)} 条演示知识")


# ====================== 数据模型 ======================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户输入")
    user_id: str = Field(default="default_user", description="用户 ID")
    history: list[dict] = Field(default=[], description="历史对话 [{role, content}]")


class ChatResponse(BaseModel):
    answer: str
    rag_sources: list[dict]
    memories_used: list[str]
    model: str


class AddTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="知识内容")
    source: str = Field(default="手动录入", description="来源标识")
    category: str = Field(default="通用", description="分类")


# ====================== 对话接口 ======================

@app.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(req: ChatRequest):
    """普通对话接口"""
    if not agent:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后")
    try:
        result = agent.chat(
            user_input=req.message,
            user_id=req.user_id,
            conversation_history=req.history,
        )
        return result
    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"对话失败：{str(e)}")


@app.get("/chat/stream", tags=["对话"])
async def stream_chat(message: str, user_id: str = "default_user"):
    """流式对话（SSE）"""
    if not agent:
        raise HTTPException(status_code=503, detail="服务初始化中")

    async def event_generator():
        try:
            async for chunk in agent.stream_chat(
                user_input=message,
                user_id=user_id,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ====================== 知识库接口 ======================

@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge_text(req: AddTextRequest):
    """添加文本到知识库"""
    if not kb:
        raise HTTPException(status_code=503, detail="服务初始化中")
    try:
        ids = kb.add_text(
            req.text,
            metadata={"source": req.source, "category": req.category},
        )
        return {"success": True, "chunk_count": len(ids), "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/file", tags=["知识库"])
async def upload_knowledge_file(file: UploadFile = File(...)):
    """上传文件到知识库（支持 .txt/.md/.pdf/.docx）"""
    if not kb:
        raise HTTPException(status_code=503, detail="服务初始化中")

    allowed = {".txt", ".md", ".pdf", ".docx"}
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    # 写入临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ids = kb.add_file(tmp_path)
        return {
            "success": True,
            "filename": file.filename,
            "chunk_count": len(ids),
        }
    except Exception as e:
        error_msg = str(e)
        # 识别 401 / API Key 无效错误，提示清晰
        if "401" in error_msg or "invalid_api_key" in error_msg.lower() or "Incorrect API key" in error_msg:
            raise HTTPException(
                status_code=500,
                detail=(
                    "API Key 无效或已过期！\n\n"
                    "请到阿里云百炼控制台（dashscope.aliyun.com）重新获取 API Key，"
                    "然后编辑项目根目录的 .env 文件替换 DASHSCOPE_API_KEY 的值，"
                    "保存后重启服务即可。"
                ),
            )
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        os.unlink(tmp_path)


@app.get("/knowledge/list", tags=["知识库"])
async def list_knowledge():
    """列出知识库所有来源"""
    if not kb:
        raise HTTPException(status_code=503, detail="服务初始化中")
    sources = kb.list_sources()
    return {"total_chunks": kb.count(), "sources": sources}


@app.delete("/knowledge/{source_name}", tags=["知识库"])
async def delete_knowledge(source_name: str):
    """按来源名称删除知识"""
    if not kb:
        raise HTTPException(status_code=503, detail="服务初始化中")
    deleted = kb.delete_by_source(source_name)
    return {"success": True, "deleted_chunks": deleted}


@app.get("/knowledge/{source_name}/chunks", tags=["知识库"])
async def get_knowledge_chunks(source_name: str):
    """查看指定文档来源的所有文本块内容"""
    if not kb:
        raise HTTPException(status_code=503, detail="服务初始化中")
    try:
        chunks = kb.get_by_source(source_name)
        return {
            "source": source_name,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "heading": c["metadata"].get("heading"),
                    "section_index": c["metadata"].get("section_index", 0),
                    "chunk_index": c["metadata"].get("chunk_index", 0),
                    "total_chunks": c["metadata"].get("total_chunks", 1),
                }
                for c in chunks
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================== 记忆接口 ======================

@app.get("/memory/{user_id}/{memory_id}/history", tags=["记忆"])
async def get_memory_history(user_id: str, memory_id: str):
    """获取单条记忆的变更历史"""
    if not mm:
        raise HTTPException(status_code=503, detail="服务初始化中")
    history = mm.get_memory_history(memory_id)
    return {"memory_id": memory_id, "history_count": len(history), "history": history}


@app.get("/memory/{user_id}/{memory_id}", tags=["记忆"])
async def get_memory_detail(user_id: str, memory_id: str):
    """获取单条记忆详情"""
    if not mm:
        raise HTTPException(status_code=503, detail="服务初始化中")
    memory = mm.get_memory_by_id(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在或已删除")
    return memory


@app.put("/memory/{user_id}/{memory_id}", tags=["记忆"])
async def update_memory_detail(user_id: str, memory_id: str, data: dict):
    """更新单条记忆内容"""
    if not mm:
        raise HTTPException(status_code=503, detail="服务初始化中")
    content = data.get("content", "")
    if not content.strip():
        raise HTTPException(status_code=400, detail="记忆内容不能为空")
    ok = mm.update_memory(memory_id, content)
    if not ok:
        raise HTTPException(status_code=500, detail="更新失败")
    return {"success": True, "memory_id": memory_id}


@app.get("/memory/{user_id}", tags=["记忆"])
async def get_user_memory(user_id: str, category: str = None):
    """查看用户所有记忆（可选按分类过滤）"""
    if not mm:
        raise HTTPException(status_code=503, detail="服务初始化中")
    all_memories = mm.get_all_memory(user_id)
    if category:
        all_memories = [m for m in all_memories
                        if (m.get("metadata") or {}).get("category") == category]
    return {"user_id": user_id, "memory_count": len(all_memories), "memories": all_memories}


@app.delete("/memory/{user_id}/{memory_id}", tags=["记忆"])
async def delete_user_memory_item(user_id: str, memory_id: str):
    """删除用户单条记忆"""
    if not mm:
        raise HTTPException(status_code=503, detail="服务初始化中")
    ok = mm.delete_memory_by_id(memory_id)
    return {"success": ok, "user_id": user_id, "memory_id": memory_id}


@app.delete("/memory/{user_id}", tags=["记忆"])
async def clear_user_memory(user_id: str):
    """清除用户所有记忆"""
    if not mm:
        raise HTTPException(status_code=503, detail="服务初始化中")
    success = mm.delete_user_memory(user_id)
    return {"success": success, "user_id": user_id}


# ====================== 健康检查 ======================

@app.get("/health", tags=["系统"])
async def health():
    return {
        "status": "ok",
        "knowledge_chunks": kb.count() if kb else 0,
        "services": {
            "knowledge_base": kb is not None,
            "memory_manager": mm is not None,
            "agent": agent is not None,
        },
    }


# ====================== 前端页面 ======================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """返回前端聊天页面"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>前端未找到，请检查 static/index.html</h1>")


# ====================== 启动 ======================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
        log_level="info",
    )
