"""
清鉴 (Qing Jian) — FastAPI 后端服务
将 agent4.py 的 MasterReasoner 封装为 REST API，
与前端 清鉴-清史研究智能体.html 对接。

启动方式：
    python server.py
    或
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import re
import json
import threading
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# ============================================================
# 导入 Agent 模块
# ============================================================
from agent4 import MasterReasoner

# ============================================================
# 全局状态
# ============================================================
agent_lock = threading.Lock()
master: MasterReasoner | None = None


# ============================================================
# FastAPI 应用
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载知识库，关闭时清理"""
    global master
    print("=" * 60)
    print("🔍 清鉴 · 清史研究智能体 — 后端服务启动中...")
    print("=" * 60)
    print("⏳ 正在初始化知识库（加载三种检索索引，约需 1-2 分钟）...")
    try:
        master = MasterReasoner()
        print("✅ 知识库初始化完成！服务就绪。")
        print(f"   - 档案 (ArchiveRetriever): {len(master.kb.archive_retriever.documents) if hasattr(master.kb.archive_retriever, 'documents') else '已加载'} 条")
        print("   - 词典 (DictFusionSearcher): 已加载")
        print("   - 编年 (SmartHistoricalRetriever): 已加载")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 知识库初始化失败: {e}")
        traceback.print_exc()
        raise
    yield
    print("🛑 服务关闭。")


app = FastAPI(
    title="清鉴 API",
    description="清史研究智能体后端 — 词典·编年·档案三库融合检索",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 请求/响应模型
# ============================================================
class ChatRequest(BaseModel):
    query: str
    session_id: str = ""


class ChatResponse(BaseModel):
    answer: str
    references: list[dict]


# ============================================================
# 报告解析器
# ============================================================
def parse_report_and_build_response(report: str, memory: dict) -> dict:
    """
    将 MasterReasoner 生成的完整报告拆分为前端所需格式。

    报告结构（由 FinalReportAgent.compose 生成）：
        [学术考据正文，含 [DICT-1]、[ANNALS-1] 等引用标记]

        # 📚 原始证据回溯附录
        (系统自动生成的详细证据块)

    转换逻辑：
        1. 分离正文 / 附录
        2. 按 dict → annals → memorials 顺序收集 evidence
        3. 重映射引用标记：[DICT-1]→[1], [ANNALS-1]→[2], ...
        4. 从 evidence.items 构建前端引用卡片
    """
    # 1. 分离正文和附录（兼容多种附录标题变体）
    appendix_markers = [
        "# 📚 原始证据回溯附录",
        "## 📚 原始证据回溯附录",
        "# 📚 原始证据回溯附录 (System Generated)",
        "原始证据回溯附录",
    ]
    body = report.strip()
    for marker in appendix_markers:
        if marker in report:
            body = report.split(marker)[0].strip()
            break

    # 2. 收集所有 evidence，按 dict → annals → memorials 顺序
    evidence_list: list[tuple[str, dict]] = []
    for db_type in ["dict", "annals", "memorials"]:
        for ev in memory.get(db_type, []):
            evidence_list.append((db_type, ev))

    # 构建引用映射: "DICT-1" → 1, "ANNALS-1" → 2, ...
    citation_map: dict[str, int] = {}
    for i, (_, ev) in enumerate(evidence_list, start=1):
        eid = ev.get("evidence_id", "")
        if eid:
            citation_map[eid] = i

    # 3. 重映射正文中的引用标记
    def replace_citation(match: re.Match) -> str:
        eid = match.group(1)
        new_num = citation_map.get(eid)
        return f"[{new_num}]" if new_num is not None else match.group(0)

    body = re.sub(
        r'\[(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)\]',
        replace_citation,
        body,
    )

    # 4. 构建 references 列表
    references = []
    for db_type, ev in evidence_list:
        ref = build_reference(db_type, ev)
        if ref:
            references.append(ref)

    return {"answer": body, "references": references}


def build_reference(db_type: str, ev: dict) -> dict | None:
    """根据一条 evidence 构建前端引用卡片"""
    if db_type == "dict":
        return _build_dict_reference(ev)
    elif db_type == "annals":
        return _build_chronicle_reference(ev)
    elif db_type == "memorials":
        return _build_archive_reference(ev)
    return None


def _safe_score(value, default=0.70) -> float:
    """安全地将分值转换为 0~1 的浮点数"""
    if value is None or value == "":
        return default
    try:
        s = float(value)
        # 如果明显是百分制或其他量纲，尝试归一化
        if s > 10:
            s = s / 100.0 if s <= 100 else s / 1000.0
        return max(0.0, min(1.0, s))
    except (ValueError, TypeError):
        return default


def _get_item_text(item: dict) -> str:
    """从检索结果条目中提取文本，兼容多种字段名"""
    for key in ("text", "释义", "内容精要", "modern_summary", "source_text",
                "原始记载", "记载摘要", "content", "excerpt", "definition"):
        val = item.get(key, "")
        if val:
            return str(val)
    return ""


def _build_dict_reference(ev: dict) -> dict:
    items = ev.get("items", [])
    refined = ev.get("refined", "")
    eid = ev.get("evidence_id", "")

    if items:
        item = items[0]
        term = item.get("term") or "未知词条"
        source = item.get("source") or "未知来源"
        score = _safe_score(item.get("final_score"))
    else:
        term = "词典条目"
        source = "未知来源"
        score = 0.5

    # 优先用 refined（精炼摘要），其次从 item 中提取原文
    excerpt = refined or (_get_item_text(items[0]) if items else "")

    return {
        "id": eid,
        "type": "dictionary",
        "title": term,
        "excerpt": excerpt,
        "source": source,
        "date": "—",
        "relevance": round(score, 2),
    }


def _build_chronicle_reference(ev: dict) -> dict:
    items = ev.get("items", [])
    refined = ev.get("refined", "")
    eid = ev.get("evidence_id", "")

    if items:
        item = items[0]
        era = item.get("era") or item.get("emperor") or ""
        event_type_raw = item.get("event_type") or ""
        if isinstance(event_type_raw, list):
            event_type = "、".join(str(e) for e in event_type_raw)
        else:
            event_type = str(event_type_raw)
        date = item.get("date") or "未知日期"
        source_text = item.get("source_text") or ""
        score = _safe_score(item.get("score"))

        # 构建标题
        parts = [p for p in [era, event_type] if p]
        title = "·".join(parts) if parts else "编年事件"
    else:
        title = "编年事件"
        date = "未知日期"
        source_text = ""
        score = 0.5

    return {
        "id": eid,
        "type": "chronicle",
        "title": title,
        "classical_text": source_text,
        "modern_text": refined,
        "source": "《清实录》",
        "date": date,
        "relevance": round(score, 2),
    }


def _build_archive_reference(ev: dict) -> dict:
    items = ev.get("items", [])
    refined = ev.get("refined", "")
    eid = ev.get("evidence_id", "")

    if items:
        item = items[0]
        title = item.get("title") or "无题"
        date = item.get("date") or "未知日期"
        author = item.get("author") or "未知"
        score = _safe_score(item.get("final_score"))
    else:
        title = "档案文献"
        date = "未知日期"
        author = "未知"
        score = 0.5

    # 优先用 refined（精炼摘要），其次从 item 中提取原文
    excerpt = refined or (_get_item_text(items[0]) if items else "")

    return {
        "id": eid,
        "type": "archive",
        "title": title,
        "excerpt": excerpt,
        "source": "中国第一历史档案馆藏",
        "date": date,
        "relevance": round(score, 2),
        "memorial_from": author,
        "memorial_to": "",
    }


# ============================================================
# API 端点
# ============================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    主对话端点 — 三库融合检索 + LLM 考据报告。

    请求：{ query, session_id }
    响应：{ answer (Markdown), references[] }
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询不能为空。")

    if master is None:
        raise HTTPException(
            status_code=503,
            detail="知识库尚未加载完成，请稍后重试。",
        )

    try:
        # 加锁防止 os.chdir 并发冲突
        with agent_lock:
            print(f"\n📩 收到查询: {query}")
            report = master.run(query)
            result = parse_report_and_build_response(report, master.memory)
            print(f"✅ 查询完成 — 引用数: {len(result['references'])}")
        return result

    except Exception as e:
        print(f"❌ 查询处理失败: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"处理查询时出错：{str(e)}",
        )


@app.get("/api/health")
async def health():
    """健康检查"""
    if master is None:
        return {"status": "loading", "message": "知识库加载中..."}
    return {"status": "ready", "message": "服务就绪"}


# ============================================================
# 静态文件服务
# ============================================================
FRONTEND_PATH = Path(__file__).parent / "清鉴-清史研究智能体.html"


@app.get("/")
async def serve_frontend():
    """提供前端 HTML 页面"""
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="前端文件未找到。")
    return FileResponse(FRONTEND_PATH)


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        timeout_keep_alive=300,    # SSE/长轮询预留
        log_level="info",
    )
