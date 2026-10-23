import re
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from config import get_llm_client, get_llm_model

# ==========================================
# Windows 控制台 UTF-8 编码修复
# ==========================================
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ==========================================
# 强制离线
# ==========================================
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import jieba

# ==========================================
# 路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
archive_dir = os.path.join(current_dir, "archive_retrieval2")
dict_dir = os.path.join(current_dir, "dict_retrieval", "dict_search")
annal_dir = os.path.join(current_dir, "annal_retrieval")

jieba.dt.tmp_dir = os.path.join(current_dir, "archive_retrieval/cache")

sys.path.append(archive_dir)
sys.path.append(dict_dir)
sys.path.append(annal_dir)

# ==========================================
# 全局配置
# ==========================================

client = get_llm_client()
MODEL_NAME = get_llm_model("deepseek-v4-flash")



class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


# ==========================================
# 检索模块
# ==========================================
from archive_retriever import ArchiveRetriever
from search_dict_fusion_core import DictFusionSearcher
from a import SmartHistoricalRetriever


class KnowledgeBase:
    def __init__(self):
        print(f"{Colors.BLUE}▶ [系统初始化] 正在加载各类检索缓存...{Colors.ENDC}")

        print("  └-- 正在加载 ArchiveRetriever (档案)...")
        os.chdir(archive_dir)
        self.archive_retriever = ArchiveRetriever(data_format="refined")
        self.archive_retriever.load_or_build()
        os.chdir(current_dir)

        print("  └-- 正在加载 DictFusionSearcher (词典)...")
        os.chdir(dict_dir)
        try:
            self.dict_searcher = DictFusionSearcher()
            print("  词典加载成功。")
        except Exception as e:
            print(f"  ⚠ 词典加载失败 (非致命): {e}")
            self.dict_searcher = None
        os.chdir(current_dir)

        print("  └-- 正在加载 SmartHistoricalRetriever (编年实录)...")
        os.chdir(annal_dir)
        annal_files = [
            os.path.join(annal_dir, "data/光绪朝结构化数据摘要.json"),
            os.path.join(annal_dir, "data/乾隆朝结构化数据摘要_aligned.json"),
            os.path.join(annal_dir, "data/康熙朝结构化数据摘要_aligned.json")
        ]
        self.annals_retriever = SmartHistoricalRetriever(file_paths=annal_files)
        os.chdir(current_dir)

        print(f"{Colors.GREEN}✅ 所有检索缓存加载完毕！{Colors.ENDC}\n" + "=" * 50)

    def hybrid_search(self, db_type, query, time_window=None):
        """
        返回结构化证据：
        {
          "db_type": "...",
          "query": "...",
          "time_window": "...",
          "raw_text": "给 LLM 看的文本",
          "items": [{"title":..., "date":..., "text":..., "source":...}]
        }
        """
        print(f"{Colors.BLUE}▶ [底层检索] 目标库: {db_type} | Query: '{query}' | 时间窗: {time_window}{Colors.ENDC}")

# KnowledgeBase.hybrid_search 里的 memorials 分支 ---
        if db_type == "memorials":
            # 调用 v3 模块的 search 方法
            raw_results = self.archive_retriever.search(query, top_k=5)
            
            if not raw_results:
                return self._empty_result(db_type, query, time_window, "无相关档案记载。")

            items = []
            formatted_docs = []
            for i, res in enumerate(raw_results, start=1):
                # 1. 提取原始记录
                record = res.get("record", {}) or {}
                
                # 2. 详细字段提取 (同时兼容繁简键名)
                f_score = res.get("final_score", 0)
                rel_score = res.get("relation_score", 0)
                matched = res.get("matched_entities", [])
                
                title = record.get("题名") or record.get("題名") or "无题"
                abstract = record.get("内容提要") or record.get("內容提要") or ""
                author = record.get("责任者") or record.get("責任者") or "未知"
                date = record.get("责任时间(中历)") or record.get("責任時間(中曆)") or "未知"
                dynasty = record.get("来源朝代") or "未知"
                reign = record.get("年号") or dynasty
                archive_idx = res.get("index") # v3 检索器返回的原始索引

                # 3. 构建给 LLM 看的文本块
                doc_str = (
                    f"【证据-{i}：{title}】\n"
                    f"评分：综合 {f_score} | 关系加权 {rel_score}\n"
                    f"匹配实体：{matched}\n"
                    f"时间：{date} | 责任者：{author}\n"
                    f"内容精要：{abstract}"
                )
                formatted_docs.append(doc_str)

                # ============================================================
                # ！！！核心修复：这里存入的 Key 必须和 format_source_items 对应 ！！！
                # ============================================================
                items.append({
                    "local_id": f"档案-{i}",  # 这里的 ID 也会显示在报告里
                    "title": title,
                    "date": date,           # 补全这个字段，报告就不会显示"未知时间"
                    "author": author,       # 补全这个字段，报告就不会显示"未知责任者"
                    "dynasty": dynasty,     # 补全这个字段
                    "reign_title": reign,   # 补全这个字段
                    "index": archive_idx,   # 对应报告里的 archive_index
                    "rank": res.get("rank", i),
                    "final_score": f_score, # 对应报告里的 得分
                    "bm25_score": res.get("bm25_score"),
                    "vector_score": res.get("vector_score"),
                    "relation_score": rel_score,
                    "text": abstract,
                    "original_text": record.get("source_text") or "",
                    "matched_entities": matched
                })
            
            return self._result(db_type, query, time_window, formatted_docs, items)
              
        if db_type == "dict":
            if self.dict_searcher is None:
                return self._empty_result(db_type, query, time_window, "词典检索模块未加载（初始化失败/已跳过）。")
            os.chdir(dict_dir)
            raw_results = self.dict_searcher.search(query)
            os.chdir(current_dir)

            if not raw_results or not isinstance(raw_results, dict):
                return self._empty_result(db_type, query, time_window, "词典中无相关词条解释。")

            results_list = raw_results.get("results", [])
            if not results_list:
                return self._empty_result(db_type, query, time_window, f"词典中未检索到 '{query}' 的相关解释。")

            items = []
            formatted_docs = []
            for i, item in enumerate(results_list, start=1):
                # 字段探针确认：
                # 词典条目的 entry / definition / source 等字段都在 item["entity"] 下。
                # 这里保留少量旧字段 fallback，避免将来检索器返回格式变化时直接失效。
                entity = item.get("entity", {}) or {}

                term = entity.get("entry") or item.get("term") or item.get("条目") or "未知词条"
                definition = (
                    entity.get("definition")
                    or item.get("definition")
                    or item.get("解释")
                    or "无解释"
                )
                source = entity.get("source") or item.get("source") or item.get("出处") or ""
                author = entity.get("author", "")
                source_file = entity.get("source_file", "")
                entry_id = entity.get("entry_id", "")
                chunk_id = entity.get("chunk_id", item.get("id", ""))
                row_id = entity.get("row_id", "")
                data_type = entity.get("data_type", "")
                confidence = item.get("confidence", "")
                final_score = item.get("final_score", "")
                match_type = item.get("match_type", "")
                match_source = item.get("match_source", "")

                doc_str = (
                    f"【词典-{i}】{term}\n"
                    f"来源：{source or '未知'} | 作者：{author or '未知'} | 来源文件：{source_file or '未知'}\n"
                    f"条目ID：{entry_id or '未知'} | 分块ID：{chunk_id or '未知'} | 行号：{row_id or '未知'}\n"
                    f"匹配置信：{confidence or '未知'} | 综合得分：{final_score} | 匹配类型：{match_type or match_source or '未知'}\n"
                    f"释义：{definition}"
                )
                formatted_docs.append(doc_str)
                items.append({
                    "local_id": f"词典-{i}",
                    "id": item.get("id", ""),
                    "entry_id": entry_id,
                    "chunk_id": chunk_id,
                    "row_id": row_id,
                    "term": term,
                    "source": source,
                    "author": author,
                    "source_file": source_file,
                    "data_type": data_type,
                    "confidence": confidence,
                    "final_score": final_score,
                    "match_type": match_type,
                    "match_source": match_source,
                    "text": definition
                })
            return self._result(db_type, query, time_window, formatted_docs, items)

        if db_type == "annals":
            raw_results = self.annals_retriever.search(query, max_k=8, min_score=0.60, drop_threshold=0.1)
            if not raw_results:
                return self._empty_result(db_type, query, time_window, f"实录编年中未检索到关于 '{query}' 的高度相关记载。")

            items = []
            formatted_docs = []
            for i, item in enumerate(raw_results, start=1):
                # 字段探针确认：
                # 编年结果正文信息在 item["data"] 下；原始记载为 source_text，
                # 摘要为 refined_data.micro_summary。
                d = item.get("data", {}) or {}
                refined = d.get("refined_data", {}) or {}

                era = d.get("era_source") or d.get("emperor") or "未知朝代"
                emperor = d.get("emperor", era)
                date = d.get("date", "未知日期")
                nianhao = d.get("nianhao", "")
                nongli_month = d.get("nongli_month", "")
                nongli_day = d.get("nongli_day", "")
                source_text = d.get("source_text", "")
                summary = refined.get("micro_summary", "无摘要")
                event_type = refined.get("event_type", "")
                original_index = d.get("original_index")
                score = item.get("score", "")
                source = f"annals_original_index:{original_index}" if original_index is not None else ""

                lunar_date = "".join(part for part in [nianhao, nongli_month, nongli_day] if part)

                doc_str = (
                    f"【编年-{i}】{era}朝 | 日期：{date}"
                    + (f" | 农历：{lunar_date}" if lunar_date else "")
                    + (f" | 原始索引：{original_index}" if original_index is not None else "")
                    + (f" | 相似度：{score}" if score != "" else "")
                    + "\n"
                    + (f"事件类型：{event_type}\n" if event_type else "")
                    + (f"原始记载：{source_text}\n" if source_text else "")
                    + f"记载摘要：{summary}"
                )
                formatted_docs.append(doc_str)
                items.append({
                    "local_id": f"编年-{i}",
                    "original_index": original_index,
                    "era": era,
                    "emperor": emperor,
                    "date": date,
                    "nianhao": nianhao,
                    "nongli_month": nongli_month,
                    "nongli_day": nongli_day,
                    "source": source,
                    "score": score,
                    "event_type": event_type,
                    "entities_people": refined.get("entities_people", []),
                    "entities_office": refined.get("entities_office", []),
                    "entities_geo": refined.get("entities_geo", []),
                    "source_text": source_text,
                    "text": summary
                })
            return self._result(db_type, query, time_window, formatted_docs, items)

        return self._empty_result(db_type, query, time_window, "数据库类型错误。")

    @staticmethod
    def _result(db_type, query, time_window, formatted_docs, items):
        return {
            "db_type": db_type,
            "query": query,
            "time_window": time_window,
            "raw_text": "\n\n---\n\n".join(formatted_docs),
            "items": items
        }

    @staticmethod
    def _empty_result(db_type, query, time_window, message):
        return {
            "db_type": db_type,
            "query": query,
            "time_window": time_window,
            "raw_text": message,
            "items": []
        }


# 延迟到 main 中初始化，避免 import agent.py 时就加载全部缓存
kb = None


def _brief_text(text, max_len=120):
    """用于来源条目展示：截断过长文本，避免日志和最终报告过长。"""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def format_source_items(db_type, items, max_items=8):
    """
    将结构化检索条目压缩成"可辨识来源"的引用说明。
    目的：让最终报告/日志/observation 不只显示 [DICT-1]，还显示来自哪个具体条目。
    """
    if not items:
        return "无可辨识来源条目"

    lines = []
    for item in items[:max_items]:
        local_id = item.get("local_id", "条目")

        if db_type == "dict":
            lines.append(
                f"- 【{local_id}】"
                f"词条={item.get('term') or '未知'}；"
                f"来源={item.get('source') or '未知'}；"
                f"来源文件={item.get('source_file') or '未知'}；"
                f"entry_id={item.get('entry_id') or '未知'}；"
                f"chunk_id={item.get('chunk_id') or item.get('id') or '未知'}；"
                f"row_id={item.get('row_id') or '未知'}；"
                f"匹配={item.get('match_type') or item.get('match_source') or '未知'}；"
                f"得分={item.get('final_score') if item.get('final_score') != '' else '未知'}"
            )

        elif db_type == "memorials":
            lines.append(
                f"- 【{local_id}】"
                f"题名={item.get('title') or '无题'}；"
                f"archive_index={item.get('index') if item.get('index') is not None else '未知'}；"
                f"排序={item.get('rank') if item.get('rank') is not None else '未知'}；"
                f"时间={item.get('date') or '未知'}；"
                f"年号={item.get('reign_title') or '未知'}；"
                f"责任者={item.get('author') or '未知'}；"
                f"朝代={item.get('dynasty') or '未知'}；"
                f"得分={item.get('final_score') if item.get('final_score') != '' else '未知'}"
            )

        elif db_type == "annals":
            lunar_date = "".join(
                str(item.get(k) or "") for k in ["nianhao", "nongli_month", "nongli_day"]
            )
            lines.append(
                f"- 【{local_id}】"
                f"朝代={item.get('era') or '未知'}；"
                f"皇帝={item.get('emperor') or '未知'}；"
                f"日期={item.get('date') or '未知'}；"
                f"农历={lunar_date or '未知'}；"
                f"original_index={item.get('original_index') if item.get('original_index') is not None else '未知'}；"
                f"事件类型={item.get('event_type') or '未知'}；"
                f"相似度={item.get('score') if item.get('score') != '' else '未知'}；"
                f"原始记载={_brief_text(item.get('source_text'), 100) or '无'}"
            )

        else:
            lines.append(f"- 【{local_id}】{json.dumps(item, ensure_ascii=False)[:300]}")

    if len(items) > max_items:
        lines.append(f"- ……另有 {len(items) - max_items} 条来源条目未展开")

    return "\n".join(lines)


# ==========================================
# RiD 史料精炼员 Agent
# ==========================================
class RiDAgent:
    @staticmethod
    def refine(raw_text, source_type, user_query):
        print(f"{Colors.YELLOW}▶ [RiD Agent] 正在对 {source_type} 检索结果进行结构化提炼...{Colors.ENDC}")
        if source_type == "dict":
            prompt_context = "你是一个词典摘要员。请阅读以下词典条目，提取出与用户问题最相关的定义和制度背景。"
            limit = "不超过180字"
        else:
            prompt_context = "你是一个史料精炼员。请阅读以下史料，提取与用户问题最相关的核心事实。"
            limit = "不超过350字"

        prompt = f"""{prompt_context}

用户问题：
{user_query}

要求：
- {limit}
- 保留原文中的时间、人物、机构、事件
- 不要编造未出现在材料中的事实
- 如材料无关，请明确说"相关性不足"

原始资料：
{raw_text}
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()


# ==========================================
# 证据冲突分析员 Agent
# ==========================================
class ConflictAnalyzerAgent:
    @staticmethod
    def verify(annals_evidence, memorials_evidence):
        if not annals_evidence or not memorials_evidence:
            return ""

        print(f"{Colors.RED}▶ [冲突分析 Agent] 正在对比《实录》定论与《档案》私叙的底层差异...{Colors.ENDC}")
        prompt = f"""你是一个历史考据专家。请对比以下两份史料摘要，指出官方定论（实录/编年）与档案/奏折材料之间是否存在差异。
如果没有明显矛盾，请说"未发现明确矛盾"，不要强行制造冲突。
请用 150-250 字说明。

实录/编年证据：
{annals_evidence}

档案/奏折证据：
{memorials_evidence}
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()


# ==========================================
# 最终报告生成 Agent
# ==========================================
class FinalReportAgent:
    @staticmethod
    def compose(user_query, memory):
        # 1. 准备给大模型看的"证据包摘要"（用于逻辑推导）
        evidence_summary_for_llm = []
        # 2. 准备给附录用的"全量证据详情"（程序自动拼装）
        full_appendix_items = []

        for db_type in ["dict", "annals", "memorials"]:
            for ev in memory.get(db_type, []):
                evidence_id = ev['evidence_id']
                
                # --- 环节 A: 给 LLM 读的简包 ---
                evidence_summary_for_llm.append(
                    f"【{evidence_id}】\n内容精炼：{ev['refined']}"
                )
                
                # --- 环节 B: 程序生成的"硬核附录" ---
                # 提取详细的来源信息
                source_refs = format_source_items(db_type, ev.get("items", []))
                
                # 拼接单条证据的完整档案
                # 这里加入了原文(raw_text)和精炼摘要(refined)
                appendix_block = f"""### 📌 证据编号：{evidence_id}
- **库类型**：{db_type}
- **检索式**：`{ev['query']}`
- **来源详情**：
{source_refs}
- **精炼事实**：
> {ev['refined']}
- **原始检索片段/原文回显**：
```text
{ev['raw_text']}
```
"""
                full_appendix_items.append(appendix_block)

        conflict_report = memory.get("conflict_report", "")
        evidence_context = "\n\n".join(evidence_summary_for_llm) or "无可用证据"

        # 3. 构造请求正文的 Prompt（告诉 LLM 不要写参考资料，程序会自动补）
        prompt = f"""你是清史研究专家。请基于以下提供的证据包，撰写一份结构严谨的"考据报告"。

【用户问题】
{user_query}

【证据包摘要】
{evidence_context}

【交叉互证/冲突发现】
{conflict_report or "未发现明显矛盾"}

【写作要求】
1. 直接给出学术结论，随后分点陈述理由。
2. 每一个判断点后必须标注对应的证据编号引用，例如 [ANNALS-1]、[MEMORIALS-2]、[DICT-1]。
3. 请仅撰写报告正文部分，【严禁】自行生成"参考依据"或"参考文献"章节，该部分将由系统后台程序自动生成。
4. 语言风格：专业、客观、严谨。
"""
        
        # 4. 调用 LLM 生成报告正文
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        report_body = response.choices[0].message.content.strip()

        # 5. 【核心逻辑】程序手动挂载"证据附录"
        final_full_report = f"""{report_body}

---
# 📚 原始证据回溯附录 (System Generated)

{"".join(full_appendix_items)}
"""
        return final_full_report



# ==========================================
# 日志保存
# ==========================================
class QueryLogger:
    def __init__(self, log_dir="query_logs"):
        self.log_dir = Path(current_dir) / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{ts}.txt"

    def append(self, user_query, report, memory, steps):
        with self.session_file.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"用户问题：{user_query}\n")
            f.write("-" * 90 + "\n")
            f.write("最终报告：\n")
            f.write(report + "\n")
            f.write("-" * 90 + "\n")
            f.write("执行轨迹：\n")
            for i, step in enumerate(steps, start=1):
                f.write(f"\n[Step {i}]\n")
                f.write(json.dumps(step, ensure_ascii=False, indent=2))
                f.write("\n")
            f.write("-" * 90 + "\n")
            f.write("证据摘要：\n")
            for db_type in ["dict", "annals", "memorials"]:
                for ev in memory.get(db_type, []):
                    f.write(f"\n【{ev['evidence_id']}】{db_type} | query={ev['query']} | time_window={ev.get('time_window')}\n")
                    f.write("来源条目：\n")
                    f.write(format_source_items(db_type, ev.get("items", [])) + "\n")
                    f.write("精炼摘要：\n")
                    f.write(ev["refined"] + "\n")
            if memory.get("conflict_report"):
                f.write("\n【conflict_report】\n")
                f.write(memory["conflict_report"] + "\n")

        print(f"{Colors.GREEN}📝 本次问答已写入：{self.session_file}{Colors.ENDC}")


# ==========================================
# 报告解析函数（从 server.py 移入，供 run_stream 使用）
# ==========================================
def _safe_score(value, default=0.70):
    """安全地将分值转换为 0~1 的浮点数"""
    if value is None or value == "":
        return default
    try:
        s = float(value)
        if s > 10:
            s = s / 100.0 if s <= 100 else s / 1000.0
        return max(0.0, min(1.0, s))
    except (ValueError, TypeError):
        return default


def _get_item_text(item):
    """从检索结果条目中提取文本，兼容多种字段名"""
    for key in ("text", "释义", "内容精要", "modern_summary", "source_text",
                "原始记载", "记载摘要", "content", "excerpt", "definition"):
        val = item.get(key, "")
        if val:
            return str(val)
    return ""


def _build_dict_reference(ev):
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
    excerpt = refined or (_get_item_text(items[0]) if items else "")
    return {
        "id": eid, "type": "dictionary", "title": term,
        "excerpt": excerpt, "source": source, "date": "—",
        "relevance": round(score, 2),
    }


def _build_chronicle_reference(ev):
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
        parts = [p for p in [era, event_type] if p]
        title = "·".join(parts) if parts else "编年事件"
    else:
        title = "编年事件"
        date = "未知日期"
        source_text = ""
        score = 0.5
    return {
        "id": eid, "type": "chronicle", "title": title,
        "classical_text": source_text, "modern_text": refined,
        "source": "《清实录》", "date": date,
        "relevance": round(score, 2),
    }


def _build_archive_reference(ev):
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
    excerpt = refined or (_get_item_text(items[0]) if items else "")
    return {
        "id": eid, "type": "archive", "title": title,
        "excerpt": excerpt, "source": "中国第一历史档案馆藏",
        "date": date, "relevance": round(score, 2),
        "memorial_from": author, "memorial_to": "",
    }


def build_reference(db_type, ev):
    if db_type == "dict":
        return _build_dict_reference(ev)
    elif db_type == "annals":
        return _build_chronicle_reference(ev)
    elif db_type == "memorials":
        return _build_archive_reference(ev)
    return None


def parse_report_and_build_response(report, memory):
    """将完整报告拆分为前端所需格式（正文 + 引用卡片）"""
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

    evidence_list = []
    for db_type in ["dict", "annals", "memorials"]:
        for ev in memory.get(db_type, []):
            evidence_list.append((db_type, ev))

    citation_map = {}
    for i, (_, ev) in enumerate(evidence_list, start=1):
        eid = ev.get("evidence_id", "")
        if eid:
            citation_map[eid] = i

    def replace_citation(match):
        eid = match.group(1)
        new_num = citation_map.get(eid)
        return f"[{new_num}]" if new_num is not None else match.group(0)

    body = re.sub(
        r'【(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)】',
        lambda m: f"[{m.group(1)}]",
        body,
    )
    body = re.sub(
        r'\[(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)\]',
        replace_citation,
        body,
    )
    body = re.sub(
        r'（(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)）',
        lambda m: f"[{citation_map.get(m.group(1), m.group(0))}]"
                  if citation_map.get(m.group(1)) else m.group(0),
        body,
    )
    body = re.sub(
        r'（(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)([、，,])(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)）',
        lambda m: f"[{citation_map.get(m.group(1), m.group(1))}] [{citation_map.get(m.group(3), m.group(3))}]"
                  if citation_map.get(m.group(1)) else m.group(0),
        body,
    )
    body = re.sub(
        r'\((DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)\)',
        lambda m: f"[{citation_map.get(m.group(1), m.group(0))}]"
                  if citation_map.get(m.group(1)) else m.group(0),
        body,
    )
    body = re.sub(r'\[交叉互证[^]]*\]', '', body)
    body = re.sub(r'[（(]交叉互证[^）)]*[）)]', '', body)
    body = re.sub(r'[（(]冲突发现[^）)]*[）)]', '', body)

    # 6. 中文括号内含有前缀文字的引用: （如MEMORIALS-1）→ （如[1]）
    #    先提取中文括号内容，对内部做引用替换，再拼回去
    def _fix_chinese_parens(m):
        inner = m.group(1)
        inner_fixed = re.sub(
            r'(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)',
            lambda m2: f"[{citation_map.get(m2.group(1), m2.group(1))}]"
                       if citation_map.get(m2.group(1)) else m2.group(0),
            inner,
        )
        return f"（{inner_fixed}）"

    body = re.sub(r'（([^）]+?)）', _fix_chinese_parens, body)

    # 7. 裸引用（无任何括号包裹的残留引用）: MEMORIALS-1 → [1]
    #    例如 "ANNALS-1与DICT-1所载事实" → "[1]与[2]所载事实"
    body = re.sub(
        r'(?<![\[\d])(DICT-\d+|ANNALS-\d+|MEMORIALS-\d+)(?![\d\]])',
        replace_citation,
        body,
    )

    references = []
    for db_type, ev in evidence_list:
        ref = build_reference(db_type, ev)
        if ref:
            references.append(ref)

    return {"answer": body, "references": references}


# ==========================================
# 主控与状态管理
# ==========================================
class MasterReasoner:
    def __init__(self):
        self.kb = KnowledgeBase() 
        self.reset_state()

    def reset_state(self):
        self.memory = {
            "dict": [],
            "annals": [],
            "memorials": [],
            "conflict_report": ""
        }
        self.steps = []
        self.messages = [{"role": "system", "content": self._get_system_prompt()}]

    def _get_system_prompt(self):
        return """你是清史考据主控系统。你需要通过严谨的多次跨库检索来回答复杂问题。

你有三种工具，请以特定 XML 格式输出你的规划。不要生成 markdown 代码块。
1. 查名词、制度、机构、人物称谓：
<action type="search" target="dict" query="关键词" />
2. 查官方叙述、时间节点、编年事件：
<action type="search" target="annals" query="关键词" time_window="年份或年代" />
3. 查档案、奏折、内幕细节、政策博弈：
<action type="search" target="memorials" query="关键词" time_window="年份或年代" />

规则：
- 每次回复请先用一两句话简要说明你的检索意图（选什么库、为什么），然后换行输出一个 <action ... />。
- 如果你认为不需要解释，直接输出 <action ... /> 也完全可以。
- 原则上优先覆盖 dict、annals、memorials 三类库；如果某类库明显无关，可以跳过。
- 收集足够证据后，输出：<action type="finish" />
- 不要在主控阶段写最终答案；最终报告会由报告 Agent 统一生成。
- 务必确保每次回复都至少包含一个有效的 <action> 标签。
"""

    @staticmethod
    def _parse_action(ai_reply):
        """
        比原来的单条正则更稳：允许属性换序，也允许多余说明文字。
        """
        tag_match = re.search(r"<action\b[^>]*?/?>", ai_reply, flags=re.S)
        if not tag_match:
            return None

        tag = tag_match.group(0)
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        return attrs if attrs.get("type") else None

    def _next_evidence_id(self, target_db):
        prefix = {
            "dict": "DICT",
            "annals": "ANNALS",
            "memorials": "MEMORIALS"
        }.get(target_db, target_db.upper())
        return f"{prefix}-{len(self.memory.get(target_db, [])) + 1}"

    def run(self, user_query, max_steps=8):
        self.reset_state()
        self.messages.append({"role": "user", "content": f"历史问题：{user_query}"})
        print(f"\n🎯 [任务启动] {user_query}\n" + "=" * 60)

        for step_idx in range(max_steps):
            # LLM 调用（带空回复重试）
            ai_reply = ""
            for retry in range(2):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=self.messages,
                    temperature=0.2
                )
                ai_reply = (response.choices[0].message.content or "").strip()
                if ai_reply:
                    break
                # 空回复：追加提醒后重试
                print(f"{Colors.YELLOW}⚠️ AI 返回空内容 (第{retry+1}次)，重试中...{Colors.ENDC}")
                self.messages.append({
                    "role": "user",
                    "content": "（系统提示：你上一轮回复为空。请根据当前检索进度，输出下一步 action 标签。务必包含有效的 <action> 标签。）"
                })

            if not ai_reply:
                print(f"{Colors.RED}⚠️ AI 连续返回空内容，转入最终报告生成。{Colors.ENDC}")
                self.steps.append({
                    "step": step_idx + 1,
                    "ai_reply": "",
                    "parsed_action": None
                })
                break

            self.messages.append({"role": "assistant", "content": ai_reply})

            print(f"{Colors.GREEN}[主控思考 {step_idx + 1}]:\n{ai_reply}{Colors.ENDC}\n")

            action = self._parse_action(ai_reply)
            self.steps.append({
                "step": step_idx + 1,
                "ai_reply": ai_reply,
                "parsed_action": action
            })

            if not action:
                print(f"{Colors.RED}⚠️ 未检测到合规 action，转入最终报告生成。{Colors.ENDC}")
                break

            action_type = action.get("type")

            if action_type == "finish":
                print("✅ [考据闭环] 进入最终报告生成。")
                break

            if action_type != "search":
                print(f"{Colors.RED}⚠️ 未知 action type={action_type}，转入最终报告生成。{Colors.ENDC}")
                break

            target_db = action.get("target")
            query = action.get("query")
            time_window = action.get("time_window")

            if target_db not in {"dict", "annals", "memorials"} or not query:
                print(f"{Colors.RED}⚠️ search action 参数不完整，转入最终报告生成。{Colors.ENDC}")
                break

            search_result = self.kb.hybrid_search(target_db, query, time_window)
            raw_text = search_result["raw_text"]

            evidence_id = self._next_evidence_id(target_db)
            refined_data = RiDAgent.refine(raw_text, target_db, user_query)
            source_refs = format_source_items(target_db, search_result.get("items", []))

            self.memory[target_db].append({
                "query": query,
                "time_window": time_window,
                "refined": refined_data, # 经过 RiD 精炼后的文字
                "items": search_result["items"], # v3 检索器返回的原始 record 列表
                "raw_text": search_result["raw_text"],
                "evidence_id": f"{target_db.upper()}-{len(self.memory[target_db])+1}"
            })

            # 只有当两边都有证据时才做一次互证；避免反复消耗 token
            if (
                self.memory["annals"]
                and self.memory["memorials"]
                and not self.memory["conflict_report"]
            ):
                conflict_analysis = ConflictAnalyzerAgent.verify(
                    self.memory["annals"][0]["refined"],
                    self.memory["memorials"][0]["refined"]
                )
                self.memory["conflict_report"] = conflict_analysis
                observation = (
                    f"<observation evidence_id=\"{evidence_id}\">\n"
                    f"【来源条目】\n{source_refs}\n\n"
                    f"【检索精炼结果】\n{refined_data}\n\n"
                    f"【互证分析】\n{conflict_analysis}\n"
                    f"</observation>"
                )
            else:
                observation = (
                    f"<observation evidence_id=\"{evidence_id}\">\n"
                    f"【来源条目】\n{source_refs}\n\n"
                    f"【检索精炼结果】\n{refined_data}\n"
                    f"</observation>"
                )

            self.messages.append({"role": "user", "content": observation})
            print(observation + "\n" + "-" * 50)

        report = FinalReportAgent.compose(user_query, self.memory)
        print(f"\n{Colors.GREEN}📌 最终报告：{Colors.ENDC}\n{report}\n")
        return report

    def run_stream(self, user_query, max_steps=8):
        """流式版本：yield 事件字典，供 SSE 推送到前端。

        事件类型:
        - start:         查询开始
        - thinking:      LLM 正在决策下一步
        - thinking_done: LLM 返回 action 决策
        - searching:     正在检索某数据库
        - search_done:   检索完成（含结果数量）
        - refining:      正在用 RiD Agent 精炼
        - evidence_found: 证据收集完成
        - conflict_analysis / conflict_done: 实录vs档案互证
        - warning:       异常/跳过
        - finish_action: Agent 决定结束检索
        - generating_report: 正在生成最终报告
        - done:          完成（含 answer + references）
        """
        self.reset_state()
        self.messages.append({"role": "user", "content": f"历史问题：{user_query}"})

        db_labels = {"dict": "词典", "annals": "编年实录", "memorials": "档案奏折"}

        yield {
            "type": "start",
            "query": user_query,
            "step": 0,
            "max_steps": max_steps,
            "message": f"🎯 开始处理：{user_query[:50]}{'...' if len(user_query) > 50 else ''}"
        }

        for step_idx in range(max_steps):
            # --- 阶段 1: LLM 决策 ---
            yield {
                "type": "thinking",
                "step": step_idx + 1,
                "message": f"正在分析第 {step_idx + 1} 步检索策略..."
            }

            # LLM 调用（带空回复重试）
            ai_reply = ""
            for retry in range(2):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=self.messages,
                    temperature=0.2
                )
                ai_reply = (response.choices[0].message.content or "").strip()
                if ai_reply:
                    break
                print(f"{Colors.YELLOW}⚠️ [SSE] AI 返回空内容 (第{retry+1}次)，重试中...{Colors.ENDC}")
                self.messages.append({
                    "role": "user",
                    "content": "（系统提示：你上一轮回复为空。请根据当前检索进度，输出下一步 action 标签。务必包含有效的 <action> 标签。）"
                })

            if not ai_reply:
                print(f"{Colors.RED}⚠️ [SSE] AI 连续返回空内容，转入最终报告生成。{Colors.ENDC}")
                self.steps.append({
                    "step": step_idx + 1,
                    "ai_reply": "",
                    "parsed_action": None
                })
                yield {
                    "type": "warning",
                    "step": step_idx + 1,
                    "message": "AI 连续返回空内容，转入最终报告生成。"
                }
                break

            self.messages.append({"role": "assistant", "content": ai_reply})

            print(f"{Colors.GREEN}[主控思考 {step_idx + 1}]:\n{ai_reply}{Colors.ENDC}\n")

            action = self._parse_action(ai_reply)
            self.steps.append({
                "step": step_idx + 1,
                "ai_reply": ai_reply,
                "parsed_action": action
            })

            yield {
                "type": "thinking_done",
                "step": step_idx + 1,
                "content": ai_reply,
                "action": action
            }

            # --- 阶段 2: 解析 action ---
            if not action:
                yield {
                    "type": "warning",
                    "step": step_idx + 1,
                    "message": "未检测到合规 action，转入最终报告生成。"
                }
                print(f"{Colors.RED}⚠️ 未检测到合规 action，转入最终报告生成。{Colors.ENDC}")
                break

            action_type = action.get("type")

            if action_type == "finish":
                yield {
                    "type": "finish_action",
                    "step": step_idx + 1,
                    "message": "证据收集完毕，进入最终报告生成。"
                }
                print("✅ [考据闭环] 进入最终报告生成。")
                break

            if action_type != "search":
                yield {
                    "type": "warning",
                    "step": step_idx + 1,
                    "message": f"未知 action type={action_type}，转入最终报告生成。"
                }
                print(f"{Colors.RED}⚠️ 未知 action type={action_type}，转入最终报告生成。{Colors.ENDC}")
                break

            target_db = action.get("target")
            query_s = action.get("query")
            time_window = action.get("time_window")

            if target_db not in {"dict", "annals", "memorials"} or not query_s:
                yield {
                    "type": "warning",
                    "step": step_idx + 1,
                    "message": "search action 参数不完整，转入最终报告生成。"
                }
                print(f"{Colors.RED}⚠️ search action 参数不完整，转入最终报告生成。{Colors.ENDC}")
                break

            db_label = db_labels.get(target_db, target_db)

            # --- 阶段 3: 执行检索 ---
            yield {
                "type": "searching",
                "step": step_idx + 1,
                "target": target_db,
                "target_label": db_label,
                "query": query_s,
                "time_window": time_window,
                "message": f"正在检索 {db_label} 库：「{query_s}」"
            }

            search_result = self.kb.hybrid_search(target_db, query_s, time_window)
            raw_text = search_result["raw_text"]
            items = search_result.get("items", [])

            yield {
                "type": "search_done",
                "step": step_idx + 1,
                "target": target_db,
                "target_label": db_label,
                "result_count": len(items),
                "message": f"{db_label}检索完成 — 命中 {len(items)} 条结果"
            }

            evidence_id = self._next_evidence_id(target_db)

            # --- 阶段 4: RiD 精炼 ---
            yield {
                "type": "refining",
                "step": step_idx + 1,
                "target": target_db,
                "target_label": db_label,
                "evidence_id": evidence_id,
                "message": f"正在对{db_label}检索结果进行结构化提炼..."
            }

            refined_data = RiDAgent.refine(raw_text, target_db, user_query)
            source_refs = format_source_items(target_db, items)

            self.memory[target_db].append({
                "query": query_s,
                "time_window": time_window,
                "refined": refined_data,
                "items": items,
                "raw_text": raw_text,
                "evidence_id": evidence_id
            })

            yield {
                "type": "evidence_found",
                "step": step_idx + 1,
                "evidence_id": evidence_id,
                "target": target_db,
                "target_label": db_label,
                "refined_summary": refined_data[:200] + ("..." if len(refined_data) > 200 else ""),
                "result_count": len(items),
                "message": f"证据 {evidence_id} 已收集（{db_label}）"
            }

            # --- 阶段 5: 交叉互证（仅首次） ---
            if (
                self.memory["annals"]
                and self.memory["memorials"]
                and not self.memory["conflict_report"]
            ):
                yield {
                    "type": "conflict_analysis",
                    "step": step_idx + 1,
                    "message": "正在对比《实录》定论与《档案》私叙的底层差异..."
                }
                conflict_analysis = ConflictAnalyzerAgent.verify(
                    self.memory["annals"][0]["refined"],
                    self.memory["memorials"][0]["refined"]
                )
                self.memory["conflict_report"] = conflict_analysis
                yield {
                    "type": "conflict_done",
                    "step": step_idx + 1,
                    "has_conflict": "未发现明确矛盾" not in conflict_analysis,
                    "message": "发现矛盾" if ("未发现明确矛盾" not in conflict_analysis) else "未发现明确矛盾"
                }
                observation = (
                    f"<observation evidence_id=\"{evidence_id}\">\n"
                    f"【来源条目】\n{source_refs}\n\n"
                    f"【检索精炼结果】\n{refined_data}\n\n"
                    f"【互证分析】\n{conflict_analysis}\n"
                    f"</observation>"
                )
            else:
                observation = (
                    f"<observation evidence_id=\"{evidence_id}\">\n"
                    f"【来源条目】\n{source_refs}\n\n"
                    f"【检索精炼结果】\n{refined_data}\n"
                    f"</observation>"
                )

            self.messages.append({"role": "user", "content": observation})
            print(observation + "\n" + "-" * 50)

        # --- 阶段 6: 生成最终报告 ---
        yield {
            "type": "generating_report",
            "message": "正在综合所有证据，生成考据报告..."
        }

        report = FinalReportAgent.compose(user_query, self.memory)
        print(f"\n{Colors.GREEN}📌 最终报告：{Colors.ENDC}\n{report}\n")

        result = parse_report_and_build_response(report, self.memory)

        yield {
            "type": "done",
            "answer": result["answer"],
            "references": result["references"]
        }


def interactive_loop():
    logger = QueryLogger()
    system = MasterReasoner()

    print("\n清史智能体已启动。输入问题开始检索；输入 q / quit / exit / 退出 结束。")
    while True:
        try:
            user_query = input("\n请输入清史问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_query:
            continue
        if user_query.lower() in {"q", "quit", "exit", "退出"}:
            print("已退出。")
            break

        try:
            report = system.run(user_query)
            logger.append(user_query, report, system.memory, system.steps)
        except Exception as exc:
            print(f"{Colors.RED}❌ 本次查询失败：{exc}{Colors.ENDC}")
            # 失败也写入日志，方便排查
            logger.append(
                user_query,
                f"ERROR: {repr(exc)}",
                getattr(system, "memory", {}),
                getattr(system, "steps", [])
            )


if __name__ == "__main__":
    interactive_loop()
