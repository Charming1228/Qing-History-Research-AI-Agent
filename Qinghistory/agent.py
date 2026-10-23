import re
import json
import sys
import os
from config import get_llm_client, get_llm_model

# ==========================================
# 强制离线
# ==========================================
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import jieba
jieba.dt.tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive_retrieval/cache")

# ==========================================
# 添加子模块路径
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 档案检索路径
archive_dir = os.path.join(current_dir, "archive_retrieval")
sys.path.append(archive_dir)

# 2. 词典检索路径
dict_dir = os.path.join(current_dir, "dict_retrieval", "dict_search")
sys.path.append(dict_dir)

# 3. 编年检索路径
annal_dir = os.path.join(current_dir, "annal_retrieval")
sys.path.append(annal_dir)

# ==========================================
# 0. 全局配置与初始化
# ==========================================
client = get_llm_client()
MODEL_NAME = get_llm_model("deepseek-chat")

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

# ==========================================
# 1. 检索与底座工具 (整合档案、词典、编年模块)
# ==========================================
from archive_retriever import ArchiveRetriever
from search_dict_fusion_core import DictFusionSearcher


from a import SmartHistoricalRetriever

class KnowledgeBase:
    def __init__(self):
        print(f"{Colors.BLUE}▶ [系统初始化] 正在加载各类检索缓存...{Colors.ENDC}")
        
        # ------------------------------------------------
        # 1. 初始化档案/奏折检索器
        # ------------------------------------------------
        print(f"  └-- 正在加载 ArchiveRetriever (档案)...")
        os.chdir(archive_dir) 
        self.archive_retriever = ArchiveRetriever()
        self.archive_retriever.load_or_build()
        os.chdir(current_dir) 
        
        # ------------------------------------------------
        # 2. 初始化词典检索器
        # ------------------------------------------------
        print(f"  └-- 正在加载 DictFusionSearcher (词典)...")
        os.chdir(dict_dir) 
        self.dict_searcher = DictFusionSearcher()
        os.chdir(current_dir) 
        
        # ------------------------------------------------
        # 3. 初始化编年实录检索器
        # ------------------------------------------------
        print(f"  └-- 正在加载 SmartHistoricalRetriever (编年实录)...")
        os.chdir(annal_dir) 
        # 构建文件列表路径（根据你提供的路径）
        annal_files = [
            os.path.join(annal_dir, "data/光绪朝结构化数据摘要.json"),
            os.path.join(annal_dir, "data/乾隆朝结构化数据摘要_aligned.json"),
            os.path.join(annal_dir, "data/康熙朝结构化数据摘要_aligned.json")
        ]
        self.annals_retriever = SmartHistoricalRetriever(file_paths=annal_files)
        os.chdir(current_dir)
        
        print(f"{Colors.GREEN}✅ 所有检索缓存加载完毕！{Colors.ENDC}\n" + "="*50)

    def hybrid_search(self, db_type, query, time_window=None):
        """混合检索分发中枢"""
        print(f"{Colors.BLUE}▶ [底层检索] 目标库: {db_type} | Query: '{query}' | 时间窗: {time_window}{Colors.ENDC}")
        
        if db_type == "memorials":
            raw_results = self.archive_retriever.search(query, top_k=5)
            if not raw_results:
                return "无相关档案记载。"
                
            formatted_docs = []
            for item in raw_results:
                record = item.get("record", {})
                doc_str = (
                    f"【{record.get('题名', '无题')}】\n"
                    f"时间：{record.get('责任时间(中历)', '未知')} | "
                    f"责任者：{record.get('责任者', '未知')} | "
                    f"朝代：{record.get('来源朝代', '未知')}\n"
                    f"内容提要：{record.get('内容提要', '无摘要')}"
                )
                formatted_docs.append(doc_str)
            return "\n\n---\n\n".join(formatted_docs)
            
        elif db_type == "dict":
            os.chdir(dict_dir)
            raw_results = self.dict_searcher.search(query)
            os.chdir(current_dir)
            
            if not raw_results or not isinstance(raw_results, dict):
                return "词典中无相关词条解释。"
            
            results_list = raw_results.get("results", [])
            if not results_list:
                return f"词典中未检索到 '{query}' 的相关解释。"

            formatted_docs = []
            for item in results_list:
                term = item.get("term", item.get("条目", "未知词条"))
                definition = item.get("definition", item.get("解释", "无解释"))
                doc_str = f"【词条】：{term}\n【释义】：{definition}"
                formatted_docs.append(doc_str)
            return "\n\n".join(formatted_docs)
            
        elif db_type == "annals":
            raw_results = self.annals_retriever.search(query, max_k=8, min_score=0.60, drop_threshold=0.1)
            
            if not raw_results:
                return f"实录编年中未检索到关于 '{query}' 的高度相关记载。"
            
            formatted_docs = []
            for item in raw_results:
                d = item['data']
                era = d.get('era_source', '未知朝代')
                date = d.get('date', '未知日期')
                summary = d.get('refined_data', {}).get('micro_summary', '无摘要')
                
                doc_str = f"【实录编年】{era}朝 | 日期：{date}\n记载摘要：{summary}"
                formatted_docs.append(doc_str)
            return "\n\n---\n\n".join(formatted_docs)
            
        return "数据库类型错误。"

kb = KnowledgeBase()

# ==========================================
# 2. RiD 史料精炼员 Agent
# ==========================================
class RiDAgent:
    @staticmethod
    def refine(raw_text, source_type):
        print(f"{Colors.YELLOW}▶ [RiD Agent] 正在对 {source_type} 检索结果进行结构化提炼...{Colors.ENDC}")
        if source_type == "dict":
            prompt_context = "你是一个词典摘要员。请阅读以下词典条目，提取出与当前考据问题最相关的定义和制度背景，不超过150字。"
        else:
            prompt_context = "你是一个史料精炼员。请阅读以下从数据库中检索出的多条史料原文，提取出与当前考据最相关的核心事实。请综合写成一段不超过300字的白话文报告，必须包含：时间、核心人物/机构、客观发生的事实、潜在动机(如有)。"

        prompt = f"{prompt_context}\n原始资料如下：\n{raw_text}"
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

# ==========================================
# 3. 证据冲突分析员 Agent
# ==========================================
class ConflictAnalyzerAgent:
    @staticmethod
    def verify(annals_evidence, memorials_evidence):
        if not annals_evidence or not memorials_evidence:
            return "证据不足，无法进行交叉互证。"
            
        print(f"{Colors.RED}▶ [冲突分析 Agent] 正在对比《实录》定论与《档案》私叙的底层差异...{Colors.ENDC}")
        prompt = f"""你是一个历史考据专家。请对比以下两份史料，指出官方定论（实录）与实际博弈（档案/奏折）之间的矛盾或差异，并解释其背后的历史政治意义。
实录证据：{annals_evidence}
档案证据：{memorials_evidence}"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.3
        )
        return response.choices[0].message.content

# ==========================================
# 4. 主控与状态管理 (Working Memory)
# ==========================================
class MasterReasoner:
    def __init__(self):
        self.memory = {
            "dict": [],
            "annals": [],
            "memorials":[],
            "conflict_report": ""
        }
        self.messages =[{"role": "system", "content": self._get_system_prompt()}]

    def _get_system_prompt(self):
        return """你是清史考据主控系统。你需要通过严谨的多次跨库检索来回答复杂问题。
你有三种工具，请以特定的 XML 格式输出你的规划（注意：不要生成任何 markdown 代码块标记，直接输出 XML）：
1. 查名词、体制：<action type="search" target="dict" query="关键词" />
2. 查官方定论、时间节点：<action type="search" target="annals" query="关键词" time_window="年份" />
3. 查内幕细节、密折博弈：<action type="search" target="memorials" query="关键词" time_window="年份" />

规则：
- 每次回复只允许执行一个动作！
- 获得资料后必须将其纳入你的推理。不要急于下结论，优先把三个库都至少查一遍。
- 如果确认收集完了所有需要的多维度证据，请输出：<action type="finish" />，并在下一段给出带有 [1][2] 脚注的最终考据报告。"""

    def run(self, user_query, max_steps=8):
        self.messages.append({"role": "user", "content": f"历史谜题：{user_query}"})
        print(f"\n🎯 [任务启动] {user_query}\n" + "="*60)

        for step in range(max_steps):
            response = client.chat.completions.create(model=MODEL_NAME, messages=self.messages, temperature=0.2)
            ai_reply = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": ai_reply})
            
            print(f"{Colors.GREEN}[主控思考 {step+1}]:\n{ai_reply}{Colors.ENDC}\n")

            action_match = re.search(r'<action type="(.*?)"(?: target="(.*?)")?(?: query="(.*?)")?(?: time_window="(.*?)")? />', ai_reply)
            
            if not action_match:
                print(f"{Colors.RED}⚠️ 警告：未检测到合规的 action 指令，强制结束。{Colors.ENDC}")
                break 
                
            action_type = action_match.group(1)

            if action_type == "finish":
                print(f"✅ [考据闭环] 输出最终报告。")
                break
                
            elif action_type == "search":
                target_db = action_match.group(2)
                query = action_match.group(3)
                time_window = action_match.group(4)

                # 调用底层检索
                raw_docs = kb.hybrid_search(target_db, query, time_window)
                
                # 调用 RiD 进行去噪
                refined_data = RiDAgent.refine(raw_docs, target_db)
                self.memory[target_db].append(refined_data) 
                
                # 冲突分析检测
                if self.memory["annals"] and self.memory["memorials"] and not self.memory["conflict_report"]:
                    conflict_analysis = ConflictAnalyzerAgent.verify(self.memory["annals"][0], self.memory["memorials"][0])
                    self.memory["conflict_report"] = conflict_analysis
                    observation = f"<observation>【检索精炼结果】\n{refined_data}\n【系统警告：互证发现矛盾】\n{conflict_analysis}</observation>"
                else:
                    observation = f"<observation>【检索精炼结果】\n{refined_data}</observation>"

                self.messages.append({"role": "user", "content": observation})
                print(observation + "\n" + "-"*50)

# ==========================================
# 5. 运行执行
# ==========================================
if __name__ == "__main__":
    system = MasterReasoner()
    system.run("光绪年间刘铭传开办的基隆煤矿，为何最终被朝廷叫停？深层原因是什么？")
