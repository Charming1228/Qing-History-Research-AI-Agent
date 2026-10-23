"""
清史档案检索 + LLM 问答系统 v2
改进：
  1. 多字段加权 BM25（题名权重×3, 内容提要×1, 责任者×1）
  2. 清代术语查询扩展（地名/制度/军事/人名 自动扩展）
  3. LLM 语义重排序（BM25 Top-20 → LLM Rerank → Top-5）
"""

import json
import os
import pickle
import re
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from opencc import OpenCC

_cc = OpenCC('t2s')

# ============================================================
# 清代专有名词词典（确保 jieba 正确切分官职/制度/人名等术语）
# ============================================================
_QING_TERMS = [
    "掌印郎中", "员外郎", "笔帖式", "给事中", "协办大学士", "军机大臣",
    "总督", "巡抚", "布政使", "按察使", "提督", "总兵", "参将", "游击",
    "都统", "副都统", "参领", "佐领", "领催", "内阁学士", "侍读学士",
    "监察御史", "道员", "知府", "知州", "知县", "同知", "通判",
    "钦差大臣", "办事大臣", "领队大臣", "参赞大臣",
    "军机处", "内务府", "理藩院", "都察院", "大理寺", "太常寺",
    "光禄寺", "太仆寺", "鸿胪寺", "国子监", "钦天监", "宗人府",
    "吏部", "户部", "礼部", "兵部", "刑部", "工部",
    "总理衙门", "总理各国事务衙门", "外务部",
    "稽察房", "典籍厅", "档房",
    "八旗", "满洲八旗", "蒙古八旗", "汉军八旗", "绿营",
    "科举", "乡试", "会试", "殿试", "引见", "议叙", "议处",
    "丁忧", "起复", "休致", "原品休致",
    "伊犁", "乌里雅苏台", "盛京", "乌鲁木齐", "喀什噶尔",
    "福康安", "和珅", "曾国藩", "左宗棠", "李鸿章", "张之洞",
    "刘铭传", "沈葆桢", "林则徐", "琦善", "僧格林沁", "袁世凯",
    "太平天国", "义和团", "甲午战争", "鸦片战争", "洋务运动",
    "百日维新", "戊戌变法", "辛亥革命", "捻军",
    # 复合词（防止 jieba 切碎，提升 BM25 匹配精度）
    "光绪皇帝", "光绪帝", "慈禧太后", "慈禧",
    "咸丰皇帝", "同治皇帝", "道光皇帝", "嘉庆皇帝",
    "乾隆皇帝", "雍正皇帝", "康熙皇帝", "顺治皇帝",
    "北洋水师", "北洋海军", "南洋水师",
    "虎门销烟", "安庆之战", "马关条约", "南京条约", "北京条约",
    "瀛台", "圆明园", "颐和园",
]
for _term in _QING_TERMS:
    jieba.add_word(_term)

# ============================================================
# 查询扩展映射（同义词/下位词/关联词）
# ============================================================
_QING_QUERY_EXPANSION = {
    # ── 地名 ──
    "台湾": ["台湾府", "台北府", "基隆", "澎湖", "淡水", "台南", "凤山", "嘉义", "彰化"],
    "新疆": ["伊犁", "乌鲁木齐", "喀什噶尔", "回疆", "塔尔巴哈台", "阿克苏", "叶尔羌", "和阗"],
    "西藏": ["藏", "前藏", "后藏", "拉萨", "达赖喇嘛", "班禅", "驻藏大臣"],
    "蒙古": ["喀尔喀", "乌里雅苏台", "科布多", "库伦", "盟旗", "扎萨克"],
    "东北": ["盛京", "吉林", "黑龙江", "宁古塔", "奉天", "珲春", "三姓"],
    "盛京": ["奉天", "盛京将军", "盛京五部"],
    "江南": ["江宁", "苏州", "扬州", "镇江", "上海", "松江"],
    "广东": ["广州", "两广", "广州府", "澳门", "香港"],
    "福建": ["福州", "厦门", "台湾府"],
    "云南": ["滇", "昆明", "腾越"],
    "贵州": ["黔", "贵阳"],
    "四川": ["川", "成都", "重庆"],
    "陕甘": ["陕西", "甘肃", "西安", "兰州"],
    "直隶": ["直隶", "顺天", "保定", "天津"],

    # ── 军事/防务 ──
    "海防": ["水师", "炮台", "海疆", "海口", "海岸", "巡洋", "防海", "巡防"],
    "防御": ["防务", "守御", "设防", "驻防", "防守", "堵御", "防堵"],
    "军务": ["军事", "用兵", "出征", "战事", "征剿", "进剿", "剿办"],
    "练兵": ["练军", "新军", "武备", "操练", "营伍", "团练"],
    "水师": ["水师", "水军", "战船", "师船", "巡洋", "水操", "水勇"],
    "炮台": ["炮台", "炮位", "大炮", "炮械", "演炮"],
    "营制": ["营制", "汛地", "兵额", "兵丁", "马步", "军饷"],

    # ── 制度/官职 ──
    "军机": ["军机处", "军机大臣", "军机章京", "军机行走", "枢垣"],
    "科举": ["乡试", "会试", "殿试", "进士", "举人", "贡士", "科场"],
    "吏治": ["铨选", "引见", "议叙", "甄别", "考察", "大计"],
    "丁忧": ["丁忧", "守制", "起复", "终养", "省亲"],
    "捐纳": ["捐输", "捐官", "报捐", "捐职"],

    # ── 财政 ──
    "赔款": ["摊赔", "赔补", "赔缴", "分赔", "代赔", "赔项", "偿还"],
    "钱粮": ["地丁", "漕粮", "耗羡", "正项", "杂赋", "赋税"],
    "厘金": ["厘金", "抽厘", "厘捐", "卡局", "厘局"],
    "关税": ["关税", "常关", "洋关", "税则", "征税"],

    # ── 外交/通商 ──
    "鸦片": ["鸦片烟", "禁烟", "洋药", "烟土", "戒烟", "烟案"],
    "通商": ["互市", "贸易", "口岸", "海关", "通商口岸", "开埠"],
    "教案": ["教案", "传教士", "教堂", "教民", "民教", "天主堂"],
    "条约": ["条约", "约章", "章程", "条款"],

    # ── 洋务/工业 ──
    "洋务": ["机器局", "制造局", "船政", "电报", "铁路", "矿务", "织布局", "招商局"],
    "煤矿": ["煤矿", "煤窑", "煤斤", "挖煤", "开矿", "矿务"],
    "铁路": ["铁路", "铁道", "火车", "铁轨"],
    "电报": ["电报", "电线", "电局", "报房"],

    # ── 灾荒/河工 ──
    "灾荒": ["水灾", "旱灾", "蝗灾", "地震", "赈济", "蠲免", "抚恤", "灾民"],
    "水灾": ["水灾", "水患", "决口", "漫口", "河工", "修堤", "防汛"],
    "河工": ["河工", "治河", "修堤", "堵口", "挑浚"],

    # ── 人事 ──
    "休致": ["休致", "原品休致", "勒令休致", "告病", "告老"],
    "议处": ["议处", "革职", "降调", "罚俸", "处分", "严议"],
    "引见": ["引见", "带领引见", "验看"],

    # ── 人名别称 ──
    "曾国藩": ["曾文正", "曾涤生", "曾中堂", "曾帅"],
    "李鸿章": ["李中堂", "李傅相", "李少荃", "李合肥"],
    "左宗棠": ["左文襄", "左季高"],
    "林则徐": ["林文忠", "林少穆", "林制军"],
    "张之洞": ["张香涛", "张南皮"],
    "刘铭传": ["刘省三", "刘壮肃"],
    "沈葆桢": ["沈幼丹", "沈文肃"],
    "袁世凯": ["袁项城", "袁慰亭"],
    "和珅": ["和致斋", "和相"],
}


# 查询扩展优先级：人名 > 地名 > 制度/机构 > 宽泛类别
_EXPANSION_PRIORITY = {
    # ── 人名（最高优先级）──
    "曾国藩": 1, "李鸿章": 1, "左宗棠": 1, "林则徐": 1,
    "张之洞": 1, "刘铭传": 1, "沈葆桢": 1, "袁世凯": 1, "和珅": 1,
    # ── 地名 ──
    "台湾": 2, "新疆": 2, "西藏": 2, "蒙古": 2, "东北": 2,
    "盛京": 2, "江南": 2, "广东": 2, "福建": 2, "云南": 2,
    "贵州": 2, "四川": 2, "陕甘": 2, "直隶": 2,
    # ── 制度/官职/机构 ──
    "军机": 3, "科举": 3, "吏治": 3, "丁忧": 3, "捐纳": 3,
    # ── 宽泛类别（最低优先级）──
    "海防": 4, "防御": 4, "军务": 4, "练兵": 4, "水师": 4,
    "炮台": 4, "营制": 4, "赔款": 4, "钱粮": 4, "厘金": 4,
    "关税": 4, "鸦片": 4, "通商": 4, "教案": 4, "条约": 4,
    "洋务": 4, "煤矿": 4, "铁路": 4, "电报": 4,
    "灾荒": 4, "水灾": 4, "河工": 4,
    "休致": 4, "议处": 4, "引见": 4,
}

# BM25 停用词（自然语言问句中的虚词/疑问词，不应参与检索匹配）
_BM25_STOP_WORDS = {
    "的", "是", "还是", "了", "吗", "呢", "吧", "啊", "呀",
    "什么", "怎么", "怎样", "如何", "为什么", "为何",
    "多少", "几个", "哪", "哪个", "哪些", "谁",
    "不", "没", "很", "都", "就", "也", "会", "要", "能", "可以",
    "在", "和", "与", "或", "及", "之", "等", "其", "这", "那",
    "有", "被", "把", "从", "到", "对", "向", "将", "以",
    "一个", "一种", "这个", "那个", "中", "后", "前", "上", "下",
    "是否", "有无", "可否", "怎么办", "怎样办",
}

MAX_EXPANSION_TERMS = 5       # 扩展词总数硬限制
MIN_QUERY_LEN_FOR_EXPANSION = 10  # 超过此长度的查询不扩展（自然语言问题信号已足够）


class QingQueryExpander:
    """清代查询扩展器：智能扩展（限长、限数、分优先级）"""

    def expand(self, query):
        """返回 (扩展后的查询字符串, 命中的扩展词列表)"""
        query_simp = _cc.convert(query)

        # 长查询不扩展：自然语言问句已有足够关键词
        if len(query_simp) > MIN_QUERY_LEN_FOR_EXPANSION:
            return query_simp, []

        scored = []  # (priority, term)

        # 1. 词典查找（按优先级打分）
        for key, expansions in _QING_QUERY_EXPANSION.items():
            if key in query_simp:
                prio = _EXPANSION_PRIORITY.get(key, 4)
                for term in expansions:
                    if term not in query_simp:
                        scored.append((prio, term))

        # 2. 清史术语交叉匹配
        query_words = set(jieba.lcut(query_simp))
        meaningful = {w for w in query_words if len(w) >= 2}
        for term in _QING_TERMS:
            if term in query_simp:
                continue
            term_words = set(jieba.lcut(term))
            term_meaningful = {w for w in term_words if len(w) >= 2}
            if meaningful & term_meaningful:
                prio = _EXPANSION_PRIORITY.get(term, 4)
                scored.append((prio, term))

        if not scored:
            return query_simp, []

        # 按优先级排序，同优先级保留先出现的（词典顺序）
        scored.sort(key=lambda x: x[0])
        unique = list(dict.fromkeys(t for _, t in scored))
        top_terms = unique[:MAX_EXPANSION_TERMS]

        expanded_query = f"{query_simp} {' '.join(top_terms)}"
        return expanded_query, top_terms


_expander = QingQueryExpander()

# ============================================================
# LLM API 配置
# ============================================================
LLM_BASE_URL = os.environ.get("MOARK_BASE_URL", "https://api.moark.com/v1")
LLM_API_KEY = os.environ.get("MOARK_API_KEY", "")
LLM_MODEL_NAME = os.environ.get("MOARK_MODEL", "DeepSeek-V4-Pro")

USE_LLM = False       # LLM 回答生成
USE_RERANK = True     # LLM 重排序


# ============================================================
class ArchiveSearchEngine:
    """档案搜索引擎（v2：多字段加权 BM25）"""

    def __init__(self, data_dir, cache_dir=None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or data_dir
        self.cache_path = os.path.join(self.cache_dir, ".archive_bm25_cache_v2.pkl")
        self.records = []
        self.tokenized_corpus = []
        self.bm25 = None

    def load_data(self):
        json_files = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith('.json')
        )
        all_records = []
        for fname in json_files:
            with open(os.path.join(self.data_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
            dynasty = os.path.splitext(fname)[0]
            for item in data:
                item['来源朝代'] = dynasty
                all_records.append(item)
        self.records = all_records
        print(f"已加载 {len(self.records)} 条档案记录 ({len(json_files)} 个朝代)")
        return self

    def _build_doc_text(self, record):
        """构建加权文档文本：题名重复3遍提升标题命中权重"""
        title = _cc.convert(record.get('題名', '') or '')
        abstract = _cc.convert(record.get('內容提要', '') or '')
        author = _cc.convert(record.get('責任者', '') or '')

        parts = []
        for _ in range(3):
            if title:
                parts.append(title)
        if abstract:
            parts.append(abstract)
        if author:
            parts.append(author)
        return " ".join(parts)

    def build_index(self, force_rebuild=False):
        if not force_rebuild and self._load_cache():
            return self

        print("正在构建 BM25 索引（多字段加权: 题名×3 内容提要×1 责任者×1）...")
        self.tokenized_corpus = []
        for i, rec in enumerate(self.records):
            tokens = list(jieba.cut(self._build_doc_text(rec)))
            self.tokenized_corpus.append(tokens)
            if (i + 1) % 50000 == 0:
                print(f"  已分词 {i + 1}/{len(self.records)} 条...")

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_cache()
        print(f"索引构建完成，共 {len(self.records)} 条记录。")
        return self

    def _save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'records': self.records,
                'tokenized_corpus': self.tokenized_corpus,
                'bm25': self.bm25,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  索引已缓存至 .archive_bm25_cache_v2.pkl")

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return False
        cache_mtime = os.path.getmtime(self.cache_path)
        for fname in os.listdir(self.data_dir):
            if fname.endswith('.json'):
                if os.path.getmtime(os.path.join(self.data_dir, fname)) > cache_mtime:
                    print("  检测到 JSON 更新，重建索引。")
                    return False
        print("从缓存加载索引...")
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        self.records = data['records']
        self.tokenized_corpus = data['tokenized_corpus']
        self.bm25 = data['bm25']
        print(f"已加载 {len(self.records)} 条档案记录。")
        return True

    def search(self, query, top_k=20, use_expansion=True):
        """BM25 检索（含查询扩展）。top_k 默认 20 供 reranker 使用。"""
        if self.bm25 is None:
            raise RuntimeError("索引尚未构建，请先调用 build_index()")

        if use_expansion:
            expanded_query, hit_terms = _expander.expand(query)
            if hit_terms:
                print(f"  查询扩展 (+{len(hit_terms)} 词): {', '.join(hit_terms[:8])}"
                      f"{'...' if len(hit_terms) > 8 else ''}")
        else:
            expanded_query = _cc.convert(query)

        query_tokens = [t for t in jieba.cut(expanded_query)
                        if t.strip() and t not in _BM25_STOP_WORDS]
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            rec = self.records[idx]
            results.append({
                'index': int(idx),
                'score': round(float(scores[idx]), 2),
                '题名': _cc.convert(rec.get('題名', '') or ''),
                '内容提要': _cc.convert(rec.get('內容提要', '') or ''),
                '年号': _cc.convert(rec.get('年号', '') or ''),
                '责任者': _cc.convert(rec.get('責任者', '') or ''),
                '责任时间(中历)': _cc.convert(rec.get('責任時間(中曆)', '') or ''),
                '来源朝代': _cc.convert(rec.get('来源朝代', '') or ''),
            })
        return results

    def format_results(self, results):
        if not results:
            return "（未找到相关档案记录）"
        lines = []
        for r in results:
            rank = r.get('rank', '?')
            lines.append(
                f"[{rank}] 题名：{r['题名']}\n"
                f"   内容提要：{r['内容提要'][:200]}\n"
                f"   年号：{r['年号']} | 责任者：{r['责任者']} | "
                f"责任时间：{r['责任时间(中历)']} | 来源：{r['来源朝代']}朝档案"
            )
        return "\n\n".join(lines)


# ============================================================
class LLMReranker:
    """LLM 语义重排序器：BM25 粗排 Top-20 → LLM 精排 Top-5"""

    def __init__(self, base_url, api_key, model_name):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_name

    def rerank(self, query, candidates, top_k=5):
        if len(candidates) <= top_k:
            for i, c in enumerate(candidates):
                c['rank'] = i + 1
            return candidates

        # 构建候选列表
        items = []
        for i, c in enumerate(candidates):
            items.append(
                f"[{i}] 题名：{c['题名']}\n"
                f"    内容提要：{c['内容提要'][:150]}\n"
                f"    责任者：{c['责任者']} | 朝代：{c['来源朝代']}"
            )
        candidates_str = "\n\n".join(items)

        system_prompt = (
            "你是一个清史档案检索专家。给定用户查询和一批BM25初检结果，"
            "请按语义相关性重新排序，选出最相关的若干条。\n"
            "判断标准：\n"
            "1. 题名或内容是否直接涉及用户查询的核心主题\n"
            "2. 人物、地点、事件是否与查询高度相关\n"
            "3. 排除仅关键词偶然匹配但主题无关的记录\n"
            f"严格返回JSON格式：{{\"ranked_indices\": [3, 7, 12, 0, 15]}}"
            f"（选出最相关的{top_k}条，按相关性降序排列）"
        )

        user_prompt = (
            f"【用户查询】{query}\n\n"
            f"【候选档案（共{len(candidates)}条）】\n{candidates_str}\n\n"
            f"请选出最相关的{top_k}条，返回JSON。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            content = response.choices[0].message.content

            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                ranking = json.loads(json_match.group())
                ranked_indices = ranking.get("ranked_indices", [])

                reranked = []
                for idx in ranked_indices:
                    if 0 <= idx < len(candidates):
                        candidates[idx]['rank'] = len(reranked) + 1
                        reranked.append(candidates[idx])

                if reranked:
                    return reranked[:top_k]
        except Exception as e:
            print(f"  [重排序失败，回退到 BM25 原始排序] {type(e).__name__}: {e}")

        # 回退：BM25 原始排序
        for i, c in enumerate(candidates[:top_k]):
            c['rank'] = i + 1
        return candidates[:top_k]


# ============================================================
class LLMAnswerGenerator:
    """调用 LLM 基于检索结果生成带出处的自然语言回答"""

    def __init__(self, base_url, api_key, model_name):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_name

    def generate(self, query, search_results, engine):
        sources_text = engine.format_results(search_results)

        system_prompt = (
            "你是一位清史研究助手。请根据下方提供的清代档案资料，"
            "回答用户提出的问题。\n"
            "要求：\n"
            "1. 用简洁清晰的中文回答\n"
            "2. 引用具体档案时，在正文中使用 [1][2] 等编号标注出处\n"
            "3. 如果档案资料不足，请诚实说明，不要编造\n"
            "4. 回答末尾附上「参考文献」列出引用的档案题名"
        )

        user_prompt = (
            f"【用户问题】\n{query}\n\n"
            f"【相关档案资料】\n{sources_text}\n\n"
            f"请基于以上档案资料回答问题。"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        return response.choices[0].message.content


# ============================================================
# CLI
# ============================================================
def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    DATA_DIR = os.path.join(os.path.dirname(__file__), "数据基础", "档案_过滤")

    print("=" * 60)
    print("清史档案检索系统 v2")
    print("管线：查询扩展 → BM25粗排(Top-20) → LLM重排(Top-5)")
    print("=" * 60)

    engine = ArchiveSearchEngine(DATA_DIR)
    engine.load_data()
    engine.build_index()

    llm = None
    if USE_LLM:
        print(f"\nLLM 回答生成: {LLM_MODEL_NAME}")
        llm = LLMAnswerGenerator(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME)
    else:
        print("\nLLM 回答生成: 未启用 (设 USE_LLM=True 开启)")

    reranker = None
    if USE_RERANK:
        print(f"LLM 重排序:   {LLM_MODEL_NAME}")
        reranker = LLMReranker(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME)
    else:
        print("LLM 重排序:   未启用 (设 USE_RERANK=True 开启)")

    print("\n" + "=" * 60)
    print("输入清史研究问题（q 退出）：")
    print("=" * 60)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ('q', 'quit', 'exit'):
            print("再见！")
            break

        # ── 检索流水线 ──
        print("  [1/3] BM25 粗排 (Top-20)...")
        candidates = engine.search(query, top_k=20, use_expansion=True)

        if not candidates:
            print("\n（未找到相关档案记录）\n")
            continue

        # 重排序
        if reranker and USE_RERANK:
            print(f"  [2/3] LLM 语义重排 (20→5)...")
            results = reranker.rerank(query, candidates, top_k=5)
        else:
            for i, c in enumerate(candidates[:5]):
                c['rank'] = i + 1
            results = candidates[:5]

        # 展示
        if llm and USE_LLM:
            print(f"  [3/3] 生成回答...")
            answer = llm.generate(query, results, engine)
            print("\n" + "─" * 50)
            print(answer)
            print("─" * 50)
            print("\n【参考档案出处】")
            print(engine.format_results(results))
        else:
            print(f"  [2/2] 完成，展示 Top-{len(results)} 结果")
            print("\n【检索结果】")
            print(engine.format_results(results))
            print()

    print("系统已退出。")


if __name__ == "__main__":
    main()
