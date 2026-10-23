"""
清史档案混合检索系统 v3
============================================================
管线:
  用户查询 → 查询处理(OpenCC/jieba/实体抽取)
           → 并行双通道:
               BM25 关键词检索 (多字段加权)
               向量语义检索 (BGE-large-zh-v1.5 + FAISS)
           → RRF 融合 → Top-50 候选集
           → 关系加权重排序 (4维: 实体共现30% / 机构隶属30% / 时序邻近20% / 地理关联20%)
           → Top-K 最终结果

支持两种数据格式:
  - "filtered" (默认):  原始文言文档案 (数据基础/档案_过滤)
      BM25: 题名×3 + 内容提要×1 + 责任者×1
      Embedding: 题名 + 内容提要 + 责任者
  - "refined":  微调后的现代文摘要档案 (数据基础/档案refined)
      BM25: modern_summary×2 + source_text×1
      Embedding: modern_summary 仅用现代文摘要
      实体: 预提取实体 + 文本提取实体 融合

主 Agent 接口:
    from archive_retriever import ArchiveRetriever

    # 原始文言文索引 (cache/)
    r_old = ArchiveRetriever()
    r_old.load_or_build()

    # 新 refined 现代文索引 (cache_refined/)
    r_new = ArchiveRetriever(data_format="refined")
    r_new.load_or_build()

    results = r_new.search("乾隆年间福建水师海防", top_k=10)

缓存文件:
    bm25_index.pkl       BM25索引 + 分词语料 + 全部记录
    faiss_index.bin      FAISS向量索引
    doc_entities.pkl     每篇文档的实体标注
    metadata.json        索引元信息
============================================================
"""

import json
import os
import pickle
import re
import time
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Union

import numpy as np
import jieba
from opencc import OpenCC
from rank_bm25 import BM25Okapi

# ============================================================
# 0. 全局初始化
# ============================================================
_cc = OpenCC('t2s')  # 繁体→简体

_PROJECT_ROOT = Path(__file__).parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "数据基础" / "档案_过滤"
_DEFAULT_REFINED_DATA_DIR = _PROJECT_ROOT / "数据基础" / "档案refined"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "cache"
_DEFAULT_REFINED_CACHE_DIR = _PROJECT_ROOT / "cache_refined"
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models"

# 确保 jieba 缓存目录存在
os.makedirs(str(_DEFAULT_CACHE_DIR), exist_ok=True)
os.makedirs(str(_DEFAULT_MODEL_DIR), exist_ok=True)


# ============================================================
# 1. 清代专有名词词典 & jieba 初始化
# ============================================================
_QING_TERMS = [
    # 官职
    "掌印郎中", "员外郎", "笔帖式", "给事中", "协办大学士", "军机大臣",
    "总督", "巡抚", "布政使", "按察使", "提督", "总兵", "参将", "游击",
    "都统", "副都统", "参领", "佐领", "领催", "内阁学士", "侍读学士",
    "监察御史", "道员", "知府", "知州", "知县", "同知", "通判",
    "钦差大臣", "办事大臣", "领队大臣", "参赞大臣",
    # 机构
    "军机处", "内务府", "理藩院", "都察院", "大理寺", "太常寺",
    "光禄寺", "太仆寺", "鸿胪寺", "国子监", "钦天监", "宗人府",
    "吏部", "户部", "礼部", "兵部", "刑部", "工部",
    "总理衙门", "总理各国事务衙门", "外务部",
    "稽察房", "典籍厅", "档房",
    # 制度
    "八旗", "满洲八旗", "蒙古八旗", "汉军八旗", "绿营",
    "科举", "乡试", "会试", "殿试", "引见", "议叙", "议处",
    "丁忧", "起复", "休致", "原品休致",
    # 地名
    "伊犁", "乌里雅苏台", "盛京", "乌鲁木齐", "喀什噶尔",
    # 人名
    "福康安", "和珅", "曾国藩", "左宗棠", "李鸿章", "张之洞",
    "刘铭传", "沈葆桢", "林则徐", "琦善", "僧格林沁", "袁世凯",
    "太平天国", "义和团", "甲午战争", "鸦片战争", "洋务运动",
    "百日维新", "戊戌变法", "辛亥革命", "捻军",
    # 复合词（防 jieba 切碎）
    "光绪皇帝", "光绪帝", "慈禧太后", "慈禧",
    "咸丰皇帝", "同治皇帝", "道光皇帝", "嘉庆皇帝",
    "乾隆皇帝", "雍正皇帝", "康熙皇帝", "顺治皇帝",
    "北洋水师", "北洋海军", "南洋水师",
    "虎门销烟", "安庆之战", "马关条约", "南京条约", "北京条约",
    "瀛台", "圆明园", "颐和园",
]
for _term in _QING_TERMS:
    jieba.add_word(_term)

# BM25 停用词
_BM25_STOP_WORDS: Set[str] = {
    "的", "是", "还是", "了", "吗", "呢", "吧", "啊", "呀",
    "什么", "怎么", "怎样", "如何", "为什么", "为何",
    "多少", "几个", "哪", "哪个", "哪些", "谁",
    "不", "没", "很", "都", "就", "也", "会", "要", "能", "可以",
    "在", "和", "与", "或", "及", "之", "等", "其", "这", "那",
    "有", "被", "把", "从", "到", "对", "向", "将", "以",
    "一个", "一种", "这个", "那个", "中", "后", "前", "上", "下",
    "是否", "有无", "可否", "怎么办", "怎样办",
}


# ============================================================
# 2. 实体词典 (Entity Registry)
# ============================================================

# 2a. 年号
_REIGN_TITLES = [
    "天命", "天聪", "崇德",
    "顺治", "康熙", "雍正", "乾隆", "嘉庆", "道光",
    "咸丰", "同治", "光绪", "宣统",
]

# 2b. 机构 (政府衙门/军事组织)
_OFFICES = [
    # 中央机构
    "军机处", "内阁", "内务府", "理藩院", "都察院", "大理寺", "太常寺",
    "光禄寺", "太仆寺", "鸿胪寺", "国子监", "钦天监", "宗人府",
    "通政使司", "翰林院", "詹事府", "太医院",
    # 六部及其司
    "吏部", "户部", "礼部", "兵部", "刑部", "工部",
    "总理衙门", "总理各国事务衙门", "外务部",
    "总理行营事务处", "会议政务处",
    "稽察房", "典籍厅", "档房",
    "实录馆", "会典馆", "方略馆", "国史馆",
    # 地方衙门
    "布政使司", "按察使司", "盐运使司", "粮储道",
    # 军事
    "八旗", "满洲八旗", "蒙古八旗", "汉军八旗", "绿营",
    "水师", "北洋水师", "南洋水师", "北洋海军",
    "神机营", "健锐营", "火器营", "虎枪营",
    # 其他
    "理刑清吏司", "仪制清吏司", "祠祭清吏司", "主客清吏司",
    "精膳清吏司", "车驾清吏司", "职方清吏司", "武选清吏司",
    "武库清吏司", "屯田清吏司", "都水清吏司", "虞衡清吏司",
]

# 2c. 官职
_TITLES = [
    "大学士", "协办大学士", "军机大臣", "军机章京", "尚书", "侍郎",
    "总督", "巡抚", "布政使", "按察使", "道员", "知府", "知州", "知县",
    "提督", "总兵", "副将", "参将", "游击", "都司", "守备", "千总", "把总",
    "都统", "副都统", "参领", "佐领", "骁骑校", "领催",
    "掌印郎中", "员外郎", "主事", "笔帖式", "给事中", "监察御史",
    "内阁学士", "侍读学士", "侍讲学士", "侍读", "侍讲",
    "钦差大臣", "办事大臣", "参赞大臣", "领队大臣", "帮办大臣",
    "盛京将军", "伊犁将军", "黑龙江将军", "吉林将军", "乌里雅苏台将军",
    "河道总督", "漕运总督", "仓场侍郎",
    "盐政", "学政", "织造", "关监督",
    "知县", "知府", "知州", "同知", "通判", "县丞",
    "郎中", "员外郎", "主事",
    "都察院左都御史", "都察院右都御史",
    "总理", "协理", "章京", "司员",
    # 爵位
    "亲王", "郡王", "贝勒", "贝子", "公", "侯", "伯", "子", "男",
    "和硕亲王", "多罗郡王", "固山贝子",
]

# 2d. 地名
_GEO_ENTITIES = [
    # 省/大区
    "直隶", "盛京", "奉天", "吉林", "黑龙江",
    "江苏", "安徽", "江西", "浙江", "福建", "台湾",
    "河南", "山东", "山西", "陕西", "甘肃",
    "湖北", "湖南", "广东", "广西", "四川", "贵州", "云南",
    "新疆", "西藏", "蒙古", "青海",
    "江南", "湖广", "陕甘", "两广", "闽浙", "云贵",
    # 边疆
    "伊犁", "乌里雅苏台", "喀什噶尔", "塔尔巴哈台", "阿克苏",
    "叶尔羌", "和阗", "科布多", "库伦",
    "宁古塔", "三姓", "珲春", "黑龙江城",
    "拉萨", "前藏", "后藏",
    # 重要城市/府
    "北京", "顺天", "南京", "江宁", "苏州", "扬州", "镇江", "上海", "松江",
    "杭州", "福州", "厦门", "广州", "澳门", "香港",
    "天津", "保定", "西安", "兰州", "成都", "重庆",
    "武昌", "汉口", "汉阳", "长沙", "昆明", "贵阳",
    "开封", "济南", "太原", "南昌", "安庆",
    # 军事要地
    "基隆", "澎湖", "淡水", "台南", "凤山", "嘉义", "彰化",
    "虎门", "大沽", "旅顺", "威海卫", "刘公岛",
    # 园林/宫苑
    "圆明园", "颐和园", "瀛台", "畅春园", "静宜园", "静明园",
    "避暑山庄", "木兰围场",
    # 江河
    "黄河", "长江", "淮河", "运河", "永定河", "海河", "珠江",
]

# 2e. 人名
_PEOPLE = [
    # 清初
    "多尔衮", "范文程", "洪承畴", "吴三桂", "尚可喜", "耿精忠",
    "鳌拜", "索尼", "苏克萨哈", "遏必隆",
    # 康雍乾
    "纳兰明珠", "索额图", "张廷玉", "鄂尔泰", "年羹尧", "岳钟琪",
    "李卫", "田文镜", "孙嘉淦", "刘统勋",
    "福康安", "和珅", "纪晓岚", "刘墉", "阿桂", "海兰察",
    # 晚清
    "林则徐", "琦善", "曾国藩", "左宗棠", "李鸿章", "张之洞",
    "刘铭传", "沈葆桢", "曾国荃", "胡林翼", "彭玉麟",
    "僧格林沁", "肃顺", "奕訢", "奕譞", "载沣",
    "翁同龢", "康有为", "梁启超", "谭嗣同",
    "袁世凯", "荣禄", "端方", "盛宣怀",
    "赫德", "戈登", "华尔",
    # 帝王（复合形式已在 _QING_TERMS 中有）
    # 别称（部分）
    "慈禧", "慈禧太后",
]

# 2f. 制度/事件类词汇（辅助实体识别）
_EVENT_TERMS = [
    "丁忧", "起复", "休致", "原品休致", "引见", "议叙", "议处",
    "科举", "乡试", "会试", "殿试", "科场", "进士", "举人",
    "捐纳", "捐输", "捐官", "报捐",
    "海防", "塞防", "边务", "夷务", "洋务",
    "教案", "通商", "条约", "租界",
    "河工", "漕运", "盐政", "钱法", "矿务",
    "赈济", "蠲免", "抚恤", "灾赈",
]

# ── 构建各实体类型的编译正则 ──
def _build_entity_pattern(terms: List[str]) -> re.Pattern:
    """将术语列表编译为按长度降序排列的正则（长词优先匹配）"""
    escaped = [re.escape(t) for t in sorted(set(terms), key=len, reverse=True)]
    return re.compile('|'.join(escaped))

_ENTITY_REGEX = {
    "reign":  _build_entity_pattern(_REIGN_TITLES),
    "office": _build_entity_pattern(_OFFICES),
    "title":  _build_entity_pattern(_TITLES),
    "geo":    _build_entity_pattern(_GEO_ENTITIES),
    "person": _build_entity_pattern(_PEOPLE),
    "event":  _build_entity_pattern(_EVENT_TERMS),
}

# 简体→实体类型 快速查找（用于查询侧实体匹配）
_ENTITY_LOOKUP: Dict[str, str] = {}
for _etype, _terms in [
    ("reign", _REIGN_TITLES),
    ("office", _OFFICES),
    ("title", _TITLES),
    ("geo", _GEO_ENTITIES),
    ("person", _PEOPLE),
    ("event", _EVENT_TERMS),
]:
    for _t in _terms:
        _s = _cc.convert(_t)
        if _s not in _ENTITY_LOOKUP or len(_t) > len(_ENTITY_LOOKUP.get(_s, "")):
            _ENTITY_LOOKUP[_s] = _etype

# 人名别名映射（繁体 → 简体主名）
_PERSON_ALIASES: Dict[str, str] = {
    "曾文正": "曾国藩", "曾涤生": "曾国藩", "曾中堂": "曾国藩", "曾帅": "曾国藩",
    "李中堂": "李鸿章", "李傅相": "李鸿章", "李少荃": "李鸿章", "李合肥": "李鸿章",
    "左文襄": "左宗棠", "左季高": "左宗棠",
    "林文忠": "林则徐", "林少穆": "林则徐", "林制军": "林则徐",
    "张香涛": "张之洞", "张南皮": "张之洞",
    "刘省三": "刘铭传", "刘壮肃": "刘铭传",
    "沈幼丹": "沈葆桢", "沈文肃": "沈葆桢",
    "袁项城": "袁世凯", "袁慰亭": "袁世凯",
    "和致斋": "和珅", "和相": "和珅",
}


# ============================================================
# 3. 实体抽取器
# ============================================================

class EntityExtractor:
    """从文本和档案记录中抽取清代专有实体（年号/机构/官职/地名/人名/事件）"""

    # 责任时间 → 年号 解析正则
    _REIGN_DATE_RE = re.compile(
        r'(' + '|'.join(_REIGN_TITLES) + r')(\d+)年'
    )
    # 责任者字段清理
    _AUTHOR_CLEAN_RE = re.compile(r'\s*[\(（].*?[\)）]\s*')

    @classmethod
    def from_query(cls, query_simp: str) -> Dict[str, Set[str]]:
        """从简体中文查询中抽取实体，返回 {type: {entity_str, ...}}"""
        entities: Dict[str, Set[str]] = defaultdict(set)

        for etype, pattern in _ENTITY_REGEX.items():
            for m in pattern.finditer(query_simp):
                entities[etype].add(m.group())

        # 人名别名处理
        resolved_persons = set()
        for p in entities.get("person", set()):
            resolved_persons.add(_PERSON_ALIASES.get(p, p))
        if resolved_persons:
            entities["person"] = resolved_persons

        # 从 jieba 分词中补充未匹配的实体
        tokens = set(jieba.lcut(query_simp))
        tokens.update(query_simp)  # 同时搜索原始字符串中的子串
        for token in tokens:
            if token in _ENTITY_LOOKUP:
                etype = _ENTITY_LOOKUP[token]
                resolved = _PERSON_ALIASES.get(token, token)
                entities[etype].add(resolved)

        return dict(entities)

    @classmethod
    def from_record(cls, record: Dict[str, str]) -> Dict[str, Set[str]]:
        """从档案记录中抽取实体，返回 {type: {entity_str, ...}}

        对于 refined 格式：融合预提取实体 + 文本提取实体
          - people/office/geo: 优先使用预提取实体，同时从 source_text 补充
          - reign/title/event: 从 source_text 文本中提取
        """
        entities: Dict[str, Set[str]] = defaultdict(set)

        # ── refined 专属：预提取实体直接使用 ──
        refined = record.get("refined_data", None)
        if refined is not None:
            # 年号：从 emperor 字段直接获取
            emperor = record.get("emperor", "") or ""
            if emperor:
                for rt in _REIGN_TITLES:
                    if rt in emperor:
                        entities["reign"].add(rt)
                        break

            # 从 date 字段补充年号
            date_str = record.get("責任時間(中曆)", "") or record.get("date", "") or ""
            m = cls._REIGN_DATE_RE.search(date_str)
            if m:
                entities["reign"].add(m.group(1))

            # 预提取实体：entities_people → person
            for p in refined.get("entities_people", []) or []:
                p_clean = p.strip()
                if p_clean:
                    # 尝试匹配已知人名
                    matched = False
                    for known in _PEOPLE:
                        if known in p_clean or p_clean in known:
                            entities["person"].add(known)
                            matched = True
                            break
                    if not matched:
                        entities["person"].add(p_clean)

            # 预提取实体：entities_office → office
            for o in refined.get("entities_office", []) or []:
                o_clean = o.strip()
                if o_clean:
                    entities["office"].add(o_clean)

            # 预提取实体：entities_geo → geo
            for g in refined.get("entities_geo", []) or []:
                g_clean = g.strip()
                if g_clean:
                    entities["geo"].add(g_clean)

            # 从 source_text 补充提取（补漏：event、title 及其他类型）
            source = record.get("source_text", "") or ""
            if source:
                for etype, pattern in _ENTITY_REGEX.items():
                    for m in pattern.finditer(source):
                        entities[etype].add(m.group())

            return dict(entities)

        # ── filtered（原始）格式：原有逻辑不变 ──
        # 责任时间 → 年号
        date_str = record.get("責任時間(中曆)", "") or record.get("责任时间(中曆)", "")
        m = cls._REIGN_DATE_RE.search(date_str)
        if m:
            entities["reign"].add(m.group(1))

        # 年号字段
        reign_field = record.get("年号", "") or ""
        if reign_field:
            for rt in _REIGN_TITLES:
                if rt in reign_field:
                    entities["reign"].add(rt)

        # 责任者 → 人名
        author = record.get("責任者", "") or record.get("责任者", "") or ""
        author_clean = cls._AUTHOR_CLEAN_RE.sub('', author).strip()
        if author_clean and len(author_clean) >= 2:
            matched = False
            for p in _PEOPLE:
                if p in author_clean:
                    entities["person"].add(p)
                    matched = True
                    break
            if not matched:
                for t in _TITLES:
                    if t in author_clean:
                        entities["title"].add(t)
                        matched = True
                        break
            if not matched:
                entities["person"].add(author_clean)

        # 文本字段抽取
        title = record.get("題名", "") or record.get("题名", "") or ""
        abstract = record.get("內容提要", "") or record.get("内容提要", "") or ""
        combined = title + abstract

        if combined:
            for etype, pattern in _ENTITY_REGEX.items():
                for m in pattern.finditer(combined):
                    entities[etype].add(m.group())

        return dict(entities)


# ============================================================
# 4. 主检索器
# ============================================================

class ArchiveRetriever:
    """
    清史档案混合检索器

    用法:
        # 原始文言文档案
        retriever = ArchiveRetriever(
            data_dir="数据基础/档案_过滤",
            cache_dir="cache"
        )

        # 微调后的现代文摘要档案
        retriever = ArchiveRetriever(
            data_format="refined",
            data_dir="数据基础/档案refined",
            cache_dir="cache_refined"
        )

        retriever.load_or_build()    # 首次构建(30-50分钟)，后续秒级加载
        results = retriever.search("乾隆年间兵部军需奏销", top_k=10)
    """

    def __init__(
        self,
        data_format: str = "filtered",
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        use_gpu: bool = True,
    ):
        self.data_format = data_format  # "filtered" | "refined"

        # 根据 data_format 设置默认目录
        if data_dir:
            self.data_dir = Path(data_dir)
        elif data_format == "refined":
            self.data_dir = _DEFAULT_REFINED_DATA_DIR
        else:
            self.data_dir = _DEFAULT_DATA_DIR

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif data_format == "refined":
            self.cache_dir = _DEFAULT_REFINED_CACHE_DIR
        else:
            self.cache_dir = _DEFAULT_CACHE_DIR

        self.model_name = model_name
        self.use_gpu = use_gpu
        os.makedirs(str(self.cache_dir), exist_ok=True)

        # 缓存文件路径
        self._bm25_cache_path = self.cache_dir / "bm25_index.pkl"
        self._faiss_cache_path = self.cache_dir / "faiss_index.bin"
        self._entities_cache_path = self.cache_dir / "doc_entities.pkl"
        self._metadata_path = self.cache_dir / "metadata.json"

        # 运行状态
        self.records: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_entities: List[Dict[str, Set[str]]] = []
        self.entity_to_docs: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))

        # 索引对象（延迟初始化）
        self._bm25: Optional[BM25Okapi] = None
        self._faiss_index = None
        self._embedding_model = None
        self._ready = False

        # 懒惰导入标记
        self._bm25_class = None

    # ──── 公共接口 ────

    def load_or_build(self, force_rebuild: bool = False) -> "ArchiveRetriever":
        """
        加载缓存索引，或从零构建。
        首次运行自动构建所有索引（BM25 + FAISS + 实体标注），
        后续运行直接从缓存加载（秒级完成）。
        """
        tag = f"[ArchiveRetriever:{self.data_format}]"
        if not force_rebuild and self._all_caches_valid():
            print(f"{tag} 从缓存加载索引...")
            self._load_all_caches()
            self._ready = True
            print(f"  加载完成: {len(self.records)} 条档案记录")
        else:
            print(f"{tag} 开始构建索引...", flush=True)

            # Step 1: 数据 + BM25（支持断点续跑）
            bm25_loaded = self._try_load_bm25_cache()
            if bm25_loaded:
                print("  [1/3] BM25 索引: 从缓存加载完成", flush=True)
            else:
                self._load_data()
                self._build_bm25_index()
                self._save_bm25_cache_only()

            # Step 2: 向量
            self._build_vector_index()

            # Step 3: 实体
            self._build_entity_index()

            self._save_all_caches()
            self._ready = True
            print(f"{tag} 索引构建完成: {len(self.records)} 条记录", flush=True)
        return self

    def search(self, query: str, top_k: int = 10, verbose: bool = True) -> List[Dict]:
        """
        混合检索主入口。

        Args:
            query: 用户查询（简体中文）
            top_k: 返回结果数量，默认 10
            verbose: 是否打印检索过程信息

        Returns:
            [
                {
                    "rank": 1,
                    "final_score": 0.934,
                    "bm25_score": 0.88,
                    "vector_score": 0.91,
                    "rrf_score": 0.032,
                    "relation_score": 0.15,
                    "entity_score": 0.18,
                    "inst_score": 0.12,
                    "temporal_score": 0.15,
                    "geo_score": 0.10,
                    "matched_entities": ["乾隆", "兵部", "军需"],
                    "record": { ... }   # 原始档案记录（简体转换后）
                },
                ...
            ]
        """
        if not self._ready:
            raise RuntimeError("索引未就绪，请先调用 load_or_build()")

        t0 = time.time()

        # Step 0: 查询处理
        query_simp = _cc.convert(query)
        query_entities = EntityExtractor.from_query(query_simp)

        if verbose:
            ent_summary = ", ".join(
                f"{t}:{','.join(list(e)[:3])}"
                for t, e in query_entities.items() if e
            )
            if ent_summary:
                print(f"  [查询实体] {ent_summary}")

        # Step 1: 双通道检索
        N_CANDIDATES = 100

        t1 = time.time()
        bm25_results = self._bm25_search(query_simp, top_k=N_CANDIDATES)
        t2 = time.time()

        vector_results = self._vector_search(query_simp, top_k=N_CANDIDATES)
        t3 = time.time()

        if verbose:
            print(f"  BM25: {len(bm25_results)}条 ({t2-t1:.2f}s)  "
                  f"向量: {len(vector_results)}条 ({t3-t2:.2f}s)")

        # Step 2: RRF 融合
        fused = self._rrf_fusion(bm25_results, vector_results, top_k=50)
        t4 = time.time()

        if verbose:
            print(f"  RRF融合: {len(fused)}条候选 ({t4-t3:.2f}s)")

        # Step 3: 关系加权重排序
        reranked = self._relationship_rerank(
            fused, query_entities, top_k=top_k
        )
        t5 = time.time()

        if verbose:
            print(f"  关系重排: Top-{top_k} ({t5-t4:.2f}s)")
            print(f"  总耗时: {t5-t0:.2f}s")
            if reranked:
                top = reranked[0]
                print(f"  最佳匹配: {top.get('record',{}).get('題名','(无题名)')[:60]}")

        return reranked

    def get_record(self, index: int) -> Optional[Dict]:
        """按索引获取原始档案记录"""
        if 0 <= index < len(self.records):
            return self.records[index]
        return None

    def format_result(self, result: Dict) -> str:
        """格式化单条结果为可读文本（record 在 search 中已转为简体键）"""
        rec = result.get("record", {})

        # refined 格式：同时展示现代文摘要 + 原文
        if self.data_format == "refined":
            rd = rec.get("refined_data", {})
            modern = rd.get("modern_summary", "") if isinstance(rd, dict) else ""
            source = rec.get("source_text", "") or ""
            if isinstance(source, str) and len(source) > 200:
                source_display = source[:200] + "..."
            else:
                source_display = str(source) if source else ""

            lines = [
                f"[{result['rank']}] 📜 现代文摘要:",
                f"    {modern[:200]}{'...' if len(str(modern)) > 200 else ''}",
            ]
            if source_display:
                lines.append(f"    📖 原文: {source_display}")
            lines += [
                f"    朝代: {rec.get('来源朝代', '')}  |  "
                f"时间: {rec.get('责任时间(中历)', rec.get('責任時間(中曆)', ''))}",
                f"    评分: final={result['final_score']:.3f}  "
                f"bm25={result['bm25_score']:.3f}  "
                f"vec={result['vector_score']:.3f}  "
                f"rel={result['relation_score']:.3f}",
            ]
            return "\n".join(lines)

        # filtered（原始）格式：原逻辑不变
        lines = [
            f"[{result['rank']}] {rec.get('题名', '(无题名)')}",
            f"    内容: {rec.get('内容提要', '')[:150]}",
            f"    责任者: {rec.get('责任者', '')}  |  "
            f"时间: {rec.get('责任时间(中历)', rec.get('責任時間(中曆)', ''))}  |  "
            f"朝代: {rec.get('来源朝代', '')}",
            f"    评分: final={result['final_score']:.3f}  "
            f"bm25={result['bm25_score']:.3f}  "
            f"vec={result['vector_score']:.3f}  "
            f"rel={result['relation_score']:.3f}",
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.records)

    # ──── 数据加载 ────

    @staticmethod
    def _parse_source_text(source_text: str) -> Dict[str, str]:
        """从 refined 数据的 source_text 字符串中提取结构化字段。

        source_text 格式示例:
            题名：<title>
            内容提要：<abstract>
            责任者：<author> (基本資料)
            时间：<date>
        """
        result = {'题名': '', '内容提要': '', '责任者': '', '时间': ''}
        if not source_text:
            return result

        patterns = [
            ('题名', r'题名[：:]\s*(.*?)(?:\n|$)'),
            ('内容提要', r'内容提要[：:]\s*(.*?)(?:\n|$)'),
            ('责任者', r'责任者[：:]\s*(.*?)(?:\n|$)'),
            ('时间', r'时间[：:]\s*(.*?)(?:\n|$)'),
        ]
        for key, pat in patterns:
            m = re.search(pat, source_text)
            if m:
                val = m.group(1).strip()
                # 清理责任者中的 "(基本資料)" 等后缀
                if key == '责任者':
                    val = re.sub(r'\s*[\(（].*?[\)）]\s*$', '', val).strip()
                result[key] = val
        return result

    def _load_data(self):
        """从数据目录加载全部 JSON 数据"""
        json_files = sorted(
            f for f in os.listdir(str(self.data_dir))
            if f.endswith('.json')
        )
        if not json_files:
            raise FileNotFoundError(f"未找到 JSON 文件: {self.data_dir}")

        all_records = []
        for fname in json_files:
            fpath = self.data_dir / fname
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 从文件名推断朝代（去除 _refined 后缀）
            raw_stem = Path(fname).stem
            dynasty = raw_stem.replace('_refined', '')

            if self.data_format == "refined":
                for item in data:
                    parsed = self._parse_source_text(
                        item.get('source_text', '')
                    )
                    record = {
                        # 标准化字段（兼容下游检索逻辑）
                        '題名': parsed.get('题名', ''),
                        '內容提要': parsed.get('内容提要', ''),
                        '責任者': parsed.get('责任者', ''),
                        '責任時間(中曆)': item.get('date', ''),
                        '来源朝代': dynasty,
                        # refined 专属字段
                        'emperor': item.get('emperor', ''),
                        'refined_data': item.get('refined_data', {}),
                        'source_text': item.get('source_text', ''),
                        'original_index': item.get('original_index', -1),
                    }
                    all_records.append(record)
                print(f"    {fname}: {len(data)} 条 (refined 格式)")
            else:
                for item in data:
                    item['来源朝代'] = dynasty
                    all_records.append(item)
                print(f"    {fname}: {len(data)} 条")

        self.records = all_records
        print(f"  已加载 {len(self.records)} 条档案记录 ({len(json_files)} 个朝代)")

    # ──── 文档文本构建 ────

    def _build_doc_text_bm25(self, record: Dict) -> str:
        """BM25用：根据数据格式使用不同的加权策略。

        filtered: 题名×3 + 内容提要×1 + 责任者×1
        refined:  modern_summary×2 + source_text×1
        """
        if self.data_format == "refined":
            rd = record.get('refined_data', {})
            modern = rd.get('modern_summary', '')
            source = record.get('source_text', '')
            parts = []
            for _ in range(2):
                if modern:
                    parts.append(modern)
            if source:
                parts.append(source)
            return " ".join(parts)
        else:
            title = _cc.convert(record.get('題名', '') or record.get('题名', '') or '')
            abstract = _cc.convert(record.get('內容提要', '') or record.get('内容提要', '') or '')
            author = _cc.convert(record.get('責任者', '') or record.get('责任者', '') or '')
            parts = []
            for _ in range(3):
                if title:
                    parts.append(title)
            if abstract:
                parts.append(abstract)
            if author:
                parts.append(author)
            return " ".join(parts)

    def _build_doc_text_embed(self, record: Dict) -> str:
        """Embedding用：根据数据格式使用不同文本。

        filtered: 题名 + 内容提要 + 责任者（自然文本，不重复）
        refined:  modern_summary 仅用现代文摘要（语义质量更好）
        """
        if self.data_format == "refined":
            rd = record.get('refined_data', {})
            return rd.get('modern_summary', '')
        else:
            title = _cc.convert(record.get('題名', '') or record.get('题名', '') or '')
            abstract = _cc.convert(record.get('內容提要', '') or record.get('内容提要', '') or '')
            author = _cc.convert(record.get('責任者', '') or record.get('责任者', '') or '')
            parts = [p for p in [title, abstract, author] if p]
            return " ".join(parts)

    def _build_doc_flattened(self, record: Dict) -> str:
        """构建简体文本（用于实体匹配）。

        refined: modern_summary + source_text
        filtered: 题名 + 内容提要 + 责任者
        """
        if self.data_format == "refined":
            rd = record.get('refined_data', {})
            modern = rd.get('modern_summary', '')
            source = record.get('source_text', '')
            parts = [p for p in [modern, source] if p]
            return " ".join(parts)
        else:
            parts = []
            for key in ['題名', '题名', '內容提要', '内容提要', '責任者', '责任者']:
                val = record.get(key, '')
                if val:
                    parts.append(_cc.convert(val))
            return " ".join(parts)

    # ──── BM25 索引 ────

    def _build_bm25_index(self):
        """构建多字段加权 BM25 索引"""
        if self.data_format == "refined":
            bm25_desc = "modern_summary×2 source_text×1"
        else:
            bm25_desc = "题名×3 内容提要×1 责任者×1"
        print(f"  [1/3] 构建 BM25 索引 ({bm25_desc})...")
        from rank_bm25 import BM25Okapi
        self._bm25_class = BM25Okapi

        self.tokenized_corpus = []
        for i, rec in enumerate(self.records):
            tokens = list(jieba.cut(self._build_doc_text_bm25(rec)))
            # 停用词过滤
            tokens = [t for t in tokens if t.strip() and t not in _BM25_STOP_WORDS]
            self.tokenized_corpus.append(tokens)
            if (i + 1) % 50000 == 0:
                print(f"    已分词 {i + 1}/{len(self.records)} 条...")

        self._bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"    BM25 索引完成: {len(self.tokenized_corpus)} 篇文档")

    def _bm25_search(self, query_simp: str, top_k: int = 100) -> List[Dict]:
        """BM25 关键词检索"""
        if self._bm25 is None:
            return []

        query_tokens = [
            t for t in jieba.cut(query_simp)
            if t.strip() and t not in _BM25_STOP_WORDS
        ]
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                'index': int(idx),
                'score': round(score, 4),
                'channel': 'bm25',
            })
        return results

    # ──── 向量索引 ────

    def _get_embedding_model(self):
        """延迟加载 embedding 模型"""
        if self._embedding_model is not None:
            return self._embedding_model

        print("    加载 BGE-large-zh-v1.5 模型...")
        from sentence_transformers import SentenceTransformer
        from pathlib import Path  # 必须导入这个
        import os

        # 1. 定义模型目录（确保是 Path 对象）
        local_path = _DEFAULT_MODEL_DIR / "bge-large-zh-v1.5"
        
        # 2. 修改判断逻辑：兼容 .bin 和 .safetensors (我们之前转换过格式)
        is_local = local_path.exists() and (
            (local_path / "pytorch_model.bin").exists() or 
            (local_path / "model.safetensors").exists()
        )

        if is_local:
            model_path = str(local_path)
            print(f"    ✅ 成功检测到本地模型: {model_path}")
        else:
            # 如果本地没找到，依然走原来的逻辑
            model_path = self.model_name
            print(f"    ⚠️ 未检测到本地文件，路径检查为: {local_path}")
            print(f"    尝试从 HuggingFace 下载到: {_DEFAULT_MODEL_DIR}")
            os.makedirs(str(_DEFAULT_MODEL_DIR), exist_ok=True)

        device = "cuda" if self.use_gpu else "cpu"
        
        # 3. 加载模型（去掉 cache_folder，防止它又去联网检查更新）
        self._embedding_model = SentenceTransformer(
            model_path,
            device=device
        )
        
        # BGE 模型需要指令前缀以提升检索效果
        self._embedding_model.max_seq_length = 512
        return self._embedding_model
    def _build_vector_index(self):
        """构建 FAISS 向量索引（batch_size=16 + 每5000条checkpoint + 断点续跑）

        checkpoint 策略:
          - 原子写入：先写 .tmp 再 os.replace，防止 truncate 后进程被杀导致文件损毁
          - 保留最近 3 个快照，即使最新快照损坏也能从上一个恢复
        """
        print("  [2/3] 构建向量索引 (BGE-large-zh-v1.5 + FAISS)...", flush=True)
        import faiss

        model = self._get_embedding_model()

        doc_texts = [self._build_doc_text_embed(rec) for rec in self.records]
        total = len(doc_texts)
        PREFIX = "为这个句子生成表示以用于检索相关文章："
        CHUNK_SIZE = 5000
        MAX_SNAPSHOTS = 3

        # ── 断点续跑：扫描所有快照，取最新可用的 ──
        def _find_latest_snapshot():
            # 过滤掉 .tmp.npy 残留文件（写入中途崩溃留下的垃圾）
            snapshots = sorted(
                [p for p in self.cache_dir.glob("embeddings_ckpt_*.npy")
                 if ".tmp." not in p.name],
                key=lambda p: int(p.stem.rsplit("_", 1)[-1])
            )
            # 清理残留 tmp 文件
            for tmp in self.cache_dir.glob("embeddings_ckpt_*.tmp.npy"):
                tmp.unlink()
            for snap in reversed(snapshots):
                try:
                    start = int(snap.stem.rsplit("_", 1)[-1])
                    emb = np.load(str(snap))
                    if emb.shape[0] == start:
                        print(f"    [恢复] 从 checkpoint 继续: 已完成 {start}/{total} "
                              f"({start*100/total:.1f}%)  (快照: {snap.name})", flush=True)
                        return start, [emb]
                    else:
                        print(f"    快照 {snap.name} 记录数不匹配 (预期{start} 实际{emb.shape[0]})，跳过",
                              flush=True)
                except Exception as e:
                    print(f"    快照 {snap.name} 损坏 ({e})，尝试上一个...", flush=True)
            return 0, []

        start_idx, embeddings_list = _find_latest_snapshot()

        # 兼容旧版 checkpoint 格式（embeddings_checkpoint.npy）
        old_ckpt = self.cache_dir / "embeddings_checkpoint.npy"
        old_cnt = self.cache_dir / "embeddings_checkpoint_count.txt"
        if start_idx == 0 and old_ckpt.exists() and old_cnt.exists():
            try:
                with open(old_cnt, 'r') as f:
                    old_start = int(f.read().strip())
                old_emb = np.load(str(old_ckpt))
                if old_emb.shape[0] == old_start and old_start > start_idx:
                    print(f"    [恢复] 从旧格式 checkpoint 继续: 已完成 {old_start}/{total}", flush=True)
                    start_idx = old_start
                    embeddings_list = [old_emb]
            except Exception:
                pass

        if start_idx >= total:
            print(f"    全部文档已编码完成，跳过。", flush=True)
        else:
            print(f"    编码 {total} 篇文档 (batch_size=16, 每{CHUNK_SIZE}条保存checkpoint)",
                  flush=True)
            print(f"    GPU: NVIDIA GeForce RTX 3050 6GB", flush=True)

            t_start = time.time()
            chunk_start = start_idx

            while chunk_start < total:
                chunk_end = min(chunk_start + CHUNK_SIZE, total)
                chunk_texts = doc_texts[chunk_start:chunk_end]
                chunk_prefixed = [f"{PREFIX}{t}" for t in chunk_texts]

                chunk_emb = model.encode(
                    chunk_prefixed,
                    batch_size=16,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                embeddings_list.append(chunk_emb)
                chunk_start = chunk_end

                # 原子写入 checkpoint
                combined = np.vstack(embeddings_list)
                snap_name = f"embeddings_ckpt_{chunk_start:08d}.npy"
                snap_path = self.cache_dir / snap_name
                tmp_path = self.cache_dir / (snap_name + ".tmp.npy")
                np.save(str(tmp_path), combined)
                os.replace(str(tmp_path), str(snap_path))

                # 清理旧快照和旧格式 checkpoint
                all_snaps = sorted(
                    self.cache_dir.glob("embeddings_ckpt_*.npy"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[-1])
                )
                for old in all_snaps[:-MAX_SNAPSHOTS]:
                    old.unlink()
                for p in [old_ckpt, old_cnt]:
                    if p.exists():
                        p.unlink()

                elapsed = time.time() - t_start
                pct = chunk_start * 100 / total
                eta = elapsed / (chunk_start - start_idx) * (total - chunk_start) if chunk_start > start_idx else 0
                print(f"    [checkpoint] {chunk_start}/{total} ({pct:.1f}%)  "
                      f"耗时 {elapsed/60:.1f}min  ETA {eta/60:.1f}min",
                      flush=True)

        # ── 合并 & 构建 FAISS ──
        embeddings = np.vstack(embeddings_list)
        print(f"    向量编码完成: {embeddings.shape}", flush=True)

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings.astype(np.float32))
        print(f"    FAISS 索引完成: {self._faiss_index.ntotal} 条, 维度={dim}", flush=True)

        # 清理 checkpoint 快照
        for snap in self.cache_dir.glob("embeddings_ckpt_*.npy"):
            snap.unlink()

    def _vector_search(self, query_simp: str, top_k: int = 100) -> List[Dict]:
        """向量语义检索"""
        if self._faiss_index is None:
            return []

        model = self._get_embedding_model()
        # BGE 查询指令前缀（与文档侧前缀不同）
        query_with_prefix = f"为这个句子生成表示以用于检索相关文章：{query_simp}"
        query_embedding = model.encode(
            [query_with_prefix],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self._faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            results.append({
                'index': int(idx),
                'score': round(float(score), 4),
                'channel': 'vector',
            })
        return results

    # ──── 实体索引 ────

    def _build_entity_index(self):
        """为全部文档预抽取实体，构建文档→实体 和 实体→文档 索引"""
        print("  [3/3] 构建实体索引...")
        self.doc_entities = []
        self.entity_to_docs: Dict[str, Dict[str, Set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for i, rec in enumerate(self.records):
            entities = EntityExtractor.from_record(rec)
            self.doc_entities.append(entities)
            # 更新倒排索引
            for etype, ent_set in entities.items():
                for ent in ent_set:
                    ent_simp = _cc.convert(ent)
                    self.entity_to_docs[etype][ent_simp].add(i)

            if (i + 1) % 50000 == 0:
                print(f"    已标注实体 {i + 1}/{len(self.records)} 条...")

        total_entities = sum(len(v) for v in self.entity_to_docs.values())
        print(f"    实体索引完成: {len(self.doc_entities)} 篇文档, "
              f"{total_entities} 个实体倒排条目")

    # ──── RRF 融合 ────

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        k: int = 60,
        top_k: int = 50,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion: RRF(d) = Σ_{ranking} 1/(k + rank_d)

        并行双通道结果融合，k=60是标准参数。
        返回融合后的候选文档列表，带 rrf_score。
        """
        rrf_scores: Dict[int, float] = {}
        doc_info: Dict[int, Dict] = {}

        for ranking in [bm25_results, vector_results]:
            for rank, item in enumerate(ranking):
                idx = item['index']
                rrf = 1.0 / (k + rank + 1)
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + rrf
                if idx not in doc_info:
                    doc_info[idx] = {
                        'index': idx,
                        'bm25_score': 0.0,
                        'vector_score': 0.0,
                    }
                doc_info[idx][f'{item["channel"]}_score'] = max(
                    doc_info[idx][f'{item["channel"]}_score'],
                    item['score']
                )

        # 按 RRF 得分排序
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for idx, rrf_score in sorted_indices[:top_k]:
            info = doc_info[idx]
            info['rrf_score'] = round(rrf_score, 6)
            info['record'] = self.records[idx]
            fused.append(info)

        return fused

    # ──── 关系加权重排序 ────

    def _relationship_rerank(
        self,
        candidates: List[Dict],
        query_entities: Dict[str, Set[str]],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        基于实体关系图谱对候选集重排序。

        4个维度:
          - 实体共现  (30%): 与 Top-5 锚文档共享实体的程度
          - 机构隶属  (30%): 责任者/衙门与查询实体的匹配
          - 时序邻近  (20%): 年号/朝代与查询的时间关联
          - 地理关联  (20%): 地名与查询的地理关联

        综合得分: final = 0.5 * rrf_norm + 0.5 * relation
        """
        N = len(candidates)
        if N == 0:
            return []

        # Step 1: 收集候选文档实体
        cand_entities = []
        for c in candidates:
            idx = c['index']
            entities = self.doc_entities[idx] if idx < len(self.doc_entities) else {}
            # 将繁体实体转为简体（与查询实体一致）
            simp_entities: Dict[str, Set[str]] = defaultdict(set)
            for etype, ent_set in entities.items():
                for ent in ent_set:
                    simp_entities[etype].add(_cc.convert(ent))
            cand_entities.append(simp_entities)

        # Step 2: 归一化 RRF 分数
        rrf_scores = np.array([c['rrf_score'] for c in candidates])
        rrf_max = rrf_scores.max()
        if rrf_max > 0:
            rrf_norm = rrf_scores / rrf_max
        else:
            rrf_norm = np.ones(N) / N

        # Step 3: 找到 Top-5 锚文档
        top5_indices = np.argsort(rrf_scores)[-min(5, N):]

        # Step 4: 计算各维度得分
        entity_scores = np.zeros(N)
        inst_scores = np.zeros(N)
        temporal_scores = np.zeros(N)
        geo_scores = np.zeros(N)

        for i in range(N):
            c = candidates[i]
            entities_i = cand_entities[i]
            record = c.get('record', {})

            # ── 4a. 实体共现得分 (30%) ──
            cohesion = 0.0
            for j in top5_indices:
                if i != j:
                    shared = self._count_shared_entities(
                        entities_i, cand_entities[j]
                    )
                    cohesion += shared * rrf_norm[j]
            # 归一化
            max_shared_possible = max(1, sum(
                len(self._flatten_entities(cand_entities[j]))
                for j in top5_indices if j != i
            ))
            entity_scores[i] = min(1.0, cohesion / max(1.0, max_shared_possible / len(top5_indices)))

            # ── 4b. 机构隶属得分 (30%) ──
            inst_scores[i] = self._compute_institutional_score(
                record, entities_i, query_entities
            )

            # ── 4c. 时序邻近得分 (20%) ──
            temporal_scores[i] = self._compute_temporal_score(
                record, entities_i, query_entities
            )

            # ── 4d. 地理关联得分 (20%) ──
            geo_scores[i] = self._compute_geo_score(
                entities_i, query_entities
            )

        relation_scores = (
            0.30 * entity_scores
            + 0.30 * inst_scores
            + 0.20 * temporal_scores
            + 0.20 * geo_scores
        )

        # Step 5: 综合得分
        final_scores = 0.5 * rrf_norm + 0.5 * relation_scores

        # Step 6: 排序输出
        ranked_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            c = candidates[idx]
            rec = c['record']
            simp_rec = {}
            for k, v in rec.items():
                simp_rec[_cc.convert(k)] = _cc.convert(str(v)) if isinstance(v, str) else v

            # 匹配到的实体（查询与文档的交集）
            matched = set()
            for etype in query_entities:
                q_set = query_entities.get(etype, set())
                d_set = cand_entities[idx].get(etype, set())
                matched.update(q_set & d_set)

            results.append({
                'rank': rank + 1,
                'index': int(c['index']),
                'final_score': round(float(final_scores[idx]), 4),
                'bm25_score': round(float(c.get('bm25_score', 0)), 4),
                'vector_score': round(float(c.get('vector_score', 0)), 4),
                'rrf_score': round(float(rrf_scores[idx]), 4),
                'relation_score': round(float(relation_scores[idx]), 4),
                'entity_score': round(float(entity_scores[idx]), 4),
                'inst_score': round(float(inst_scores[idx]), 4),
                'temporal_score': round(float(temporal_scores[idx]), 4),
                'geo_score': round(float(geo_scores[idx]), 4),
                'matched_entities': sorted(list(matched)),
                'query_entities': {
                    t: sorted(list(e)) for t, e in query_entities.items() if e
                },
                'record': simp_rec,
            })

        return results

    def _flatten_entities(self, entities: Dict[str, Set[str]]) -> Set[str]:
        """将所有类型的实体合并为一个集合"""
        result = set()
        for ent_set in entities.values():
            result.update(ent_set)
        return result

    def _count_shared_entities(
        self,
        entities_a: Dict[str, Set[str]],
        entities_b: Dict[str, Set[str]],
    ) -> int:
        """计算两个实体集之间的共享实体数"""
        total = 0
        for etype in set(list(entities_a.keys()) + list(entities_b.keys())):
            a_set = entities_a.get(etype, set())
            b_set = entities_b.get(etype, set())
            total += len(a_set & b_set)
        return total

    def _compute_institutional_score(
        self,
        record: Dict,
        doc_entities: Dict[str, Set[str]],
        query_entities: Dict[str, Set[str]],
    ) -> float:
        """
        机构隶属得分: 检查文档的 office/title 实体与查询的重叠，
        以及责任者与查询中的人名匹配。
        """
        score = 0.0

        q_offices = query_entities.get("office", set())
        q_titles = query_entities.get("title", set())
        q_persons = query_entities.get("person", set())

        d_offices = doc_entities.get("office", set())
        d_titles = doc_entities.get("title", set())
        d_persons = doc_entities.get("person", set())

        # 机构匹配
        if q_offices:
            overlap = len(d_offices & q_offices)
            score += 0.5 * min(1.0, overlap / len(q_offices))
            # 部分匹配 (如查询"兵部"，文档有"兵部职方清吏司")
            for qo in q_offices:
                for do in d_offices:
                    if qo in do or do in qo:
                        score += 0.1
                        break

        # 官职匹配
        if q_titles:
            overlap = len(d_titles & q_titles)
            score += 0.3 * min(1.0, overlap / len(q_titles))

        # 人名匹配（责任者）
        if q_persons and d_persons:
            overlap = len(d_persons & q_persons)
            score += 0.2 * min(1.0, overlap / len(q_persons))

        # 责任者字段直接检查
        author = _cc.convert(record.get("責任者", "") or record.get("责任者", "") or "")
        if q_persons:
            for qp in q_persons:
                if qp in author:
                    score += 0.2
                    break

        return min(1.0, score)

    def _compute_temporal_score(
        self,
        record: Dict,
        doc_entities: Dict[str, Set[str]],
        query_entities: Dict[str, Set[str]],
    ) -> float:
        """
        时序邻近得分: 文档的年号是否匹配查询中提到的年号。
        """
        q_reigns = query_entities.get("reign", set())
        d_reigns = doc_entities.get("reign", set())

        if not q_reigns:
            return 0.5  # 查询没有时态约束，中性分

        # 精确年号匹配
        if q_reigns & d_reigns:
            return 1.0

        # 相邻朝代匹配
        _REIGN_ORDER = {
            "顺治": 0, "康熙": 1, "雍正": 2, "乾隆": 3,
            "嘉庆": 4, "道光": 5, "咸丰": 6, "同治": 7,
            "光绪": 8, "宣统": 9,
        }

        for qr in q_reigns:
            q_order = _REIGN_ORDER.get(qr)
            if q_order is None:
                continue
            for dr in d_reigns:
                d_order = _REIGN_ORDER.get(dr)
                if d_order is None:
                    continue
                distance = abs(q_order - d_order)
                if distance == 1:
                    return 0.6  # 相邻朝代
                elif distance == 2:
                    return 0.3

        # 来源朝代后备检查
        dynasty = _cc.convert(record.get("来源朝代", "") or "")
        for qr in q_reigns:
            if qr in dynasty or dynasty in qr:
                return 1.0

        return 0.1  # 朝代不匹配

    def _compute_geo_score(
        self,
        doc_entities: Dict[str, Set[str]],
        query_entities: Dict[str, Set[str]],
    ) -> float:
        """
        地理关联得分: 文档的地名实体与查询的地理实体重叠度。
        """
        q_geo = query_entities.get("geo", set())
        d_geo = doc_entities.get("geo", set())

        if not q_geo:
            return 0.5  # 查询无地理约束，中性分

        if not d_geo:
            return 0.0

        overlap = len(d_geo & q_geo)
        # 部分匹配（如查询"台湾"，文档有"台湾府"）
        partial_bonus = 0.0
        for qg in q_geo:
            for dg in d_geo:
                if qg != dg and (qg in dg or dg in qg):
                    partial_bonus += 0.15

        return min(1.0, overlap / len(q_geo) + partial_bonus)

    # ──── 缓存管理 ────

    def _try_load_bm25_cache(self) -> bool:
        """尝试加载已有的 BM25 缓存（断点续跑用）"""
        if not self._bm25_cache_path.exists():
            return False
        # 检查源数据是否比缓存新（源目录不存在则跳过检查，信任缓存）
        if self.data_dir.exists():
            cache_mtime = os.path.getmtime(str(self._bm25_cache_path))
            for fname in os.listdir(str(self.data_dir)):
                if fname.endswith('.json'):
                    if os.path.getmtime(str(self.data_dir / fname)) > cache_mtime:
                        return False
        try:
            with open(self._bm25_cache_path, 'rb') as f:
                data = pickle.load(f)
            self.records = data['records']
            self.tokenized_corpus = data['tokenized_corpus']
            self._bm25 = data['bm25']
            print(f"  [1/3] BM25 缓存加载: {len(self.records)} 条记录", flush=True)
            return True
        except Exception:
            return False

    def _save_bm25_cache_only(self):
        """BM25 完成后立即单独保存，避免重跑"""
        with open(self._bm25_cache_path, 'wb') as f:
            pickle.dump({
                'records': self.records,
                'tokenized_corpus': self.tokenized_corpus,
                'bm25': self._bm25,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    BM25 已缓存 → {self._bm25_cache_path.name}", flush=True)

    def _all_caches_valid(self) -> bool:
        """检查所有缓存文件是否有效"""
        required = [
            self._bm25_cache_path,
            self._faiss_cache_path,
            self._entities_cache_path,
            self._metadata_path,
        ]
        if not all(p.exists() for p in required):
            return False

        # 检查 JSON 源数据是否比缓存更新
        try:
            meta = json.load(open(self._metadata_path, 'r', encoding='utf-8'))
            cache_time = meta.get('build_time', 0)
        except Exception:
            return False

        if self.data_dir.exists():
            for fname in os.listdir(str(self.data_dir)):
                if fname.endswith('.json'):
                    if os.path.getmtime(str(self.data_dir / fname)) > cache_time:
                        print("  检测到档案 JSON 更新，将重建索引。")
                        return False
        return True

    def _save_all_caches(self):
        """保存所有索引到缓存"""
        print("  保存缓存...")
        import faiss

        # BM25
        with open(self._bm25_cache_path, 'wb') as f:
            pickle.dump({
                'records': self.records,
                'tokenized_corpus': self.tokenized_corpus,
                'bm25': self._bm25,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    BM25 索引 → {self._bm25_cache_path}")

        # FAISS (C++ 底层不支持中文路径，切到缓存目录用相对路径)
        old_cwd = os.getcwd()
        os.chdir(str(self.cache_dir))
        faiss.write_index(self._faiss_index, self._faiss_cache_path.name)
        os.chdir(old_cwd)
        print(f"    FAISS 索引 → {self._faiss_cache_path}")

        # 实体
        with open(self._entities_cache_path, 'wb') as f:
            pickle.dump({
                'doc_entities': self.doc_entities,
                'entity_to_docs': dict(self.entity_to_docs),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    实体索引 → {self._entities_cache_path}")

        # 元信息
        meta = {
            'build_time': time.time(),
            'build_time_iso': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_records': len(self.records),
            'model_name': self.model_name,
            'data_dir': str(self.data_dir),
        }
        with open(self._metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"    元信息 → {self._metadata_path}")

    def _load_all_caches(self):
        """从缓存加载全部索引"""
        import faiss

        # BM25
        with open(self._bm25_cache_path, 'rb') as f:
            data = pickle.load(f)
        self.records = data['records']
        self.tokenized_corpus = data['tokenized_corpus']
        self._bm25 = data['bm25']

        # FAISS (C++ 底层不支持中文路径，切到缓存目录用相对路径)
        old_cwd = os.getcwd()
        os.chdir(str(self.cache_dir))
        self._faiss_index = faiss.read_index(self._faiss_cache_path.name)
        os.chdir(old_cwd)

        # 实体
        with open(self._entities_cache_path, 'rb') as f:
            data = pickle.load(f)
        self.doc_entities = data['doc_entities']
        raw = data.get('entity_to_docs', {})
        self.entity_to_docs = defaultdict(lambda: defaultdict(set))
        for etype, ent_map in raw.items():
            for ent, idx_set in ent_map.items():
                self.entity_to_docs[etype][ent] = idx_set

        # 预热 embedding 模型，避免首次检索时等待
        print("    预加载 embedding 模型...")
        self._get_embedding_model()
        print("    模型就绪")


# ============================================================
# 5. CLI 交互入口
# ============================================================

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # 解析命令行参数
    data_format = "refined"  # 默认使用 refined（现代文摘要）
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("filtered", "old", "文言文", "原始"):
            data_format = "filtered"
        elif arg in ("refined", "new", "现代文", "摘要"):
            data_format = "refined"
        else:
            print(f"未知参数 '{sys.argv[1]}'，使用默认 refined")
            print("可选: refined (现代文摘要, 默认) | filtered (文言文原文)")

    fmt_label = "现代文摘要 (refined)" if data_format == "refined" else "文言文原文 (filtered)"

    print("=" * 60)
    print("清史档案混合检索系统 v3")
    print(f"数据格式: {fmt_label}")
    print("管线: 查询处理 → BM25+向量 双通道并行 → RRF融合")
    print("      → 关系加权重排序(实体共现/机构隶属/时序/地理)")
    print("=" * 60)

    retriever = ArchiveRetriever(data_format=data_format)
    retriever.load_or_build()

    print("\n" + "=" * 60)
    print(f"已就绪，共 {len(retriever)} 条档案记录。")
    print("输入清史问题（q 退出）:")
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

        results = retriever.search(query, top_k=5, verbose=True)
        if not results:
            print("\n（未找到相关档案记录）")
            continue

        print("\n" + "─" * 60)
        print(f"【检索结果 Top-{len(results)}】")
        print("─" * 60)
        for r in results:
            print(retriever.format_result(r))
            print("─" * 60)
        print()


if __name__ == "__main__":
    main()
