import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 强制开启离线模式，防止联网检查
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class SmartHistoricalRetriever:
    def __init__(self, file_paths, index_dir="./search_index", model_name='shibing624/text2vec-base-chinese'):
        # 1. 优先定义所有路径属性（防止 build_index 找不到变量）
        self.file_paths = file_paths
        self.index_dir = index_dir
        self.index_file = os.path.join(index_dir, "vector.index")
        self.data_file = os.path.join(index_dir, "metadata.pkl")
        self.data = []
        self.index = None

        # 2. 定位本地模型路径 (支持 Windows/Linux 双平台)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 优先查找项目本地的 models--shibing624--text2vec-base-chinese 目录
        local_model_dir = os.path.join(script_dir, "models--shibing624--text2vec-base-chinese")
        hf_snapshot_dir = os.path.join(local_model_dir, "snapshots")

        model_found = False
        if os.path.exists(hf_snapshot_dir):
            snapshots = os.listdir(hf_snapshot_dir)
            if snapshots:
                self.model_path = os.path.join(hf_snapshot_dir, snapshots[0])
                print(f"检测到本地模型 (项目内): {self.model_path}")
                model_found = True

        if not model_found:
            # 备选：查找 HuggingFace 标准缓存路径 (Linux)
            for base_cache in [
                os.path.expanduser("~/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots"),
            ]:
                if os.path.exists(base_cache):
                    snapshots = os.listdir(base_cache)
                    if snapshots:
                        self.model_path = os.path.join(base_cache, snapshots[0])
                        print(f"检测到本地模型 (HF缓存): {self.model_path}")
                        model_found = True
                        break

        if not model_found:
            self.model_path = model_name
            print(f"未找到本地模型，尝试使用模型名: {model_name}")

        # 3. 加载模型
        print(f"正在加载语义模型...")
        self.model = SentenceTransformer(self.model_path)

        # 4. 执行索引逻辑
        if self._is_index_valid():
            self.load_index()
        else:
            print("未检测到有效索引，开始初始化并构建（此过程仅需执行一次）...")
            self.build_index()

    def _is_index_valid(self):
        """检查本地索引文件是否存在"""
        return os.path.exists(self.index_file) and os.path.exists(self.data_file)

    def build_index(self):
        """读取数据，计算向量，并保存到本地"""
        all_items = []
        for path in self.file_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        era = os.path.basename(path).split('朝')[0]
                        if not isinstance(content, list): continue

                        for idx, item in enumerate(content):
                            # 鲁棒性检查：确保核心字段存在
                            if isinstance(item, dict) and 'refined_data' in item and 'micro_summary' in item['refined_data']:
                                item['era_source'] = era
                                all_items.append(item)
                except Exception as e:
                    print(f"解析文件 {path} 出错: {e}")
        
        if not all_items:
            print("错误: 未能加载任何有效数据。")
            return

        self.data = all_items
        summaries = [item['refined_data']['micro_summary'] for item in self.data]
        
        print(f"成功加载 {len(summaries)} 条记录，正在进行向量化计算...")
        embeddings = self.model.encode(summaries, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # 创建 FAISS 索引
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension) 
        self.index.add(embeddings)

        # 确保目录存在并保存
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"索引已保存至 {self.index_dir}")

    def load_index(self):
        """直接加载缓存索引"""
        print("正在从本地秒级加载缓存索引...")
        self.index = faiss.read_index(self.index_file)
        with open(self.data_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"加载成功，共 {len(self.data)} 条数据。")

    def _calculate_entity_score(self, query, entry):
        """元数据增强评分逻辑"""
        score = 0
        refined = entry.get('refined_data', {})
        raw_entities = []
        for key in ['entities_people', 'entities_geo', 'entities_office', 'event_type']:
            val = refined.get(key, [])
            if isinstance(val, list): raw_entities.extend(val)
            elif val: raw_entities.append(val)

        for entity in raw_entities:
            if isinstance(entity, str) and entity and entity in query:
                score += 0.15
        return score

    def search(self, query, max_k=10, min_score=0.65, drop_threshold=0.1):
        """
        动态截断检索
        :param query: 搜索词
        :param max_k: 最多允许返回的条数（上限）
        :param min_score: 绝对及格线，低于这个分数的直接不要
        :param drop_threshold: 断崖式落差阈值，相邻两名分差大于此值则截断
        """
        query_vec = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)

        # 1. 召回足够多的候选样本 (比如扩大到 max_k 的 3倍)
        candidate_count = max_k * 3
        similarities, indices = self.index.search(query_vec, candidate_count)

        results =[]
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1 or idx >= len(self.data): continue
            entry = self.data[idx]
            # 加上你写的实体加分逻辑
            meta_boost = self._calculate_entity_score(query, entry)
            results.append({
                "score": round(float(sim + meta_boost), 4),
                "data": entry
            })
        
        # 2. 按最终总分从高到低重排序
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # ==========================================
        # 3. 核心：动态分数断层截断机制 (Dynamic Gap Truncation)
        # ==========================================
        valid_results =[]
        if not results:
            return valid_results
            
        # 第一名如果满足最低及格线，无条件加入
        if results[0]['score'] >= min_score:
            valid_results.append(results[0])
            
        for i in range(1, len(results)):
            # 条件 A：达到最大返回数量上限，强制停止
            if len(valid_results) >= max_k:
                break
                
            current_score = results[i]['score']
            previous_score = results[i-1]['score']
            
            # 条件 B：绝对分数不达标，后续全是噪音，直接停止
            if current_score < min_score:
                break
                
            # 条件 C：检测到“断崖式下跌” (Elbow Method)
            score_drop = previous_score - current_score
            if score_drop > drop_threshold:
                print(f"  [系统提示] 检测到分数断崖 (降幅 {score_drop:.4f})，过滤后续噪音，实际返回 {len(valid_results)} 条。")
                break
                
            # 通过所有考核，加入返回列表
            valid_results.append(results[i])
            
        return valid_results
# --- 执行主程序 ---

# --- 执行主程序 ---

if __name__ == "__main__":
    # 请确保路径正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_list = [
        os.path.join(script_dir, "data", "光绪朝结构化数据摘要.json"),
        os.path.join(script_dir, "data", "乾隆朝结构化数据摘要_aligned.json"),
        os.path.join(script_dir, "data", "康熙朝结构化数据摘要_aligned.json")
    ]

    retriever = SmartHistoricalRetriever(file_list)

    while True:
        user_query = input("\n请输入搜索词 (输入q退出): ")
        if user_query.lower() == 'q': break
        
        # 使用动态检索：最多返回 8 条，分差 > 0.05 就截断
        hits = retriever.search(user_query, max_k=8, min_score=0.60, drop_threshold=0.1)
        
        if not hits:
            print("没有找到高度相关的结果，请尝试更换搜索词。")
            continue

        for i, hit in enumerate(hits):
            d = hit['data']
            # 安全获取字段
            era = d.get('era_source', '未知')
            nianhao = d.get('nianhao', '无年号')
            date = d.get('date', '未知日期')
            summary = d.get('refined_data', {}).get('micro_summary', '无摘要')
            
            print(f"\n[Top {i+1}] 得分: {hit['score']} | {era} {nianhao} {date}")
            print(f"摘要: {summary}")
            people = d.get('refined_data', {}).get('entities_people',[])
            if people: 
                # 兼容 people 是列表或字符串的情况
                people_str = ', '.join(people) if isinstance(people, list) else str(people)
                print(f"涉及人物: {people_str}")