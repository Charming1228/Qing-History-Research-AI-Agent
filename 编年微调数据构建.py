import asyncio
import pandas as pd
import json
import os
import re
from tqdm.asyncio import tqdm  # 异步进度条
from openai import AsyncOpenAI # 注意是 AsyncOpenAI

# 1. 配置
client = AsyncOpenAI(
    base_url=os.environ.get("MOARK_BASE_URL", "https://api.moark.com/v1"),
    api_key=os.environ["MOARK_API_KEY"]
)
MODEL_NAME = "DeepSeek-V3.2"
INPUT_CSV = "/Users/zhouyijing/Desktop/数据科学实践/第三组/数据基础/编年/光绪朝结构化数据.csv"
OUTPUT_JSON = "hardcore_history_training_data.json"

# 限制并发数（比如同时发 5 个请求），防止被 API 封禁或报错
sem = asyncio.Semaphore(5) 

async def get_hardcore_rid_async(raw_text):
    system_prompt = """你是一个清史文献处理专家。请解析下述史料，剥离套话，提取事实要素。
输出必须为严格的 JSON 格式，包含以下五个字段：
1. entities_people: 仅提取文中出现的【人名】（如：沈桂芬）。若无则空列表。
2. entities_office: 仅提取文中出现的【职衔/机构】（如：兵部尚书、军机处）。若无则空列表。
3. entities_geo: 提取提到的【地名】（如：台北府、基隆）。若无则空列表。
4. event_type: 判定事件类型（如：人事任免、军事防御、洋务建设、灾荒赈济、外交事务）。
5. micro_summary: 200字以内的白话文摘要。要求：客观、学术，包含时间、主体、动作、结果。
"""
    async with sem: # 使用信号量控制并发
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"原始史料：\n{raw_text}"}
                ],
                temperature=0.1,
                timeout=60
            )
            content = response.choices[0].message.content
            clean_json_str = re.sub(r'```json\s*|```', '', content).strip()
            return json.loads(clean_json_str), raw_text
        except Exception as e:
            return None, raw_text

async def main():
    df = pd.read_csv(INPUT_CSV, encoding="gb18030", encoding_errors="replace")
    df['事件正文'] = df['事件正文'].fillna("")
    
    # 过滤掉太短的数据
    tasks_data = df[df['事件正文'].str.len() > 5]['事件正文'].tolist()
    
    # 创建异步任务池
    tasks = [get_hardcore_rid_async(text) for text in tasks_data]
    
    training_data = []
    # 使用 tqdm 显示异步进度
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        refined_json, raw_text = await coro
        if refined_json:
            training_data.append({
                "conversations": [
                    {"from": "human", "value": f"请对以下清史文献执行结构化精炼：\n{raw_text}"},
                    {"from": "gpt", "value": json.dumps(refined_json, ensure_ascii=False)}
                ],
                "system": "你是一个专业的清史文献解析助手。"
            })
            
            # 定期保存
            if len(training_data) % 20 == 0:
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(training_data, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print("✅ 任务全部完成！")

if __name__ == "__main__":
    asyncio.run(main())
