#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
档案微调数据构建脚本
基于档案JSON数据构建文言文白话化微调训练数据
参考：编年微调数据构建.py

输入：数据基础/档案/下的JSON文件（或过滤后的版本）
输出：archive_training_data.json（与hardcore_history_training_data.json类似格式）
"""

import asyncio
import pandas as pd
import json
import os
import re
import random
import time
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI  # 注意是 AsyncOpenAI

# ==================== 配置部分 ====================
# 用户需要填写API配置
client = AsyncOpenAI(
    base_url=os.environ.get("MOARK_BASE_URL", "https://api.moark.com/v1"),
    api_key=os.environ["MOARK_API_KEY"]
)
MODEL_NAME = "GLM-5"  # 可修改为其他模型

# ==================== 运行模式配置 ====================
TEST_MODE = False  # 设为True进入快速测试模式
TEST_DYNASTY = "道光"  # 测试朝代
MAX_TEST_RECORDS = 20  # 测试条数

# 跨年号均衡采样模式（TEST_MODE=False时生效）
SAMPLE_TOTAL = 3000  # 采样总目标

# 输入输出路径
ARCHIVE_DIR = Path("数据基础/档案_过滤/")
OUTPUT_JSON = "archive_training_data.json"

# 并发控制
MAX_CONCURRENCY = 15  # 最大并发请求数
sem = asyncio.Semaphore(MAX_CONCURRENCY)

# ==================== 数据准备函数 ====================
def load_random_sample(file_path, sample_size):
    """
    从单个档案JSON文件中随机抽取指定数量的记录。
    返回列表，每个元素是 {raw_text, metadata}。
    metadata.index 保留在原文件中的原始位置。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"警告: {file_path.name} 数据不是列表格式，跳过")
        return []

    total = len(data)
    actual_size = min(sample_size, total)
    selected_indices = sorted(random.sample(range(total), actual_size))

    dynasty = file_path.stem
    samples = []

    for idx in selected_indices:
        item = data[idx]
        if not isinstance(item, dict):
            continue

        nianhao = item.get("年号") or item.get("年号", "")
        title = item.get("题名") or item.get("題名", "")
        summary = item.get("内容提要") or item.get("內容提要", "")
        responsible = item.get("责任者") or item.get("責任者", "")
        time_keys = [k for k in item.keys() if ("责任" in k or "責任" in k) and ("时间" in k or "時間" in k)]
        time_str = item.get(time_keys[0], "") if time_keys else ""

        raw_text_parts = []
        if title:
            raw_text_parts.append(f"题名：{title}")
        if summary:
            raw_text_parts.append(f"内容提要：{summary}")
        if responsible:
            raw_text_parts.append(f"责任者：{responsible}")
        if time_str:
            raw_text_parts.append(f"责任时间：{time_str}")

        raw_text = "\n".join(raw_text_parts)

        samples.append({
            "raw_text": raw_text,
            "metadata": {
                "dynasty": dynasty,
                "file": file_path.name,
                "index": idx,
                "nianhao": nianhao,
                "title": title,
                "responsible": responsible,
                "time_str": time_str
            }
        })

    return samples

# ==================== API调用函数 ====================
async def refine_archive_text_async(raw_text, seq_id):
    """
    调用API对档案文本进行精炼处理（含自动重试）
    返回：(精炼后的JSON字典, seq_id)
    """
    system_prompt = """你是一个清史档案处理专家。请解析下述档案记录，剥离套话，提取事实要素。

输出必须为严格的 JSON 格式，包含以下六个字段：
1. entities_people: 仅提取文中出现的【人名】（如：阮元）。若无则空列表。
2. entities_office: 仅提取文中出现的【职衔/机构】（如：两广总督、廂藍旗蒙古）。若无则空列表。
3. entities_geo: 提取提到的【地名】（如：广东省、广西省）。若无则空列表。
4. document_type: 判定档案类型（如：档册、奏折、移会、题本、移付、册、摺件等）。
5. event_type: 判定事件类型（如：人事任免、军事防务、财政税务、司法刑案、礼仪典制、工程营造、外交事务、灾荒赈济等）。
6. modern_summary: 200字以内的现代白话文摘要。要求：客观、学术，包含时间、主体、动作、结果。

示例输出格式：
{
  "entities_people": ["人名1", "人名2"],
  "entities_office": ["职衔/机构1", "职衔/机构2"],
  "entities_geo": ["地名1", "地名2"],
  "document_type": "档案类型",
  "event_type": "事件类型",
  "modern_summary": "白话文摘要内容"
}"""

    for attempt in range(3):  # 最多重试3次
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"档案记录：\n{raw_text}"}
                    ],
                    temperature=0.1,
                    timeout=60
                )
                content = response.choices[0].message.content

                # 尝试多种方式提取JSON
                def extract_json(text):
                    # 方式1: 直接解析
                    text = text.strip()
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
                    # 方式2: 提取```json ... ``` 块
                    m = re.search(r'```(?:json)?\s*\n?({.*?})\s*\n?```', text, re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group(1))
                        except json.JSONDecodeError:
                            pass
                    # 方式3: 提取第一个 { ... } 顶层JSON对象
                    brace_depth = 0
                    start = -1
                    for i, ch in enumerate(text):
                        if ch == '{':
                            if start == -1:
                                start = i
                            brace_depth += 1
                        elif ch == '}':
                            brace_depth -= 1
                            if brace_depth == 0 and start != -1:
                                try:
                                    return json.loads(text[start:i+1])
                                except json.JSONDecodeError:
                                    start = -1
                    raise ValueError("无法从API响应中提取有效JSON")

                refined = extract_json(content)
                return refined, seq_id

        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 3  # 第1次重试等3秒，第2次等6秒
                print(f"  [重试{attempt+1}/2] seq={seq_id}: {type(e).__name__}，等待{wait}秒")
                await asyncio.sleep(wait)
            else:
                print(f"  [失败] seq={seq_id}: {e}")

    return None, seq_id

# ==================== 跨年号采样分配函数 ====================
def allocate_records(archive_dir, total_target, min_per_dynasty=50):
    """
    扫描所有年号档案，按文件大小比例分配采样额度。
    确保每个年号至少有min_per_dynasty条。
    返回 { dynasty: allocation_count } 字典
    """
    counts = {}
    for f in sorted(archive_dir.glob("*.json")):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            counts[f.stem] = len(data)
        except Exception as e:
            print(f"扫描文件 {f.name} 失败: {e}")
            counts[f.stem] = 0

    total = sum(counts.values())
    if total == 0:
        return {}

    # 第一轮：按比例分配
    raw_alloc = {}
    for dynasty, count in sorted(counts.items()):
        alloc = max(min_per_dynasty, round(total_target * count / total))
        raw_alloc[dynasty] = min(alloc, count)  # 不超过实际总数

    # 第二轮：如果总额超出目标，等比缩减（但不低于min_per_dynasty）
    allocated_total = sum(raw_alloc.values())
    if allocated_total > total_target:
        # 找出超过min的可以缩减的部分
        excess = allocated_total - total_target
        reducible = {d: raw_alloc[d] - min_per_dynasty for d in raw_alloc if raw_alloc[d] > min_per_dynasty}
        reducible_total = sum(reducible.values())
        if reducible_total > 0:
            for d in reducible:
                cut = round(excess * reducible[d] / reducible_total)
                raw_alloc[d] = max(min_per_dynasty, raw_alloc[d] - cut)

    print("年号采样分配计划:")
    for d, c in sorted(raw_alloc.items()):
        print(f"  {d}: {c}条 (共{counts[d]}条)")
    print(f"  合计: {sum(raw_alloc.values())}条")

    return raw_alloc


# ==================== 主函数 ====================
async def main():
    if TEST_MODE:
        output_file = f"archive_training_data_test_{TEST_DYNASTY}.json"
        print(f"测试模式: {TEST_DYNASTY} (最多{MAX_TEST_RECORDS}条)")

        json_path = ARCHIVE_DIR / f"{TEST_DYNASTY}.json"
        samples = load_random_sample(json_path, MAX_TEST_RECORDS)
        if not samples:
            print("未加载到有效数据，退出")
            return

        tasks = [refine_archive_text_async(s["raw_text"], i) for i, s in enumerate(samples)]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=TEST_DYNASTY, unit="条"):
            refined_json, seq_id = await coro
            if refined_json:
                results.append((seq_id, refined_json))

        results.sort(key=lambda x: x[0])
        training_data = []
        for seq_id, refined_json in results:
            rtxt = samples[seq_id]["raw_text"]
            training_data.append({
                "conversations": [
                    {"from": "human", "value": f"请对以下清史档案记录执行结构化精炼：\n{rtxt}"},
                    {"from": "gpt", "value": json.dumps(refined_json, ensure_ascii=False)}
                ],
                "system": "你是一个专业的清史档案解析助手。"
            })
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"[完成] 共生成 {len(training_data)} 条，输出: {output_file}")
        return

    # ==== 采样模式 ====
    if SAMPLE_TOTAL:
        output_file = f"archive_training_data_sample_{SAMPLE_TOTAL}.json"
        print(f"跨年号采样: 目标{SAMPLE_TOTAL}条, 并发{MAX_CONCURRENCY}")
    else:
        output_file = OUTPUT_JSON

    # 获取分配计划
    allocations = allocate_records(ARCHIVE_DIR, SAMPLE_TOTAL)
    if not allocations:
        print("分配失败，程序退出")
        return

    total_target = sum(allocations.values())
    all_training_data = []
    overall_start = time.time()
    total_ok = 0

    # 逐朝处理
    dynasty_list = sorted(allocations.items())
    for dyn_idx, (dynasty, dyn_count) in enumerate(dynasty_list, 1):
        if dyn_count == 0:
            continue

        dyn_start = time.time()
        print(f"\n{'='*50}")
        print(f"[{dyn_idx}/{len(dynasty_list)}] {dynasty} - {dyn_count}条")
        print(f"{'='*50}")

        # 随机抽取
        json_path = ARCHIVE_DIR / f"{dynasty}.json"
        samples = load_random_sample(json_path, dyn_count)
        if not samples:
            print(f"  {dynasty}: 未抽到有效记录，跳过")
            continue
        print(f"  加载 {len(samples)} 条")

        # 并发处理
        tasks = [refine_archive_text_async(s["raw_text"], i) for i, s in enumerate(samples)]
        dyn_results = []

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                         desc=f"{dynasty}", unit="条"):
            refined_json, seq_id = await coro
            if refined_json:
                dyn_results.append((seq_id, refined_json))

                # 每成功处理10条，保存一次检查点
                if len(dyn_results) % 10 == 0:
                    temp_sorted = sorted(dyn_results, key=lambda x: x[0])
                    temp_data = []
                    for sid, rjson in temp_sorted:
                        rtxt = samples[sid]["raw_text"]
                        temp_data.append({
                            "conversations": [
                                {"from": "human", "value": f"请对以下清史档案记录执行结构化精炼：\n{rtxt}"},
                                {"from": "gpt", "value": json.dumps(rjson, ensure_ascii=False)}
                            ],
                            "system": "你是一个专业的清史档案解析助手。"
                        })
                    # 合并已有的跨年号数据
                    full_data = all_training_data + temp_data
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(full_data, f, ensure_ascii=False, indent=2)

        # 按原始顺序排序并构建
        dyn_results.sort(key=lambda x: x[0])
        dyn_ok = len(dyn_results)
        total_ok += dyn_ok

        for seq_id, refined_json in dyn_results:
            rtxt = samples[seq_id]["raw_text"]
            all_training_data.append({
                "conversations": [
                    {"from": "human", "value": f"请对以下清史档案记录执行结构化精炼：\n{rtxt}"},
                    {"from": "gpt", "value": json.dumps(refined_json, ensure_ascii=False)}
                ],
                "system": "你是一个专业的清史档案解析助手。"
            })

        # 保存（每个年号处理完后写入一次，覆盖检查点）
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_training_data, f, ensure_ascii=False, indent=2)

        # 进度统计
        dyn_elapsed = time.time() - dyn_start
        overall_elapsed = time.time() - overall_start
        success_rate = dyn_ok / max(dyn_count, 1) * 100
        eta = (overall_elapsed / total_ok * (total_target - total_ok)) if total_ok > 0 else 0

        print(f"  [+] {dynasty} 完成: {dyn_ok}/{dyn_count}条成功 ({success_rate:.0f}%)")
        print(f"  [时间] {dynasty}用时: {dyn_elapsed/60:.1f}分 | 总用时: {overall_elapsed/60:.1f}分")
        print(f"  [进度] 总进度: {total_ok}/{total_target} ({total_ok/total_target*100:.1f}%)")
        if eta > 0:
            print(f"  [预计] 剩余: {eta/60:.1f}分 ({eta/3600:.1f}小时)")

    # 最终保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_training_data, f, ensure_ascii=False, indent=2)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*50}")
    print(f"全部完成!")
    print(f"  成功: {total_ok}/{total_target} 条")
    print(f"  总用时: {overall_elapsed/60:.1f}分 ({overall_elapsed/3600:.1f}小时)")
    print(f"  输出: {output_file}")

# ==================== 脚本入口 ====================
if __name__ == "__main__":
    # 注意：运行前请确保填写正确的API配置
    print("=" * 60)
    print("档案微调数据构建脚本")
    print("=" * 60)

    if TEST_MODE:
        print(f"[测试模式]")
        print(f"   - 朝代: {TEST_DYNASTY}")
        print(f"   - 记录数: {MAX_TEST_RECORDS}条")
        print(f"   - 输出: archive_training_data_test_{TEST_DYNASTY}.json")
    elif SAMPLE_TOTAL:
        print(f"[跨年号采样模式]")
        print(f"   - 总目标: {SAMPLE_TOTAL}条")
        print(f"   - 并发数: {MAX_CONCURRENCY}")
        print(f"   - 抽取方式: 随机")
        print(f"   - 处理方式: 逐朝顺序处理")
        print(f"   - 输出: archive_training_data_sample_{SAMPLE_TOTAL}.json")
    else:
        print(f"[全量处理模式]")
        print(f"   - 输出: {OUTPUT_JSON}")

    print("注意事项：")
    print("1. 请确保已填写正确的API配置")
    print("2. 输入目录: 数据基础/档案_过滤/")
    print("3. 如需切换模式，请修改文件开头的TEST_MODE/SAMPLE_TOTAL配置")
    print("=" * 60)

    # 运行主函数
    asyncio.run(main())
