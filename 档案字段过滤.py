#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
档案JSON字段过滤脚本
用于过滤档案JSON文件，只保留指定字段：
- 年号
- 题名
- 内容提要
- 责任者
- 责任时间（包括"责任时间"、"责任时间(中曆)"等变体）

输出保存到新的目录：数据基础/档案_过滤/

使用说明：
1. 在VS Code中打开此脚本，确保使用配置好的Python环境（base环境）
2. 在终端中切换到项目目录：cd "项目"
3. 运行脚本：python 档案字段过滤.py
4. 过滤后的文件将保存在"数据基础/档案_过滤/"目录中
"""

import os
import json
from pathlib import Path

# ==================== 配置 ====================
# 注意：此脚本假设在"项目"目录下运行
INPUT_DIR = Path("数据基础/档案/")      # 原始档案目录
OUTPUT_DIR = Path("数据基础/档案_过滤/") # 过滤后输出目录

# 需要保留的字段（支持多种可能的字段名）
RETAIN_FIELDS = {
    "年号",  # 主要字段
    "題名", "题名",  # 繁体/简体
    "內容提要", "内容提要",  # 繁体/简体
    "責任者", "责任者",  # 繁体/简体
    "責任時間", "责任时间",  # 基本字段
    "責任時間(中曆)", "责任时间(中曆)",  # 带括号变体
    "責任時間(西曆)", "责任时间(西曆)",  # 西历时间
}

# ==================== 函数定义 ====================
def filter_json_file(input_path, output_path):
    """过滤单个JSON文件，只保留指定字段"""
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保数据是列表
        if not isinstance(data, list):
            print(f"警告: {input_path.name} 数据不是列表格式，跳过")
            return False

        filtered_data = []
        for item in data:
            if not isinstance(item, dict):
                # 非字典项直接保留
                filtered_data.append(item)
                continue

            # 创建新对象，只保留需要的字段
            new_item = {}
            for key, value in item.items():
                # 检查字段是否匹配需要保留的字段
                if key in RETAIN_FIELDS:
                    new_item[key] = value
                # 特殊处理：责任时间可能有多个变体，都保留
                elif "责任" in key and "时间" in key:
                    new_item[key] = value

            filtered_data.append(new_item)

        # 保存过滤后的数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 已处理: {input_path.name} -> {output_path.name}")
        return True

    except Exception as e:
        print(f"✗ 处理失败 {input_path.name}: {e}")
        return False

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("档案JSON字段过滤脚本")
    print("=" * 60)

    # 检查输入目录是否存在
    if not INPUT_DIR.exists():
        print(f"错误: 输入目录不存在: {INPUT_DIR.absolute()}")
        print("请确保在'项目'目录下运行此脚本")
        return

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有JSON文件
    json_files = list(INPUT_DIR.glob("*.json"))
    print(f"找到 {len(json_files)} 个档案JSON文件")

    if not json_files:
        print(f"错误: 在 {INPUT_DIR.absolute()} 中未找到JSON文件")
        return

    # 处理每个文件
    success_count = 0
    for json_file in json_files:
        output_file = OUTPUT_DIR / json_file.name
        if filter_json_file(json_file, output_file):
            success_count += 1

    print(f"\n" + "=" * 60)
    print(f"处理完成: {success_count}/{len(json_files)} 个文件成功")
    print(f"过滤后的文件保存在: {OUTPUT_DIR.absolute()}")
    print("=" * 60)

# ==================== 脚本入口 ====================
if __name__ == "__main__":
    main()