import re
import pandas as pd
import os
import uuid

def normalize_marker(text):
    """统一全角/半角括号，方便在正文中精确匹配引用标记"""
    return text.replace('（', '(').replace('）', ')')

def parse_historical_md_to_list(file_path):
    """
    修改版解析函数：不直接导出CSV，而是返回记录列表。
    保留了所有原始的正则和提取逻辑。
    """
    records = []
    current_record = None

    # === 全局状态/记忆变量 ===
    is_content_started = False
    context_emperor = "光绪"
    context_reign_year = ""
    context_sexagenary_year = ""
    context_base_greg_year = 0
    current_actual_greg_year = 0
    last_greg_month = 0
    context_lunar_month = ""
    context_is_leap = "FALSE"

    current_text_mode = "event"  
    active_footnote_record = None 

    # 宣统朝分界点
    XUANTONG_START = "1909-01-22"

    # === 正则表达式预编译 (保持原样) ===
    re_year_line = re.compile(r'^\s*光绪(.*?年)\s+([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])\s+公元(\d{4})年\s*$')
    
    re_date_line = re.compile(
        r'^\s*(?:(闰)?(正|一|二|三|四|五|六|七|八|九|十|十一|十二)月)?'
        r'(初[一二三四五六七八九十]|十[一二三四五六七八九]?|二十[一二三四五六七八九]?|廿[一二三四五六七八九]?|三十)日?' 
        r'([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])?\s*'
        r'[（\(]\s*(?:(?:公元)?\d{4}\s*年\s*)?(\d{1,2}\s*月\s*\d{1,2}\s*日?)\s*[）\)]'
    )
    
    re_greg_date = re.compile(r'(\d+)\s*月\s*(\d+)\s*日?')
    re_footnote_line = re.compile(r'^\s*([\(（]\d+[\)）]|[①-㊿])\s*(.*)$')
    re_page_number = re.compile(r'^\s*[-—_]*\s*\d+\s*[-—_]*\s*$')

    if not os.path.exists(file_path):
        print(f"警告：文件 {file_path} 不存在，跳过。")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue
        if re_page_number.match(line): continue

        year_match = re_year_line.match(line)
        if year_match:
            is_content_started = True
            current_text_mode = "event" 
            if current_record:
                records.append(current_record)
                current_record = None

            context_reign_year = "光绪" + year_match.group(1)
            context_sexagenary_year = year_match.group(2)
            context_base_greg_year = int(year_match.group(3))
            current_actual_greg_year = context_base_greg_year
            last_greg_month = 0
            context_lunar_month = ""
            context_is_leap = "FALSE"
            continue

        if not is_content_started: continue

        date_match = re_date_line.match(line)
        
        is_fake_date = False
        if date_match and current_record and current_record["事件正文"]:
            stripped_text = current_record["事件正文"].rstrip()
            if stripped_text:
                last_char = stripped_text[-1]
                hanging_chars = "，、；：,;:于在自至与和及以并"
                if last_char in hanging_chars:
                    is_fake_date = True 

        if date_match and not is_fake_date:
            current_text_mode = "event" 
            if current_record:
                records.append(current_record)

            leap_str = date_match.group(1)
            month_str = date_match.group(2)
            day_str = date_match.group(3) + "日"
            gz_day_str = date_match.group(4) if date_match.group(4) else ""
            orig_greg_str = date_match.group(5)

            if month_str:
                context_lunar_month = month_str + "月"
                context_is_leap = "TRUE" if leap_str else "FALSE"

            std_greg_date = ""
            greg_m_match = re_greg_date.search(orig_greg_str)
            if greg_m_match:
                g_month = int(greg_m_match.group(1))
                g_day = int(greg_m_match.group(2))
                if last_greg_month > 8 and g_month < 4:
                    current_actual_greg_year += 1
                last_greg_month = g_month
                std_greg_date = f"{current_actual_greg_year}-{g_month:02d}-{g_day:02d}"

            # --- 朝代修正核心逻辑 ---
            target_emperor = context_emperor
            target_reign_year = context_reign_year
            
            if std_greg_date >= XUANTONG_START:
                target_emperor = "宣统"
                if "1909" in std_greg_date: target_reign_year = "宣统元年"
                elif "1910" in std_greg_date: target_reign_year = "宣统二年"
                elif "1911" in std_greg_date or "1912" in std_greg_date: target_reign_year = "宣统三年"

            current_record = {
                "帝号": target_emperor,
                "年号纪年": target_reign_year,
                "干支纪年": context_sexagenary_year,
                "农历月": context_lunar_month,
                "是否闰月": context_is_leap,
                "农历日": day_str,
                "干支纪日": gz_day_str,
                "原始公历日期": orig_greg_str,
                "标准公历日期": std_greg_date,
                "事件正文": "",
                "其他": "" 
            }

            remaining_text = line[date_match.end():].strip()
            if remaining_text:
                current_record["事件正文"] = remaining_text
            continue

        footnote_match = re_footnote_line.match(line)
        if footnote_match:
            marker = footnote_match.group(1)  
            norm_marker = normalize_marker(marker) 
            found_target_record = False
            search_list =[current_record] + records[::-1] 
            
            for rec in search_list[:30]:  
                if rec and rec["事件正文"]:
                    norm_event_text = normalize_marker(rec["事件正文"])
                    if norm_marker in norm_event_text:
                        if rec["其他"]: rec["其他"] += "\n" + line
                        else: rec["其他"] = line
                        current_text_mode = "footnote"
                        active_footnote_record = rec
                        found_target_record = True
                        break 
            
            if not found_target_record and current_record:
                if current_record["其他"]: current_record["其他"] += "\n" + line
                else: current_record["其他"] = line
                current_text_mode = "footnote"
                active_footnote_record = current_record
            continue 

        if is_fake_date: current_text_mode = "event"

        if current_text_mode == "footnote" and active_footnote_record is not None:
            last_footnote_char = ""
            if active_footnote_record["其他"]:
                stripped_footnote = active_footnote_record["Other"].rstrip() if "Other" in active_footnote_record else active_footnote_record["其他"].rstrip()
                if stripped_footnote: last_footnote_char = stripped_footnote[-1]
            if last_footnote_char in "。！？.”）)】》":
                current_text_mode = "event"

        if current_text_mode == "footnote" and active_footnote_record is not None:
            if active_footnote_record["其他"]: active_footnote_record["其他"] += "\n" + line
            else: active_footnote_record["其他"] = line
        else:
            if current_record:
                if current_record["事件正文"]:
                    last_char = current_record["事件正文"].rstrip()[-1] if current_record["事件正文"].rstrip() else ""
                    if last_char in "。！？；：”’）)】》":
                        current_record["事件正文"] += "\n" + line 
                    else:
                        current_record["事件正文"] += line
                else:
                    current_record["事件正文"] = line

    if current_record:
        records.append(current_record)
    
    return records

if __name__ == "__main__":
    # 输入文件
    input_file_up = "清史编年11.光绪朝.上.md"
    input_file_down = "清史编年12.光绪朝.下.md"
    
    # 导出文件
    output_guangxu = "光绪朝结构化数据_最终版.csv"
    output_xuantong = "宣统朝结构化数据_最终版.csv"

    print("开始处理《光绪朝·上》...")
    records_up = parse_historical_md_to_list(input_file_up)
    
    print("开始处理《光绪朝·下》...")
    records_down = parse_historical_md_to_list(input_file_down)

    # 合并所有提取到的记录
    all_records = records_up + records_down

    # 根据“帝号”字段进行分离
    guangxu_records = [r for r in all_records if r["帝号"] == "光绪"]
    xuantong_records = [r for r in all_records if r["帝号"] == "宣统"]

    # 导出光绪 CSV
    if guangxu_records:
        df_gx = pd.DataFrame(guangxu_records)
        df_gx.to_csv(output_guangxu, index=False, encoding='utf-8-sig')
        print(f"✅ 光绪朝处理完成：共 {len(guangxu_records)} 条，保存至 {output_guangxu}")

    # 导出宣统 CSV
    if xuantong_records:
        df_xt = pd.DataFrame(xuantong_records)
        df_xt.to_csv(output_xuantong, index=False, encoding='utf-8-sig')
        print(f"✅ 宣统朝处理完成：共 {len(xuantong_records)} 条，保存至 {output_xuantong}")