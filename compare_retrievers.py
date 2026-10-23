"""
清史档案检索系统 新旧对比脚本
============================================================
对比原始文言文检索器 vs refined 现代文摘要检索器

用法:
    python compare_retrievers.py

输出:
    - 每条查询的并排对比结果
    - 耗时对比
    - 评分统计对比
============================================================
"""

import sys
import io
import time
from archive_retriever import ArchiveRetriever

# 确保 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# 测试查询集（覆盖不同领域）
# ============================================================
TEST_QUERIES = [
    # 军事/海防
    ("乾隆年间福建水师海防", "军事-海防"),
    ("甲午战争北洋水师", "军事-海战"),
    ("八旗驻防边疆军务", "军事-驻防"),

    # 行政/司法
    ("秋审斩监候刑案", "司法-刑案"),
    ("官员贪腐革职议处", "行政-监察"),
    ("科举会试殿试", "行政-科举"),

    # 财政/经济
    ("漕运盐政钱法", "财政-税收"),
    ("捐纳捐输报捐", "财政-捐纳"),

    # 河工/民生
    ("黄河河工堤防赈济", "民生-河工"),
    ("灾赈抚恤蠲免", "民生-赈灾"),

    # 外交/条约
    ("鸦片战争南京条约通商", "外交-条约"),
    ("总理各国事务衙门洋务", "外交-洋务"),

    # 人物
    ("林则徐虎门销烟", "人物-林则徐"),
    ("曾国藩湘军", "人物-曾国藩"),
    ("左宗棠新疆伊犁", "人物-左宗棠"),
]


def main():
    print("=" * 80)
    print("  清史档案检索系统 — 新旧对比测试")
    print("=" * 80)

    # ── 加载两个检索器 ──
    print("\n[1] 加载检索器...")
    t0 = time.time()

    print("\n  旧系统（文言文原文）:")
    r_old = ArchiveRetriever(data_format="filtered")
    r_old.load_or_build()

    print("\n  新系统（refined 现代文摘要）:")
    r_new = ArchiveRetriever(data_format="refined")
    r_new.load_or_build()

    t1 = time.time()
    print(f"\n  加载总耗时: {t1 - t0:.1f}s")

    # ── 逐条对比 ──
    print("\n" + "=" * 80)
    print("  逐条对比测试")
    print("=" * 80)

    total_old_time = 0
    total_new_time = 0

    for query, category in TEST_QUERIES:
        print(f"\n{'─' * 80}")
        print(f"  【查询】: {query}")
        print(f"  【类别】: {category}")

        # 旧系统检索
        t_start = time.time()
        old_results = r_old.search(query, top_k=3, verbose=False)
        t_old = time.time() - t_start
        total_old_time += t_old

        # 新系统检索
        t_start = time.time()
        new_results = r_new.search(query, top_k=3, verbose=False)
        t_new = time.time() - t_start
        total_new_time += t_new

        print(f"  ⏱ 耗时 — 旧: {t_old:.2f}s | 新: {t_new:.2f}s")
        print()

        # 并排展示 Top-3 结果
        max_lines = max(
            len(old_results) if old_results else 0,
            len(new_results) if new_results else 0,
            3
        )

        for i in range(max_lines):
            print(f"  ┌─ Top-{i+1} ──────────────────────────────────────────")

            # 旧系统结果
            if i < len(old_results):
                orr = old_results[i]
                orec = orr.get('record', {})
                o_title = orec.get('题名', '') or orec.get('内容提要', '')[:40] or '(无)'
                o_score = orr['final_score']
                o_bm25 = orr['bm25_score']
                o_vec = orr['vector_score']
                print(f"  │ [旧-文言文] final={o_score:.3f} bm25={o_bm25:.1f} vec={o_vec:.3f}")
                print(f"  │   {o_title[:70]}")
                print(f"  │   朝代: {orec.get('来源朝代', '?')}  时间: {orec.get('责任时间(中历)', orec.get('責任時間(中曆)', '?'))}")
            else:
                print(f"  │ [旧-文言文] (无结果)")

            print(f"  │")

            # 新系统结果
            if i < len(new_results):
                nrr = new_results[i]
                nrec = nrr.get('record', {})
                nrd = nrec.get('refined_data', {})
                n_modern = (nrd.get('modern_summary', '') if isinstance(nrd, dict) else '')[:80]
                n_score = nrr['final_score']
                n_bm25 = nrr['bm25_score']
                n_vec = nrr['vector_score']
                print(f"  │ [新-现代文] final={n_score:.3f} bm25={n_bm25:.1f} vec={n_vec:.3f}")
                print(f"  │   📜 {n_modern}")
                print(f"  │   朝代: {nrec.get('来源朝代', '?')}  时间: {nrec.get('责任时间(中历)', nrec.get('責任時間(中曆)', '?'))}")
            else:
                print(f"  │ [新-现代文] (无结果)")

            print(f"  └{'─' * 56}")

    # ── 汇总统计 ──
    print("\n\n" + "=" * 80)
    print("  汇总统计")
    print("=" * 80)
    print(f"  总查询数: {len(TEST_QUERIES)}")
    print(f"  旧系统总耗时: {total_old_time:.2f}s  (平均 {total_old_time/len(TEST_QUERIES):.2f}s/查询)")
    print(f"  新系统总耗时: {total_new_time:.2f}s  (平均 {total_new_time/len(TEST_QUERIES):.2f}s/查询)")
    print(f"  旧系统索引: 248,342 条 (cache/)")
    print(f"  新系统索引: 244,338 条 (cache_refined/)")
    print()

    # ── 交互式对比 ──
    print("=" * 80)
    print("  交互式对比模式")
    print("  输入自定义查询（q 退出），将同时展示新旧系统结果")
    print("=" * 80)

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

        print(f"\n{'─' * 70}")
        print(f"  【查询】: {query}")
        print(f"{'─' * 70}")

        # 旧系统
        t0 = time.time()
        old_r = r_old.search(query, top_k=3, verbose=True)
        t_old = time.time() - t0
        print(f"\n  ▶ 旧系统 (文言文) — {t_old:.2f}s:")
        for r in old_r:
            print(r_old.format_result(r))

        # 新系统
        t0 = time.time()
        new_r = r_new.search(query, top_k=3, verbose=True)
        t_new = time.time() - t0
        print(f"\n  ▶ 新系统 (现代文摘要) — {t_new:.2f}s:")
        for r in new_r:
            print(r_new.format_result(r))

        print(f"\n  ⏱ 旧: {t_old:.2f}s | 新: {t_new:.2f}s")


if __name__ == "__main__":
    main()
