#!/usr/bin/env python
"""修复 qa.json 中提取失败的 predict(退化 JSON 原样回退)。

背景:src/CacheGraphRAG.py:395-416 在 LLM 返回非法 JSON 时,json.loads 失败
后回退正则 `(?:final_)?answer\s*[:：]` 匹配不了 `final_answer":`(键名后带
闭合引号),最终整个 raw 原样成为 predict。本脚本对这类条目(含
step1_candidates 或 ```json)重新走提取逻辑,补一个容忍键后引号的正则,
优先提取 "final_answer" 字段的值。

用法:
  python scripts/experiments/repair_predict.py \
    output/experiments/run_20260815_025512/lru_rehydrate/rehydrate_off \
    ...  # 一个或多个 case 目录;缺省时扫描最新 run 的全部 case

不改原始数据:修复前把原 predict 存入 predict_raw,再写回修复后的 predict。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def extract_answer(raw: str) -> str:
    """从 LLM 原始输出提取答案,模拟 CacheGraphRAG.py 提取逻辑 + 补容错正则。

    返回值可能为空串(raw 里确实没有可提取的答案,如 LLM 截断)。
    """
    if not raw:
        return ""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(cleaned[start:end + 1])
            answer = payload.get("final_answer") or payload.get("answer")
            if answer:
                return str(answer).strip()
        except json.JSONDecodeError:
            pass
    # 修复点:源码正则 `(?:final_)?answer\s*[:：]` 匹配不了 `final_answer":`,
    # 这里直接抓 "final_answer": "..." 的值(容忍键后带引号)。
    m2 = re.search(r'"final_answer"\s*:\s*"([^"]*)"', raw)
    if m2 and m2.group(1).strip():
        return m2.group(1).strip()
    # 源码原回退正则
    m = re.search(r"(?:final_)?answer\s*[:：]\s*(.+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def is_degenerate(predict: str) -> bool:
    """predict 是否为提取失败原样回退的退化 JSON。"""
    return "step1_candidates" in predict or "```json" in predict


def repair_case(case_dir: pathlib.Path) -> dict:
    """修复单个 case 的 qa.json,返回统计。"""
    qa_path = case_dir / "artifacts" / "qa.json"
    if not qa_path.is_file():
        return {"case": case_dir.name, "error": "无 qa.json"}
    rows = json.loads(qa_path.read_text())
    fixed, still_degenerate = 0, 0
    for r in rows:
        pr = r.get("predict", "")
        if not is_degenerate(pr):
            continue
        r["predict_raw"] = pr          # 保留原始退化输出
        ans = extract_answer(r.get("raw_answer", ""))
        if ans:
            r["predict"] = ans
            fixed += 1
        else:
            still_degenerate += 1      # raw 里连 final_answer 都没有,救不回
    qa_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return {"case": case_dir.name, "fixed": fixed, "still_degenerate": still_degenerate}


def main() -> int:
    ap = argparse.ArgumentParser(description="修复 qa.json 提取失败的 predict")
    ap.add_argument("case_dirs", nargs="*", help="case 目录;缺省自动扫描最新 run")
    args = ap.parse_args()

    if args.case_dirs:
        dirs = [pathlib.Path(d) for d in args.case_dirs]
    else:
        runs = sorted(
            (p for p in (ROOT / "output" / "experiments").glob("run_*")
             if (p / "lru_rehydrate").is_dir()),
            key=lambda p: p.name, reverse=True,
        )
        if not runs:
            print("[repair] 未找到 lru_rehydrate run")
            return 1
        dirs = sorted(
            (p for p in (runs[0] / "lru_rehydrate").glob("*")
             if p.is_dir() and (p / "artifacts" / "qa.json").is_file()),
        )
    if not dirs:
        print("[repair] 未找到含 qa.json 的 case 目录")
        return 1

    total_fixed = 0
    for d in dirs:
        stat = repair_case(d)
        print(f"[repair] {stat}")
        total_fixed += stat.get("fixed", 0)
    print(f"[repair] 共修复 {total_fixed} 条 predict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
