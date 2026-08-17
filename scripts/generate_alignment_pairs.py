#!/usr/bin/env python
"""whoqa(SpecificQA)半自动实体对齐标注对生成器

从 data/datasets/whoqa/whoqa_experiment_dataset_600.json 生成
data/annotations/entity_alignment_pairs.jsonl(与 scripts/experiments/alignment_benchmark.py
的输入格式一致:每行 {"left":{name,type,desc}, "right":{...}, "is_same": 0/1})。

标签为启发式**建议值**,带 reason/confidence,必须人工复核后才能用于评测。
whoqa 每条样本 = 同名实体消歧单元:phase_1(干扰实体) vs phase_2(正确实体)。
六类候选对:

  is_same=0(同名/变体但不同实体):
    type0  简称 vs 简称:phase_1(干扰) vs phase_2(正确),同一样本内
    type1  跨样本同名:同一 target 名出现在多个样本(不同真实实体)
    type4  名称变体硬负例:phase_1 全名 vs phase_2 简称 —— 名称匹配会误判,考验 desc 判别
  is_same=1(同一实体):
    type2  简称 vs 正确实体全名(phase_2 文档首部)
    type3  全名 vs 姓氏引用(仅 PERSON 实体)
    type5  简称 vs 干扰实体全名(phase_1 文档首部,同一干扰实体)

用法:
  python scripts/generate_alignment_pairs.py [--force] [--limit 600] [--seed 42]
已存在输出文件时默认拒绝覆盖(防止冲掉人工修正),--force 重新生成。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_FILE = ROOT / "data/datasets/whoqa/whoqa_experiment_dataset_600.json"
OUT_FILE = ROOT / "data/annotations/entity_alignment_pairs.jsonl"

DESC_LIMIT = 400  # desc 截断长度(嵌入与复核都够用)

# ── 实体类型分类:优先"X is/was a/an/the <名词>"主模式 ────────────
_ARTICLE_NOUN_MAP = {
    # 人物
    "person": "PERSON", "man": "PERSON", "woman": "PERSON", "player": "PERSON",
    "politician": "PERSON", "singer": "PERSON", "actor": "PERSON", "actress": "PERSON",
    "footballer": "PERSON", "artist": "PERSON", "writer": "PERSON", "author": "PERSON",
    "composer": "PERSON", "novelist": "PERSON", "poet": "PERSON", "painter": "PERSON",
    "engineer": "PERSON", "physician": "PERSON", "doctor": "PERSON", "scientist": "PERSON",
    "lawyer": "PERSON", "coach": "PERSON", "athlete": "PERSON", "runner": "PERSON",
    "jumper": "PERSON", "halfback": "PERSON", "professor": "PERSON", "general": "PERSON",
    "admiral": "PERSON", "captain": "PERSON", "minister": "PERSON", "lord": "PERSON",
    "president": "PERSON", "king": "PERSON", "queen": "PERSON", "philosopher": "PERSON",
    "historian": "PERSON", "economist": "PERSON", "journalist": "PERSON", "photographer": "PERSON",
    "architect": "PERSON", "mathematician": "PERSON", "biologist": "PERSON", "physicist": "PERSON",
    "chemist": "PERSON", "explorer": "PERSON", "missionary": "PERSON", "priest": "PERSON",
    "bishop": "PERSON", "cardinal": "PERSON", "saint": "PERSON", "god": "PERSON",
    "goddess": "PERSON", "prince": "PERSON", "princess": "PERSON", "duke": "PERSON",
    "duchess": "PERSON", "count": "PERSON", "countess": "PERSON", "baron": "PERSON",
    "viscount": "PERSON", "judge": "PERSON", "mayor": "PERSON", "governor": "PERSON",
    "senator": "PERSON", "cricketer": "PERSON", "golfer": "PERSON", "boxer": "PERSON",
    "wrestler": "PERSON", "swimmer": "PERSON", "cyclist": "PERSON", "sailor": "PERSON",
    "soldier": "PERSON", "officer": "PERSON", "pilot": "PERSON", "astronaut": "PERSON",
    "businessman": "PERSON", "banker": "PERSON", "entrepreneur": "PERSON", "musician": "PERSON",
    "guitarist": "PERSON", "pianist": "PERSON", "violinist": "PERSON", "conductor": "PERSON",
    "director": "PERSON", "producer": "PERSON", "screenwriter": "PERSON", "playwright": "PERSON",
    "dramatist": "PERSON", "daughter": "PERSON", "son": "PERSON", "brother": "PERSON",
    "sister": "PERSON", "father": "PERSON", "mother": "PERSON", "wife": "PERSON",
    "husband": "PERSON", "monarch": "PERSON", "emperor": "PERSON", "empress": "PERSON",
    "nobleman": "PERSON", "noblewoman": "PERSON", "military": "PERSON", "politician": "PERSON",
    # 作品
    "song": "WORK", "album": "WORK", "single": "WORK", "film": "WORK", "movie": "WORK",
    "painting": "WORK", "novel": "WORK", "book": "WORK", "symphony": "WORK", "opera": "WORK",
    "play": "WORK", "sculpture": "WORK", "artwork": "WORK", "television": "WORK",
    "series": "WORK", "episode": "WORK", "soundtrack": "WORK", "poem": "WORK", "drama": "WORK",
    "comedy": "WORK", "documentary": "WORK", "video": "WORK", "story": "WORK",
    "musical": "WORK", "concerto": "WORK", "sonata": "WORK", "treatise": "WORK",
    "biography": "WORK", "autobiography": "WORK", "textbook": "WORK", "journal": "WORK",
    "newspaper": "WORK", "magazine": "WORK", "comic": "WORK", "manga": "WORK", "anime": "WORK",
    "game": "WORK", "statue": "WORK", "monument": "WORK", "bust": "WORK", "fresco": "WORK",
    "mosaic": "WORK", "photograph": "WORK", "collection": "WORK", "anthology": "WORK",
    "band": "WORK", "group": "WORK", "choir": "WORK", "duo": "WORK", "trio": "WORK",
    # 地点
    "town": "PLACE", "village": "PLACE", "city": "PLACE", "province": "PLACE",
    "district": "PLACE", "region": "PLACE", "island": "PLACE", "river": "PLACE",
    "mountain": "PLACE", "country": "PLACE", "municipality": "PLACE", "commune": "PLACE",
    "valley": "PLACE", "lake": "PLACE", "sea": "PLACE", "ocean": "PLACE", "peninsula": "PLACE",
    "state": "PLACE", "county": "PLACE", "parish": "PLACE", "department": "PLACE",
    "canton": "PLACE", "prefecture": "PLACE", "kingdom": "PLACE", "empire": "PLACE",
    "colony": "PLACE", "settlement": "PLACE", "borough": "PLACE", "township": "PLACE",
    "archipelago": "PLACE", "atoll": "PLACE", "reef": "PLACE", "bay": "PLACE", "cove": "PLACE",
    "cape": "PLACE", "peak": "PLACE", "volcano": "PLACE", "glacier": "PLACE", "desert": "PLACE",
    "forest": "PLACE", "park": "PLACE", "reserve": "PLACE",
}
_PERSON_KW = tuple(v for v in _ARTICLE_NOUN_MAP if _ARTICLE_NOUN_MAP[v] == "PERSON")
_WORK_KW = tuple(v for v in _ARTICLE_NOUN_MAP if _ARTICLE_NOUN_MAP[v] == "WORK")
_PLACE_KW = tuple(v for v in _ARTICLE_NOUN_MAP if _ARTICLE_NOUN_MAP[v] == "PLACE")


def classify_type(doc: str) -> str:
    head = (doc or "")[:250]
    # 1) "X is/was a/an/the <形容词...> <名词>" —— 实体类型紧跟系动词,
    #    跳过数字/形容词("a 1934 novel" / "a former British long jumper")。
    m = re.search(r"\b(?:is|was)\s+(?:a|an|the)\s+((?:[^\s.]+\s+){0,4}[^\s.]+)", head)
    if m:
        phrase = m.group(1).split()
        for w in phrase:
            w = w.strip("-,.").lower()
            if w in _ARTICLE_NOUN_MAP:
                return _ARTICLE_NOUN_MAP[w]
    # 2) 兜底:仅扫描首句关键词(避免 "novel by ... writer" 之类把作品误判成人)
    low = head.split(".", 1)[0].lower()
    hits = [(low.find(k), k) for k in _PERSON_KW if k in low]
    hits += [(low.find(k), k) for k in _WORK_KW if k in low]
    hits += [(low.find(k), k) for k in _PLACE_KW if k in low]
    if hits:
        hits.sort()
        return _ARTICLE_NOUN_MAP[hits[0][1]]
    return "ENTITY"


# 非姓氏的尾词 / 官职-称谓型首词(仅用于 type3 姓氏对的过滤)
_NON_SURNAME_LAST = ("agriculture", "band", "group", "company", "party", "committee",
                     "commission", "society", "association", "council", "ministry",
                     "department", "university", "school", "hospital", "church", "museum",
                     "theatre", "theater", "temple", "station", "club", "team", "army",
                     "navy", "government", "order", "office", "house", "bank", "press",
                     "college", "institute", "center", "centre", "federation", "republic",
                     "kingdom", "empire", "province", "county", "district", "city", "town",
                     "village", "island", "river", "mountain", "lake", "sea", "park",
                     "monument", "memorial", "statue", "cathedral", "abbey", "castle",
                     "palace", "tower", "bridge", "road", "street", "square", "market",
                     "choir", "duo", "trio", "quartet", "label", "record", "film", "song")
_TITLE_FIRST = ("minister", "president", "king", "queen", "prince", "princess", "duke",
                "duchess", "lord", "saint", "emperor", "empress", "archduke",
                "archduchess", "count", "countess", "baron", "baroness", "viscount",
                "governor", "mayor", "senator", "general", "admiral", "captain", "order",
                "national", "university", "the", "of", "sir", "dr", "mr", "mrs", "ms",
                "prof", "rev", "fr", "st")


def is_personal_surname_pair(name: str, surname: str, doc: str) -> bool:
    """type3 姓氏对的门槛:名字本身像人名 + 姓氏在文档中作为独立引用出现。

    'Minister of Agriculture'/'Diante do Trono' 等官职/乐队名会被拒;
    'John Gorham' 且文档提到 'Gorham was...' 才通过。
    """
    nt = tokens(name)
    if len(nt) < 2 or nt[0] in _TITLE_FIRST or nt[-1] in _NON_SURNAME_LAST:
        return False
    # "X of Y" 贵族/地名式名字(Anne of Bohemia):Y 不是姓氏,拒绝
    if re.search(r"\bof\s+[A-Za-z]+\s*$", str(name or ""), re.IGNORECASE):
        return False
    # 姓氏作为独立引用出现:至少一处匹配的前 20 字符内不是 "大写词+空格"
    # (即排除 'John Gorham' 这种紧跟全名的出现,保留 'Gorham was born' 这种裸引用)
    for m in re.finditer(rf"\b{re.escape(surname)}\b", doc or ""):
        before = doc[max(0, m.start() - 20):m.start()]
        if not re.search(r"[A-Z][a-z]+\s$", before):
            return True
    return False


def norm_name(name: str) -> str:
    v = str(name or "").lower().strip()
    return re.sub(r"[^a-z0-9\s]", " ", v)


def tokens(name: str) -> list:
    return [t for t in norm_name(name).split() if t]


def clean_surname_target(name: str) -> bool:
    """仅对 2-3 个纯字母词的姓名生成姓氏对(避免 'Edward Butler, 2nd Viscount Galmoye' 这类)"""
    return bool(re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z]+){1,2}", str(name or "").strip()))


def extract_full_name(doc: str) -> str | None:
    """从文档首句提取全名:"John Searl Howell (December 4, 1915 - ...) was a halfback..." """
    if not doc:
        return None
    m = re.match(r"^\s*((?:Dr\.|Mr\.|Mrs\.|Ms\.|St\.)\s+)?([A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){1,3})\s*\(", doc)
    return m.group(2) if m else None


def contains_in_order(short_tokens: list, full_tokens: list) -> bool:
    """short 的 token 是否按顺序出现在 full 中(John Howell ⊆ John Searl Howell)"""
    pos = 0
    for t in short_tokens:
        try:
            pos = full_tokens.index(t, pos) + 1
        except ValueError:
            return False
    return True


def first_doc(data: list) -> str:
    return str((data or [""])[0])


def desc_of(doc: str) -> str:
    return doc[:DESC_LIMIT]


def add_pair(pairs, seen, left, right, is_same, source, reason, confidence):
    key = (norm_name(left["name"]), norm_name(right["name"]), left["desc"], right["desc"])
    if key in seen:
        return
    seen.add(key)
    pairs.append({"left": left, "right": right, "is_same": is_same,
                  "source": source, "reason": reason, "confidence": confidence})


def build_type0(items, rng, n):
    """简称 vs 简称:phase_1(干扰实体) vs phase_2(正确实体),同名不同实体"""
    pairs, seen = [], set()
    for it in items:
        name = str(it.get("target_entity") or "").strip()
        p1, p2 = first_doc(it.get("phase_1_data")), first_doc(it.get("phase_2_data"))
        if not name or not p1 or not p2:
            continue
        typ = classify_type(p1)
        add_pair(pairs, seen,
                 {"name": name, "type": typ, "desc": desc_of(p1)},
                 {"name": name, "type": typ, "desc": desc_of(p2)},
                 0, f"whoqa:type0:{name}",
                 "同一样本 phase_1(干扰实体) vs phase_2(正确实体),同名不同实体", 0.9)
    rng.shuffle(pairs)
    return pairs[:n]


def build_type1(items, rng):
    """跨样本同名:同一 target 名出现在多个样本(不同真实实体)"""
    groups = {}
    for it in items:
        name = str(it.get("target_entity") or "").strip()
        p2 = first_doc(it.get("phase_2_data"))
        if not name or not p2:
            continue
        groups.setdefault(norm_name(name), []).append((it, p2))
    pairs, seen = [], set()
    for gname, members in groups.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                it_a, d_a = members[i]
                it_b, d_b = members[j]
                if norm_name(d_a[:80]) == norm_name(d_b[:80]):
                    continue
                typ = classify_type(d_a)
                add_pair(pairs, seen,
                         {"name": it_a.get("target_entity"), "type": typ, "desc": desc_of(d_a)},
                         {"name": it_b.get("target_entity"), "type": typ, "desc": desc_of(d_b)},
                         0, f"whoqa:type1:{gname}",
                         "同一 target 名出现在不同样本,为不同真实实体(消歧难点)", 0.8)
    rng.shuffle(pairs)
    return pairs


def build_type2(items, rng):
    """简称 vs 正确实体全名(phase_2 文档首部):同一实体"""
    pairs, seen = [], set()
    for it in items:
        name = str(it.get("target_entity") or "").strip()
        p2 = first_doc(it.get("phase_2_data"))
        full = extract_full_name(p2)
        if not name or not full:
            continue
        nt, nf = tokens(name), tokens(full)
        if len(nf) <= len(nt) or not contains_in_order(nt, nf):
            continue
        typ = classify_type(p2)
        add_pair(pairs, seen,
                 {"name": name, "type": typ, "desc": desc_of(p2)},
                 {"name": full, "type": typ, "desc": desc_of(p2)},
                 1, f"whoqa:type2:{name}->{full}",
                 "phase_2 文档中的全名与 target 简称指向同一(正确)实体", 0.8)
    rng.shuffle(pairs)
    return pairs


def build_type3(items, rng, n):
    """全名 vs 姓氏引用(仅 PERSON 且姓氏独立引用):同一实体"""
    pairs, seen = [], set()
    for it in items:
        name = str(it.get("target_entity") or "").strip()
        p2 = first_doc(it.get("phase_2_data"))
        if not clean_surname_target(name) or classify_type(p2) != "PERSON":
            continue
        nt = tokens(name)
        if len(nt) < 2:
            continue
        surname = nt[-1].capitalize()
        if not is_personal_surname_pair(name, surname, p2):
            continue
        typ = "PERSON"
        add_pair(pairs, seen,
                 {"name": name, "type": typ, "desc": desc_of(p2)},
                 {"name": surname, "type": typ, "desc": desc_of(p2)},
                 1, f"whoqa:type3:{name}->{surname}",
                 "文档中以姓氏独立引用同一实体", 0.7)
    rng.shuffle(pairs)
    return pairs[:n]


def build_type4(items, rng, n):
    """名称变体硬负例:phase_1 全名 vs phase_2 简称 —— 名称匹配会误判,考验 desc 判别"""
    pairs, seen = [], set()
    for it in items:
        name = str(it.get("target_entity") or "").strip()
        p2 = first_doc(it.get("phase_2_data"))
        if not name or not p2:
            continue
        nt = tokens(name)
        for doc in (it.get("phase_1_data") or []):
            full = extract_full_name(str(doc))
            if not full:
                continue
            nf = tokens(full)
            if len(nf) <= len(nt) or not contains_in_order(nt, nf):
                continue
            typ = classify_type(p2)
            add_pair(pairs, seen,
                     {"name": full, "type": typ, "desc": desc_of(str(doc))},
                     {"name": name, "type": typ, "desc": desc_of(p2)},
                     0, f"whoqa:type4:{full}->{name}",
                     "名称变体(John Howell ⊆ John David Howell)但为不同实体:干扰 vs 正确", 0.85)
    rng.shuffle(pairs)
    return pairs[:n]


def build_type5(items, rng, n):
    """简称 vs 干扰实体全名(phase_1 各文档首部):同一干扰实体"""
    pairs, seen = [], set()
    for it in items:
        name = str(it.get("target_entity") or "").strip()
        if not name:
            continue
        nt = tokens(name)
        for doc in (it.get("phase_1_data") or []):
            full = extract_full_name(str(doc))
            if not full:
                continue
            nf = tokens(full)
            if len(nf) <= len(nt) or not contains_in_order(nt, nf):
                continue
            typ = classify_type(str(doc))
            add_pair(pairs, seen,
                     {"name": name, "type": typ, "desc": desc_of(str(doc))},
                     {"name": full, "type": typ, "desc": desc_of(str(doc))},
                     1, f"whoqa:type5:{name}->{full}",
                     "phase_1 文档中简称与全名指同一(干扰)实体", 0.8)
    rng.shuffle(pairs)
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_FILE)
    ap.add_argument("--n-type0", type=int, default=200)
    ap.add_argument("--n-type4", type=int, default=120)
    ap.add_argument("--n-type3", type=int, default=150)
    ap.add_argument("--n-type5", type=int, default=150)
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"输出已存在: {args.out} —— 已有人工修正则不要覆盖;确认重生成请加 --force")
        sys.exit(1)

    with open(DATA_FILE, encoding="utf-8") as f:
        items = json.load(f)[:args.limit]
    rng = random.Random(args.seed)

    pairs = []
    pairs += build_type0(items, rng, args.n_type0)
    pairs += build_type1(items, rng)
    pairs += build_type4(items, rng, args.n_type4)
    pairs += build_type2(items, rng)
    pairs += build_type3(items, rng, args.n_type3)
    pairs += build_type5(items, rng, args.n_type5)

    # 复核友好:is_same=0 在前,组内按 confidence 升序(最可疑的排最前)
    pairs.sort(key=lambda p: (p["is_same"], p["confidence"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    n0 = Counter(p["is_same"] for p in pairs)
    by_src = Counter(p["source"].split(":")[1] for p in pairs)
    print(f"写入 {args.out}")
    print(f"  共 {len(pairs)} 对 | is_same=0: {n0.get(0, 0)} | is_same=1: {n0.get(1, 0)}")
    print(f"  按类型: {dict(by_src)}")
    print("  按 (is_same, confidence) 排序,confidence 最低的最需人工复核")
    print("  复核后运行: bash scripts/run_reviewer_experiments.sh alignment")


if __name__ == "__main__":
    main()
