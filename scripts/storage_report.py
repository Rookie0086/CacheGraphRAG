#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""存储占用字节粒度统计(中档#10,对应 R2-5.1 / R4-W2)
====================================================================
输出一份字节口径的存储报告 JSON,拆分:
  1. 向量后备存储   Milvus 各集合(行数 × 向量维度 × 4B + 标量字段估算)
  2. L2 磁盘        Nebula L2 空间真实磁盘字节(宿主 .nebula/data/storage*/nebula/<space_id>/,
                     跨 storage 分片求和;space ID → 名字映射用 --nebula-space-map 传入)

  本版只记录「向量数据库 + L2」两项(用户指定),不再输出 L1 内存估算 / gexf 拓扑。

用法:
  python scripts/storage_report.py --dataset wikimultihopqa --start 0 --end 600 \
      --nebula-space-map "wikimultihopqa=25,exp_clone_a=50" \
      [--nebula-space wikimultihopqa] \
      [--output output/storage/storage_report.json]

只读脚本:不修改 Milvus/Nebula 任何数据。
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# 估算常量(字节)
VECTOR_BYTES_PER_DIM = 4       # float32
L1_ROW_SCALAR_BYTES = 256      # 每行标量字段(id/name/desc/chunk 引用)粗略估算
MILVUS_INDEX_FACTOR = 1.5      # HNSW 索引 ≈ 向量字节 × 1.5(粗估)

# Nebula L2 数据真实位置:宿主挂载卷 .nebula/data/storage*/nebula/<space_id>/
# (docker exec du -sb /usr/local/nebula/data 是容器内空目录, 统计结果恒为 12KB 级, 已废弃)
NEBULA_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", ".nebula", "data")


def stat_milvus(milvus_host="127.0.0.1", milvus_port="19530", include=None):
    """统计 Milvus 各集合行数/维度/字节估算(只读)。失败返回 None。

    include: 只统计这些集合名;为 None 时统计全部集合。
    """
    try:
        from pymilvus import MilvusClient
    except Exception as e:
        return {"error": f"pymilvus 不可用: {e}"}
    try:
        # 注意:此环境旧版 connections.connect + Milvus() 会挂起,
        # MilvusClient(tcp://) 工作正常;timeout 保证快速失败
        client = MilvusClient(uri=f"tcp://{milvus_host}:{milvus_port}", timeout=15)
        names = client.list_collections()
    except Exception as e:
        return {"error": f"连接 Milvus 失败: {e}"}

    if include:
        names = [n for n in names if n in include]
    collections, total = {}, 0
    for name in names:
        try:
            rows = client.get_collection_stats(name).get("row_count", 0)
            desc = client.describe_collection(name)
            dim, scalar_fields = 0, 0
            for f in desc.get("fields", []):
                if f.get("type") == 101:      # DataType.FLOAT_VECTOR
                    dim = f.get("params", {}).get("dim", 0)
                else:
                    scalar_fields += 1
            vec_bytes = rows * dim * VECTOR_BYTES_PER_DIM
            scalar_bytes = rows * scalar_fields * L1_ROW_SCALAR_BYTES
            est = vec_bytes + scalar_bytes
            collections[name] = {
                "rows": rows, "dim": dim, "scalar_fields": scalar_fields,
                "vec_bytes": vec_bytes, "scalar_bytes": scalar_bytes,
                "bytes_est": est,
            }
            total += est
        except Exception as e:
            collections[name] = {"error": str(e)}
    return {"collections": collections, "total_bytes": total,
            "milvus_hnsw_index_est": int(total * MILVUS_INDEX_FACTOR)}


def _space_dir_bytes(space_dir: str) -> int:
    """宿主磁盘目录字节数(macOS du 无 -b, 用 du -sk 的 KB 再 ×1024)。"""
    try:
        du = subprocess.run(["du", "-sk", space_dir],
                            capture_output=True, text=True, timeout=120)
        kb = int(du.stdout.split()[0]) if du.stdout.strip() else 0
        return kb * 1024
    except Exception:
        return 0


def stat_nebula_disk(space_map=None, include_spaces=None,
                     nebula_data_dir=NEBULA_DATA_DIR):
    """统计 Nebula L2 真实磁盘字节:宿主 .nebula/data/storage*/nebula/<space_id>/ 跨分片求和。

    space_map:      {space名: space_id_int} 名字 → ID 映射(供报告可读)。
    include_spaces: 只统计这些 space 名;为 None 时统计磁盘上全部 space ID。
    真实数据位于挂载卷 .nebula/data/storage{0,1,2}/nebula/<space_id>/(RocksDB data+wal),
    而非容器内 /usr/local/nebula/data(空目录)。
    """
    if not os.path.isdir(nebula_data_dir):
        return {"error": f"Nebula 宿主数据卷不存在: {nebula_data_dir}"}

    # 收集全部 storage 分片下的 space 目录
    storage_dirs = sorted(
        d for d in os.listdir(nebula_data_dir) if d.startswith("storage"))
    space_ids = set()
    for sd in storage_dirs:
        neb_dir = os.path.join(nebula_data_dir, sd, "nebula")
        if not os.path.isdir(neb_dir):
            continue
        for sid in os.listdir(neb_dir):
            if os.path.isdir(os.path.join(neb_dir, sid)) and sid.isdigit():
                space_ids.add(int(sid))

    # 名字 → ID 反向映射
    id2name = {}
    for name, sid in (space_map or {}).items():
        id2name.setdefault(int(sid), name)

    # 决定统计哪些 space
    if include_spaces:
        wanted_ids = {int(space_map[name]) for name in include_spaces
                      if name in (space_map or {})}
        if len(wanted_ids) != len(set(include_spaces)):
            return {"error": f"include_spaces 含未在 space_map 中的名字: "
                             f"{set(include_spaces) - set(space_map or {})}"}
        space_ids = wanted_ids & space_ids

    per_space, total = {}, 0
    for sid in sorted(space_ids):
        bytes_sum = sum(_space_dir_bytes(os.path.join(nebula_data_dir, sd, "nebula", str(sid)))
                        for sd in storage_dirs
                        if os.path.isdir(os.path.join(nebula_data_dir, sd, "nebula", str(sid))))
        name = id2name.get(sid, f"space_{sid}")
        per_space[sid] = {"name": name, "bytes": bytes_sum,
                          "storage_dirs_count": sum(
                              1 for sd in storage_dirs
                              if os.path.isdir(os.path.join(nebula_data_dir, sd, "nebula", str(sid))))}
        total += bytes_sum
    return {"nebula_data_dir": nebula_data_dir, "space_map": space_map or {},
            "spaces": per_space, "data_bytes": total}


def parse_space_map(text):
    """解析 --nebula-space-map "name=id,name2=id2" → {name: int(id)}。"""
    space_map = {}
    if not text:
        return space_map
    for part in text.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, sid = part.split("=", 1)
        space_map[name.strip()] = int(sid.strip())
    return space_map


def main():
    ap = argparse.ArgumentParser(description="存储占用字节粒度统计(中档#10, 只记录向量数据库+L2)")
    ap.add_argument("--dataset", default="wikimultihopqa")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=600)
    ap.add_argument("--nebula-space-map", default="",
                    help='space 名→ID 映射, 逗号分隔: "wikimultihopqa=25,exp_a=50"')
    ap.add_argument("--nebula-space", action="append", default=[],
                    help="只统计指定 space 名(可多次传); 默认统计磁盘上全部 space")
    ap.add_argument("--include-collections", default="",
                    help="只统计指定 Milvus 集合名(逗号分隔); 默认统计全部集合")
    ap.add_argument("--output", default=None, help="JSON 输出路径(默认 output/storage/storage_report_<ts>.json)")
    ap.add_argument("--skip-milvus", action="store_true", help="跳过 Milvus 统计(连不上时使用)")
    ap.add_argument("--skip-nebula", action="store_true", help="跳过 Nebula 统计")
    args = ap.parse_args()

    print(f"=== 存储占用统计 [向量数据库 + L2] {args.dataset} {args.start}:{args.end} {datetime.now():%Y-%m-%d %H:%M:%S} ===")

    space_map = parse_space_map(args.nebula_space_map)
    report = {"dataset": args.dataset, "start": args.start, "end": args.end,
              "timestamp": datetime.now().isoformat(), "estimate_constants": {
                  "vector_bytes_per_dim": VECTOR_BYTES_PER_DIM,
                  "milvus_index_factor": MILVUS_INDEX_FACTOR}}

    # 1. 向量后备存储(Milvus)
    if args.skip_milvus:
        report["vector_backend_milvus"] = {"skipped": True}
    else:
        include = [n.strip() for n in args.include_collections.split(",") if n.strip()]
        report["vector_backend_milvus"] = stat_milvus(include=include or None)
        print(f"  [Milvus 向量] {report['vector_backend_milvus'].get('total_bytes', 'N/A'):,} B")
        for name, c in report["vector_backend_milvus"].get("collections", {}).items():
            if "error" in c:
                print(f"    - {name}: {c['error']}")
            else:
                print(f"    - {name}: rows={c['rows']} dim={c['dim']} ~{c['bytes_est']:,} B")

    # 2. L2 磁盘(Nebula, 宿主真实字节)
    if args.skip_nebula:
        report["l2_nebula_disk"] = {"skipped": True}
    else:
        report["l2_nebula_disk"] = stat_nebula_disk(
            space_map=space_map, include_spaces=args.nebula_space or None)
        nb = report["l2_nebula_disk"]
        if "error" in nb:
            print(f"  [Nebula L2] {nb['error']}")
        else:
            print(f"  [Nebula L2] 磁盘: {nb['data_bytes']:,} B (跨 {len(nb['spaces'])} 个 space 分片合计)")
            for sid, sp in nb["spaces"].items():
                print(f"    - space {sid} ({sp['name']}): {sp['bytes']:,} B  "
                      f"[{sp['storage_dirs_count']} 分片]")

    # 汇总字节口径(向量 + L2 两项)
    vec = report["vector_backend_milvus"].get("total_bytes") or 0
    l2 = report["l2_nebula_disk"].get("data_bytes") or 0
    report["total_bytes"] = vec + l2
    report["breakdown_percent"] = {
        "vector_backend_milvus": round(100 * vec / (vec + l2), 2) if (vec + l2) else 0,
        "l2_nebula_disk": round(100 * l2 / (vec + l2), 2) if (vec + l2) else 0,
    }
    print(f"  [汇总] 总存储(字节口径) = 向量 {vec:,} + L2 {l2:,} = {report['total_bytes']:,} B")
    print(f"         占比: {report['breakdown_percent']}")

    # 输出 JSON
    output = args.output
    if not output:
        os.makedirs("output/storage", exist_ok=True)
        output = f"output/storage/storage_report_{args.dataset}_{args.start}_{args.end}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n存储报告已保存: {output}")


if __name__ == "__main__":
    main()
