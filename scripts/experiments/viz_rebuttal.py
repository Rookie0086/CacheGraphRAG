#!/usr/bin/env python
"""rebuttal 两组数据可视化:
1. 字节级存储核算(向量/L1/L2/gexf,来源 storage_report_2wiki600)
2. 图5 流式实验参数设置 + 结果(来源 fig5_stream_200 run)

输出 PNG 到 output/viz/。
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "viz"
OUT.mkdir(parents=True, exist_ok=True)

# 中文字体(macOS 可用字体,按优先级)
for name in ["Hiragino Sans GB", "Arial Unicode MS", "Heiti SC", "STHeiti", "Songti SC"]:
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------- 数据加载 ----------
STORE = json.loads(
    (ROOT / "output/storage/storage_report_2wiki600_20260814_152716.json").read_text()
)
RUN = ROOT / "output/experiments/run_20260815_134022_fig5_stream_200"
METRICS = json.loads((RUN / "stream_fig5/fig5/artifacts/standard_metrics.json").read_text())
SUMMARY = json.loads((RUN / "stream_fig5/fig5/artifacts/summary.json").read_text())
CONFIG = (RUN / "stream_fig5/fig5/config.yaml").read_text()


def mb(b: float) -> float:
    return b / 1024 / 1024  # MiB


# ============================================================
# 图 1:字节级存储核算
# ============================================================
# 全口径组件(含索引估算)
items = [
    ("向量数据\n(Milvus 行×维度×4B)", STORE["vector_backend_milvus"]["total_bytes"]),
    ("向量 HNSW\n索引估算", STORE["vector_backend_milvus"]["milvus_hnsw_index_est"]),
    ("gexf 拓扑\n文件", STORE["gexf_topology"]["bytes"]),
    ("L1 内存图\n(NetworkX 估算)", STORE["l1_memory"]["bytes_est"]),
    ("L2 磁盘\n(Nebula du -sb)", STORE["l2_nebula_disk"]["data_bytes"]),
]
total_all = sum(v for _, v in items)
total_no_idx = total_all - STORE["vector_backend_milvus"]["milvus_hnsw_index_est"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
labels = [l for l, _ in items]
vals = [v for _, v in items]
colors = ["#4C72B0", "#9E9AC8", "#74C476", "#F6A45C", "#DE3B2F"]
bars = ax.barh(range(len(items)), [mb(v) for v in vals], color=colors, height=0.62)
ax.set_yticks(range(len(items)))
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("字节占用 (MiB)")
ax.set_title("A. 全存储组件字节核算(wikimultihopqa 600 条)\n总计入库(含索引) %.2f MiB | 无索引口径 %.2f MiB"
             % (mb(total_all), mb(total_no_idx)), fontsize=11)
for b_, v in zip(bars, vals):
    ax.text(b_.get_width() + mb(0.3 * 1024 * 1024), b_.get_y() + b_.get_height() / 2,
            "%.2f MiB\n%.1f%%" % (mb(v), 100 * v / total_all),
            va="center", fontsize=9)

ax = axes[1]
# 无索引口径拆分(向量数据 / gexf / L1 / L2)
parts = [
    ("向量数据", STORE["vector_backend_milvus"]["total_bytes"]),
    ("gexf 拓扑", STORE["gexf_topology"]["bytes"]),
    ("L1 内存", STORE["l1_memory"]["bytes_est"]),
    ("L2 磁盘", STORE["l2_nebula_disk"]["data_bytes"]),
]
pv = [v for _, v in parts]
pl = [l for l, _ in parts]
colors2 = ["#4C72B0", "#74C476", "#F6A45C", "#DE3B2F"]
wedges, _, autotexts = ax.pie(
    pv, labels=[f"{l}\n{mb(v):.2f} MiB" for l, v in parts],
    autopct="%.1f%%", colors=colors2, startangle=90,
    textprops={"fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(8)
ax.set_title("B. 无索引口径占比(总 %.2f MiB)" % mb(total_no_idx), fontsize=11)

fig.suptitle("字节级存储核算(2wiki / wikimultihopqa 600 条构建后)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "storage_byte_breakdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 图 2:流式实验参数设置(表格)
# ============================================================
rows = [
    ("数据规模", "wikimultihopqa,start=0~end=200,seed=42"),
    ("QA 规模", "200 条(n_qa=200,query=200,batch=20,三轮 100×100×100 出指标)"),
    ("LLM 模型", "gpt-4o-mini(gptgod.cloud) | embedding bge-m3 | rerank bge-reranker-v2-m3"),
    ("L1 容量 C_max", "200(chunk 数,与 l1_max_chunks=200 一致)"),
    ("晋升阈值", "promotion_threshold=3,tau_hit=3,tau_desc=0.85"),
    ("检索超参", "hybrid,gamma=0.5,max_hops=3,beam_width B=4,top_chunks=15"),
    ("语义去重", "semantic_dup_threshold=0.95,unknown_tolerance=2"),
    ("切块", "chunk_size=800,overlap=100"),
    ("流式协议", "l2_seed_mode=empty(从空 L1+L2 重建),enable_rehydrate=true"),
    ("预热/索引", "warmup_ratio=0.0,skip_index=true"),
]
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=["参数", "取值"], loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", weight="bold")
    if c == 0 and r > 0:
        cell.set_facecolor("#EAF0F7")
        cell.set_text_props(weight="bold")
ax.set_title("图5 流式实验参数设置(run_20260815_134022_fig5_stream_200)", fontsize=13, pad=20)
fig.tight_layout()
fig.savefig(OUT / "stream_fig5_params.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 图 3:流式实验结果
# ============================================================
rounds = ["1st_odd\n(逐批 QA,晋升触发)", "2nd_odd\n(L2 固化留存)", "2nd_even\n(向量库延迟重载)"]
r_acc = [METRICS["rounds"][k]["acc"] for k in ["1st_odd", "2nd_odd", "2nd_even"]]
r_rouge = [METRICS["rounds"][k]["rouge_l_f"] for k in ["1st_odd", "2nd_odd", "2nd_even"]]
r_bs = [METRICS["rounds"][k]["bs_f1"] for k in ["1st_odd", "2nd_odd", "2nd_even"]]
r_l1 = [METRICS["rounds"][k]["l1_hit_rate"] for k in ["1st_odd", "2nd_odd", "2nd_even"]]
r_l2 = [METRICS["rounds"][k]["l2_hit_rate"] for k in ["1st_odd", "2nd_odd", "2nd_even"]]

s1, s2, s3 = SUMMARY  # 三阶段数组,机制统计相同
evicted = s1["evicted_chunks"]
att = s1["rehydrate_attempts"]
succ = s1["rehydrate_successes"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# (1) 质量指标
ax = axes[0, 0]
x = np.arange(3)
w = 0.26
ax.bar(x - w, r_acc, w, label="ACC", color="#4C72B0")
ax.bar(x, r_rouge, w, label="ROUGE-L", color="#74C476")
ax.bar(x + w, r_bs, w, label="BERTScore-F1", color="#F6A45C")
for xi, v in zip(x - w, r_acc):
    ax.text(xi, v + 0.01, "%.2f" % v, ha="center", fontsize=8)
for xi, v in zip(x, r_rouge):
    ax.text(xi, v + 0.01, "%.2f" % v, ha="center", fontsize=8)
for xi, v in zip(x + w, r_bs):
    ax.text(xi, v + 0.01, "%.2f" % v, ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=9)
ax.set_ylim(0, 0.75)
ax.set_ylabel("得分")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("精度:驱逐后 ACC 持平(L2/向量库通道)")

# (2) L1/L2 命中率
ax = axes[0, 1]
ax.plot(x, r_l1, "o-", label="L1 命中率", color="#DE3B2F")
ax.plot(x, r_l2, "s-", label="L2 命中率", color="#4C72B0")
for xi, v in zip(x, r_l1):
    ax.text(xi, v + 0.03, "%.2f" % v, ha="center", fontsize=9)
for xi, v in zip(x, r_l2):
    ax.text(xi, v + 0.03, "%.2f" % v, ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=9)
ax.set_ylim(0, 0.85)
ax.set_ylabel("命中率")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("命中率:L1 降(0.71→0.07),L2 升(0.26→0.45)")

# (3) 驱逐 / 重载
ax = axes[0, 2]
ev = [evicted, att, succ]
el = ["累计驱逐\nchunk", "重载尝试\nattempts", "重载成功\nsuccesses"]
b = ax.bar(el, ev, color=["#DE3B2F", "#9E9AC8", "#74C476"])
for bb, v in zip(b, ev):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 40, "%d" % v, ha="center", fontsize=9)
ax.set_ylabel("次数")
ax.set_title("机制激活:驱逐 4151 / 重载尝试 1966 / 成功 184")

# (4) token 成本
ax = axes[1, 0]
tok = [s1["prompt_tokens"], s1["completion_tokens"]]
tl = ["prompt tokens", "completion tokens"]
b = ax.bar(tl, tok, color=["#4C72B0", "#74C476"])
for bb, v in zip(b, tok):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 8000, f"{v:,}", ha="center", fontsize=9)
ax.set_ylabel("token 数")
ax.set_ylim(0, max(tok) * 1.15)
ax.set_title("LLM 成本:724 calls,共 %.1fK token(约 0.29 USD)"
             % ((s1["prompt_tokens"] + s1["completion_tokens"]) / 1000))

# (5) 时延 P50(全链路 / 检索 / 答案)
ax = axes[1, 1]
p50 = [s1["total_p50_s"], s2["total_p50_s"], s3["total_p50_s"]]
p95 = [s1["total_p95_s"], s2["total_p95_s"], s3["total_p95_s"]]
ax.plot(x, p50, "o-", label="全链路 P50", color="#4C72B0")
ax.plot(x, p95, "s-", label="全链路 P95", color="#DE3B2F")
for xi, v in zip(x, p50):
    ax.text(xi, v + 0.5, "%.1f" % v, ha="center", fontsize=8)
for xi, v in zip(x, p95):
    ax.text(xi, v + 0.5, "%.1f" % v, ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=9)
ax.set_ylabel("秒")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("全链路时延 P50/P95(查询阶段)")

# (6) 检索分阶段耗时
ax = axes[1, 2]
stage_keys = ["stage_embed_avg_s", "stage_dpr_avg_s", "stage_entity_avg_s",
              "stage_fusion_avg_s", "stage_graph_avg_s"]
stage_labels = ["embed", "dpr", "entity", "fusion", "graph"]
x2 = np.arange(len(stage_keys))
for si, s in enumerate(SUMMARY):
    vals = [s[k] for k in stage_keys]
    ax.bar(x2 + si * 0.25, vals, 0.25, label=rounds[si].split("\n")[0])
ax.set_xticks(x2 + 0.25)
ax.set_xticklabels(stage_labels, fontsize=9)
ax.set_ylabel("平均秒/查询")
ax.legend(fontsize=8)
ax.set_title("检索内部分阶段平均耗时")

fig.suptitle("图5 流式实验结果(wikimultihopqa 0-200,空 L1+L2 重建,200 QA×3 轮)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "stream_fig5_results.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("已生成:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p)
