# 验证 P1-4:τ_sim / τ_desc 双阈值配置化(论文式5 + 算法1 行9/12/15)
# τ_sim=0.85 强对齐阈值(alignment_threshold);τ_desc 弱语义阈值(alignment_desc_threshold,
# 默认 0.6,替换原硬编码 0.6/0.5)。纯内存 mock,不连接 Milvus。
import sys
import os
import inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from unittest import mock

import src.entity.resolver as RE


class FakeMilvusDB:
    def __init__(self, db_name=None, overwrite=False, embed_model=None):
        self.db_name = db_name

    def create_entity_collection(self):
        pass

    def load(self):
        pass


class FakeMilvusClient:
    def list_collections(self):
        return []  # 触发 create_entity_collection 分支


def make_resolver(**kw):
    with mock.patch.object(RE, "MilvusDB", FakeMilvusDB), \
         mock.patch.object(RE, "myMilvus", lambda: FakeMilvusClient()):
        return RE.AsyncEntityResolver(
            embedding_func=lambda t: [0.1] * 4,
            collection_name="entity_index_test", **kw)


ok = True

# ── 场景 1:构造函数签名含 desc_threshold,默认 0.6 ──
sig = inspect.signature(RE.AsyncEntityResolver.__init__)
params = sig.parameters
s1 = ("desc_threshold" in params and params["desc_threshold"].default == 0.6
      and params["threshold"].default == 0.85)
ok &= s1
print(f"  [签名] desc_threshold 默认 {params['desc_threshold'].default}, "
      f"threshold 默认 {params['threshold'].default} {'PASS' if s1 else 'FAIL'}")

# ── 场景 2:默认值实例化后正确存储 ──
r = make_resolver()
s2 = (r.threshold == 0.85 and r.desc_threshold == 0.6)
ok &= s2
print(f"  [默认实例] threshold={r.threshold} desc_threshold={r.desc_threshold} "
      f"(期望 0.85/0.6) {'PASS' if s2 else 'FAIL'}")

# ── 场景 3:显式传参覆盖 ──
r3 = make_resolver(threshold=0.9, desc_threshold=0.7)
s3 = (r3.threshold == 0.9 and r3.desc_threshold == 0.7)
ok &= s3
print(f"  [显式传参] threshold={r3.threshold} desc_threshold={r3.desc_threshold} "
      f"(期望 0.9/0.7) {'PASS' if s3 else 'FAIL'}")

# ── 场景 4:config.yaml 双阈值显式赋值且符号语义正确 ──
cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "config.yaml")
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
ret = cfg.get("retrieval", {})
hyp = cfg.get("hyperparameters", {})
s4 = (ret.get("alignment_threshold") == 0.85
      and ret.get("alignment_desc_threshold") == 0.6
      and hyp.get("tau_sim") == 0.85 and hyp.get("tau_desc") == 0.6)
ok &= s4
print(f"  [config] alignment_threshold={ret.get('alignment_threshold')} "
      f"alignment_desc_threshold={ret.get('alignment_desc_threshold')} "
      f"tau_sim={hyp.get('tau_sim')} tau_desc={hyp.get('tau_desc')} "
      f"(期望 0.85/0.6/0.85/0.6) {'PASS' if s4 else 'FAIL'}")

# ── 场景 5:判定分支不再含硬编码 0.6/0.5(应引用 self.desc_threshold)──
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "entity", "resolver.py")
with open(src_path, encoding="utf-8") as f:
    src = f.read()
# 取 _resolve_with_vector 决策区(排除 _desc_similarity 的 0.15/0.65 与 name_cache 的 0.3)
decision_zone = src.split("def _resolve_with_vector")[1].split("def resolve_batch_async")[0]
s5 = (decision_zone.count("self.desc_threshold") >= 3
      and "desc_sim >= 0.6" not in decision_zone
      and "desc_sim < 0.6" not in decision_zone
      and "desc_sim >= 0.5" not in decision_zone)
ok &= s5
print(f"  [硬编码清除] _resolve_with_vector 决策区引用 self.desc_threshold "
      f"{decision_zone.count('self.desc_threshold')} 次,无残留 0.6/0.5 "
      f"{'PASS' if s5 else 'FAIL'}")

print("\n=== τ_sim/τ_desc 阈值配置化验证:", "全部通过 ✅" if ok else "存在失败 ❌")
sys.exit(0 if ok else 1)
