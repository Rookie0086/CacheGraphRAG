#!/usr/bin/env python
"""Create a verified, isolated NebulaGraph L2 snapshot for one experiment case."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import Any, Dict, Iterable, List

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.nebulagraph import NebulaClient


_SPACE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


def _validate_space(name: str) -> str:
    if not _SPACE_RE.fullmatch(name or ""):
        raise ValueError(f"invalid Nebula space name: {name!r}")
    return name


def _execute(session, query: str):
    result = session.execute(query)
    if result is None or not result.is_succeeded():
        detail = result.error_msg() if result is not None else "no result"
        raise RuntimeError(f"Nebula query failed: {detail}; query={query}")
    return result


def _rows(result) -> List[Dict[str, Any]]:
    keys = [str(key) for key in result.keys()]
    columns = {key: [value.cast() for value in result.column_values(key)] for key in keys}
    size = max((len(values) for values in columns.values()), default=0)
    return [{key: columns[key][idx] for key in keys} for idx in range(size)]


def _names(session, query: str) -> set[str]:
    rows = _rows(_execute(session, query))
    return {str(value) for row in rows for value in row.values() if value is not None}


def _wait_for(check, description: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except Exception as exc:  # schema propagation can transiently reject USE/SHOW
            last_error = exc
        time.sleep(0.5)
    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"timed out waiting for {description}{detail}")


def _execute_retry(session, query: str, description: str, timeout: float):
    """Retry idempotent snapshot inserts while Nebula schema propagates to storage."""
    holder = {}

    def attempt():
        holder["result"] = _execute(session, query)
        return True

    _wait_for(attempt, description, timeout)
    return holder["result"]


def _count(session, space: str, pattern: str) -> int:
    result = _execute(session, f"USE {space}; MATCH {pattern} RETURN count(*) AS n;")
    values = result.column_values("n")
    return int(values[0].cast()) if values else 0


def _quote(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    return f'"{text}"'


def _chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _paged_rows(session, query: str, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    offset = 0
    while True:
        batch = _rows(_execute(session, f"{query} SKIP {offset} LIMIT {batch_size};"))
        if not batch:
            return
        yield batch
        offset += len(batch)


def ensure_target_schema(session, target: str, timeout: float = 60.0) -> None:
    target = _validate_space(target)
    spaces = _names(session, "SHOW SPACES;")
    if target not in spaces:
        _execute(session, f"CREATE SPACE {target}(vid_type=INT64, partition_num=10, replica_factor=1);")
        _wait_for(lambda: target in _names(session, "SHOW SPACES;"), f"space {target}", timeout)
    _wait_for(lambda: bool(_execute(session, f"USE {target};")),
              f"space {target} to become usable", timeout)
    _execute(session, f"USE {target}; CREATE TAG IF NOT EXISTS entity(name string, type string, source_chunk string);")
    _execute(session, f"USE {target}; CREATE EDGE IF NOT EXISTS relationship(relationship string, source_chunk string);")
    _wait_for(lambda: "entity" in _names(session, f"USE {target}; SHOW TAGS;"),
              f"entity tag in {target}", timeout)
    _wait_for(lambda: "relationship" in _names(session, f"USE {target}; SHOW EDGES;"),
              f"relationship edge in {target}", timeout)
    # Nebula 的 CREATE SPACE 在 metad 立即可见,但 storage 的 parts 需心跳周期
    # (~10s)才真正就绪;期间 USE 可执行、DDL 可成功,但 MATCH 会报
    # "Storage Error: Space not found"。必须等到 storage 层可查询,否则
    # clone_l2 的目标计数 MATCH 必然失败。
    _wait_for(lambda: bool(_execute(session, f"USE {target}; MATCH (v) RETURN count(*) AS n;")),
              f"space {target} queryable at storage level", timeout)


def clone_l2(session, source: str, target: str, batch_size: int = 100,
             empty: bool = False, timeout: float = 60.0) -> dict:
    source, target = _validate_space(source), _validate_space(target)
    if source == target:
        raise ValueError("source and target Nebula spaces must differ")
    spaces = _names(session, "SHOW SPACES;")
    if source not in spaces:
        raise RuntimeError(f"source L2 space does not exist: {source}")
    ensure_target_schema(session, target, timeout)

    source_counts = {
        "vertices": _count(session, source, "(v)"),
        "edges": _count(session, source, "()-[e]->()"),
    }
    if not empty and source_counts["vertices"] == 0:
        raise RuntimeError(f"source L2 space is empty: {source}")
    before = {
        "vertices": _count(session, target, "(v)"),
        "edges": _count(session, target, "()-[e]->()"),
    }
    expected = {"vertices": 0, "edges": 0} if empty else source_counts
    if before == expected:
        return {"source_space": source, "target_space": target, "mode": "empty" if empty else "clone",
                "source": source_counts, "target_before": before, "target_after": before,
                "verified": True, "reused": True}
    if before["vertices"] or before["edges"]:
        raise RuntimeError(
            f"target L2 space is partially populated ({before}); refusing to overwrite {target}")
    if empty:
        return {"source_space": source, "target_space": target, "mode": "empty",
                "source": source_counts, "target_before": before, "target_after": before,
                "verified": True, "reused": False}

    vertex_query = (
        f"USE {source}; MATCH (v:entity) RETURN id(v) AS vid, "
        "properties(v).name AS name, properties(v).type AS type, "
        "properties(v).source_chunk AS source_chunk")
    for page in _paged_rows(session, vertex_query, batch_size):
        for batch in _chunks(page, batch_size):
            values = ", ".join(
                f"{int(row['vid'])}:({_quote(row.get('name'))}, {_quote(row.get('type'))}, "
                f"{_quote(row.get('source_chunk'))})" for row in batch)
            _execute_retry(
                session,
                f"USE {target}; INSERT VERTEX entity(name, type, source_chunk) VALUES {values};",
                f"entity schema in {target} to accept writes", timeout)

    edge_query = (
        f"USE {source}; MATCH ()-[e:relationship]->() RETURN src(e) AS src, dst(e) AS dst, "
        "rank(e) AS edge_rank, properties(e).relationship AS relation, "
        "properties(e).source_chunk AS source_chunk")
    for page in _paged_rows(session, edge_query, batch_size):
        for batch in _chunks(page, batch_size):
            values = ", ".join(
                f"{int(row['src'])}->{int(row['dst'])}@{int(row['edge_rank'])}:"
                f"({_quote(row.get('relation'))}, {_quote(row.get('source_chunk'))})"
                for row in batch)
            _execute_retry(
                session,
                f"USE {target}; INSERT EDGE relationship(relationship, source_chunk) VALUES {values};",
                f"relationship schema in {target} to accept writes", timeout)

    after = {
        "vertices": _count(session, target, "(v)"),
        "edges": _count(session, target, "()-[e]->()"),
    }
    if after != source_counts:
        raise RuntimeError(f"L2 clone verification failed: source={source_counts}, target={after}")
    return {"source_space": source, "target_space": target, "mode": "clone",
            "source": source_counts, "target_before": before, "target_after": after,
            "verified": True, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--empty", action="store_true",
                        help="create and verify an empty isolated L2 instead of cloning data")
    args = parser.parse_args()
    started = time.time()
    client = NebulaClient()
    if client.session is None:
        raise RuntimeError("NebulaGraph is unavailable; cannot prepare experiment L2")
    report = clone_l2(client.session, args.source, args.target,
                      batch_size=max(1, args.batch_size), empty=args.empty,
                      timeout=args.timeout)
    report["elapsed_s"] = round(time.time() - started, 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
