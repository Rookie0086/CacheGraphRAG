# Reviewer experiment suite

This suite runs the experiments requested in `revise.md`. It preserves raw
predictions and configuration snapshots and creates JSON, CSV, PNG and PDF
outputs under `output/experiments/run_<timestamp>/`.

## Safety and prerequisites

- Start Milvus and NebulaGraph before model-backed experiments.
- Build the base index once before QA-only experiments.
- Each QA case uses an isolated `exp_*` Nebula space; resetting cold/warm state
  does not clear the original paper space. The existing Milvus base index is
  reused read-only.
- Before an L2-enabled QA case starts, the runner clones the configured source
  L2 (`retrieval.l2_source_space`, normally `wikimultihopqa`) into that isolated
  space. QA starts only after the schema is writable and source/target vertex
  and edge counts match. The `fairness` protocol intentionally starts with a
  verified empty L2 so it can measure cold versus post-warm state.
- Reviewer runs set `retrieval.fail_on_l2_error=true`: a missing tag, unavailable
  space or failed graph traversal invalidates the case instead of silently
  falling back to L1/vector retrieval.
- `embedding_batch` creates uniquely named experiment collections because it
  must rebuild the index.
- Do not commit API keys. Prefer environment overrides. For the local server:

```bash
export CACHEGRAPH_MODEL_BACKEND=openai
export CACHEGRAPH_MODEL_NAME=Llama-3.1-8B-Instruct
export CACHEGRAPH_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export CACHEGRAPH_MODEL_API_KEY="$(<.secrets/local_llm_key)"
export CACHEGRAPH_EMBED_BACKEND=api
export CACHEGRAPH_EMBED_NAME=bge-m3-mlx-fp16
export CACHEGRAPH_EMBED_BASE_URL=http://127.0.0.1:8000/v1
# export CACHEGRAPH_EMBED_API_KEY=...  # only if different from chat key
export CACHEGRAPH_RERANK_BACKEND=none
```

For Agentic experiments on a local Apple Silicon/MLX server, start with
`retrieval.qa_concurrency=2`, `retrieval.agentic_parallelism=2`, and
`model.max_concurrency=2`. The first two expose question-level and beam-level
parallelism; the last value is a shared safety cap across all LLM calls. Keep
`retrieval.agentic_batch_planning=true` to combine same-depth planner calls.
Increase the global cap only after checking server latency and error rate.

Embedding environment overrides take precedence over the selected config. The
wrapper reuses `CACHEGRAPH_MODEL_API_KEY` for embeddings unless an explicit
`CACHEGRAPH_EMBED_API_KEY` is provided, so config snapshots need no credential.
All `api_key`, `token` and `password` fields are removed from generated case
configs. For an API reranker, use `CACHEGRAPH_RERANK_BACKEND/NAME/BASE_URL/API_KEY`.

## Commands

Preview every case without running models:

```bash
bash scripts/run_reviewer_experiments.sh --all --dry-run --start 0 --end 100
```

Run individual groups:

```bash
bash scripts/run_reviewer_experiments.sh fairness --start 0 --end 100 \
  --seeds 42 43 44 --warmup-ratio 0.15

bash scripts/run_reviewer_experiments.sh latency_cost --start 0 --end 100 --queries 100

bash scripts/run_reviewer_experiments.sh locality --start 0 --end 100 --queries 100

bash scripts/run_reviewer_experiments.sh lru_rehydrate --start 0 --end 100 --queries 100

bash scripts/run_reviewer_experiments.sh embedding_batch --start 0 --end 100

bash scripts/run_reviewer_experiments.sh beam_gamma loop_blocking \
  --start 0 --end 100 --queries 100

bash scripts/run_reviewer_experiments.sh cache_ablation storage \
  --start 0 --end 100 --queries 100

bash scripts/run_reviewer_experiments.sh table8_recheck \
  --start 0 --end 200 --seeds 42 43 44

bash scripts/run_reviewer_experiments.sh streaming_analysis --dataset whoqa
```

## Experiment mapping

| Group | Reviewer concern | Cases / outputs |
|---|---|---|
| `fairness` | R4-W2 leakage | Same held-out questions at cold and post-warm states; three seeds |
| `latency_cost` | R2-6.2, R3-W4, R4-W3 | baseline, agentic B=1/B=4; P50/P95, stages, calls, tokens |
| `locality` | R4-W3 | repeat rate, Zipf alpha and gradual topic drift |
| `lru_rehydrate` | R4-W4-1/2/6 | Cmax sensitivity and rehydrate on/off |
| `embedding_batch` | R4-W4-5 | batch size 1/16/32/64/128 and wait windows |
| `beam_gamma` | R2-6.4, R3-W6 | B, gamma and max-hops sensitivity |
| `loop_blocking` | R4-W4-4 | none/string/semantic/semantic+UNKNOWN |
| `cache_ablation` | R2-6.1 | vector, L1, L1+L2, agentic-only, full |
| `storage` | R2-5-1, R4-W2 | full byte-level storage report |
| `table8_recheck` | R2-6.3 | identical slice/config with seeds 42/43/44 |
| `alignment` | R3-W1/W5, R4-W1 | labeled-pair alignment accuracy/cost benchmark |
| `streaming_analysis` | R3-W7/W8 | update volume, recurrence, growth and drift plots |

## Entity-alignment annotations and real baselines

The first alignment run creates
`data/annotations/entity_alignment_pairs.jsonl` if it is missing. Replace and
extend the two examples with 500-1000 manually labeled pairs. Then run:

```bash
bash scripts/run_reviewer_experiments.sh alignment
```

The built-in `light_name_proxy` and `hippo_vec_proxy` are clearly labeled
mechanism-level proxies. They must not be reported as end-to-end LightRAG or
HippoRAG results. To include predictions exported from real external baselines:

```bash
.conda/cachegraphrag-mac/bin/python scripts/experiments/alignment_benchmark.py \
  --pairs data/annotations/entity_alignment_pairs.jsonl \
  --output output/experiments/alignment_real \
  --prediction LightRAG=/path/lightrag_predictions.jsonl \
  --prediction HippoRAG=/path/hipporag_predictions.jsonl
```

Prediction files must have one JSON object per labeled pair in the same order,
with `{"prediction": 0}` or `{"prediction": 1}`.

## Output layout

```text
output/experiments/run_<timestamp>/
  run.json
  summary.json
  summary.csv
  fairness/
    summary.json
    summary.csv
    figures/*.png, *.pdf
    seed_42/
      config.yaml
      command.json
      l2_seed_command.json
      l2_seed.log
      stdout.log
      artifacts/l2_seed.json
      artifacts/*.json
```

`l2_seed.json` records the source and target spaces, seed mode, before/after
vertex and edge counts, verification status and elapsed time. The consolidated
summary also includes `l2_seed_verified`, `l2_vertices_seeded`,
`l2_edges_seeded`, `l2_query_errors` and `l2_measurement_valid`. Agentic cases
aggregate L1/L2 hits across every internal retrieval call, including beam paths
that were evaluated but not selected as the final path.

Use `summary.csv` for paper tables and PDF figures for the camera-ready paper.
Always retain `run.json`, per-case configs and raw prediction files as the
reproducibility package.
