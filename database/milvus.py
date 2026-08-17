import time
import json
from datetime import datetime, timezone
from typing import List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from src.utils.base import print_text
from src.llm.env import EmbeddingEnv


class myMilvus:
    """Lightweight wrapper around the Milvus native client, providing collection management and debugging helpers."""

    def __init__(self, host="127.0.0.1", port="19530", **kwargs):
        import uuid as _uuid
        self._my_alias = f"mgr_{_uuid.uuid4().hex[:8]}"
        kwargs.pop("using", None)
        connections.connect(self._my_alias, host=host, port=port, **kwargs)

    def list_collections(self):
        return utility.list_collections(using=self._my_alias)

    def drop_collection(self, name):
        return utility.drop_collection(name, using=self._my_alias)

    def create_collection(self, name, schema_dict, **kwargs):
        fields = []
        for raw in schema_dict.get("fields", []):
            params = dict(raw.get("params", {}))
            fields.append(FieldSchema(
                name=raw["name"], dtype=raw["type"],
                is_primary=raw.get("is_primary", False),
                auto_id=raw.get("auto_id", False), **params))
        schema = CollectionSchema(fields, auto_id=schema_dict.get("auto_id", False))
        return Collection(name, schema=schema, using=self._my_alias, **kwargs)

    def create_index(self, name, field_name, index_params):
        return Collection(name, using=self._my_alias).create_index(field_name, index_params)

    def get_collection_stats(self, name):
        return {"row_count": Collection(name, using=self._my_alias).num_entities}

    def describe_collection(self, name):
        return Collection(name, using=self._my_alias).schema

    def show_all_collections(self):
        """Print all collection names."""
        ret = self.list_collections()
        print(f"=== all collections name: {ret}")

    def show_collections_stats(self, db_name):
        """Print statistics (row count, etc.) for a specified collection."""
        ret = self.get_collection_stats(db_name)
        print(f"=== stat of {db_name}: {ret}")

    def show_collections_schema(self, db_name):
        """Print the schema of a specified collection."""
        ret = self.describe_collection(db_name)
        print(f"=== schema of {db_name}: {ret}")

    def drop(self, db_name):
        """Drop a specified collection."""
        ret = self.drop_collection(db_name)
        print(f"=== clear collection {db_name}: {ret}")

    def get_vector_count(self, db_name):
        """Get the total number of vectors in the collection."""
        ret = self.get_collection_stats(db_name)
        return ret["row_count"]


# ── Module-level global lock, only protects collection creation ──
import threading as _threading
_CREATE_LOCK = _threading.Lock()


class MilvusDB:
    """Milvus vector database wrapper. Thread-local connection for high-concurrency lock-free access."""

    def __init__(
        self,
        db_name,
        model_name="BAAI/bge-small-en-v1.5",
        overwrite=False,
        server_ip="127.0.0.1",
        server_port="19530",
        metric="COSINE",
        verbose=False,
        embed_model=None,
    ):
        self.db_name = db_name
        self.overwrite = overwrite
        self.server_ip = server_ip
        self.server_port = server_port
        self.client = myMilvus(server_ip, server_port)
        self.verbose = verbose
        self.metric = metric
        self._local = _threading.local()  # Thread-local storage
        self._prefix = f"{db_name}_"  # Thread alias prefix

        if embed_model is not None:
            self.embed_model = embed_model
        else:
            self.embed_model = EmbeddingEnv(model_name=model_name, batch_size=20)

    def _alias(self) -> str:
        """Unique connection alias for the current thread."""
        tid = _threading.get_ident()
        return f"{self._prefix}{tid}"

    def _get_db(self):
        """Get the Collection object for the current thread (lazy init + auto reconnect)."""
        alias = self._alias()
        db = getattr(self._local, 'db', None)
        if db is not None:
            try:
                _ = db.num_entities  # Check connection validity
                return db
            except Exception:
                pass  # Connection invalid, reconnect

        # Establish connection + load collection
        with _CREATE_LOCK:
            try:
                connections.get_connection(alias)
            except Exception:
                connections.connect(alias, host="localhost", port=self.server_port)
        db = Collection(self.db_name, using=alias)
        db.load()
        self._local.db = db
        return db

    def search(self, vector, search_params, limit, output_fields=None):
        """Vector search (thread-safe, lock-free)."""
        db = self._get_db()
        kwargs = dict(data=[vector], anns_field="vec", param=search_params,
                      limit=limit, consistency_level="Strong")
        if output_fields:
            kwargs["output_fields"] = output_fields
        return db.search(**kwargs)

    def query(self, expr, output_fields, limit=1):
        """Scalar query (thread-safe, lock-free)."""
        db = self._get_db()
        return db.query(expr=expr, output_fields=output_fields, limit=limit)

    def insert(self, entities):
        """Insert entity data."""
        db = self._get_db()
        time_s = time.time()
        db.insert(entities)
        if self.verbose:
            print(f"insert cost {time.time() - time_s}")

    def load(self):
        """Load collection into memory."""
        self._get_db()  # Trigger connection + load

    def search_simple(self, embedding, limit=3):
        """Simple vector search, returns (primary key list, distance list)."""
        db = self._get_db()
        search_params = {"metric_type": self.metric, "params": {"nprobe": 10}}
        start_time = time.time()
        result = db.search(embedding, "vec", search_params, limit=limit)
        distance = [hit.distance for hits in result for hit in hits]
        pk = [hit.pk for hits in result for hit in hits]
        if self.verbose:
            print(f"search cost {time.time() - start_time:.3f}")
        return pk, distance

    def flush(self):
        """Flush the buffer."""
        self._get_db().flush()

    # ── Collection Creation ──────────────────────────────────

    def _create_collection(self, fields: dict, consistency_level="Strong", ttl_seconds=None):
        """Generic collection creation: create collection -> create index -> load."""
        if self.overwrite and self.db_name in self.client.list_collections():
            self.client.drop_collection(self.db_name)
        kwargs = {"consistency_level": consistency_level}
        if ttl_seconds:
            kwargs["properties"] = {"collection.ttl.seconds": ttl_seconds}
        self.client.create_collection(self.db_name, fields, **kwargs)
        index = {"index_type": "HNSW", "metric_type": self.metric, "params": {"M": 16, "efConstruction": 200}}
        self.client.create_index(self.db_name, "vec", index)
        # Pre-load the current thread's db
        self._get_db()

    def create(self, consistency_level="Strong"):
        """Create a basic collection containing only pk + vec."""
        fields = {
            "fields": [
                {"name": "pk", "type": DataType.INT64, "is_primary": True},
                {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": self.embed_model.dim}},
            ],
            "auto_id": False,
        }
        self._create_collection(fields, consistency_level)

    def create_chunk_collection(self, consistency_level="Strong"):
        """Create a document chunk collection with chunk_id / chunk_text / entity_uids / graph_meta fields."""
        fields = {
            "fields": [
                {"name": "pk", "type": DataType.INT64, "is_primary": True},
                {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": self.embed_model.dim}},
                {"name": "chunk_id", "type": DataType.VARCHAR, "params": {"max_length": 255}},
                {"name": "chunk_text", "type": DataType.VARCHAR, "params": {"max_length": 8192}},
                {"name": "entity_uids", "type": DataType.VARCHAR, "params": {"max_length": 4096}},
                {"name": "graph_meta", "type": DataType.JSON},
                {"name": "ts", "type": DataType.VARCHAR, "params": {"max_length": 64}},
            ],
            "auto_id": False,
        }
        self._create_collection(fields, consistency_level, ttl_seconds=1209600)

    def create_entity_collection(self, consistency_level="Strong"):
        """Create an entity index collection with uid / name / type / desc fields."""
        fields = {
            "fields": [
                {"name": "uid", "type": DataType.INT64, "is_primary": True},
                {"name": "name", "type": DataType.VARCHAR, "params": {"max_length": 255}},
                {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": self.embed_model.dim}},
                {"name": "type", "type": DataType.VARCHAR, "params": {"max_length": 255}},
                {"name": "desc", "type": DataType.VARCHAR, "params": {"max_length": 2048}},
            ],
            "auto_id": False,
        }
        self._create_collection(fields, consistency_level, ttl_seconds=1209600)

    # ── Schema Detection ──────────────────────────────────

    def _has_field(self, field_name: str) -> bool:
        """Check if the current collection contains the specified field."""
        try:
            db = self._get_db()
            return any(getattr(field, "name", "") == field_name for field in db.schema.fields)
        except Exception:
            return False

    def _get_field_dtype(self, field_name: str):
        """Get the data type of the specified field."""
        try:
            db = self._get_db()
            for field in db.schema.fields:
                if getattr(field, "name", "") == field_name:
                    return getattr(field, "dtype", None)
        except Exception:
            return None
        return None

    def _serialize_graph_meta(self, graph_meta: dict):
        """Serialize graph_meta to JSON string (if the field type is not native JSON)."""
        if graph_meta is None:
            return None
        dtype = self._get_field_dtype("graph_meta")
        if dtype == DataType.JSON:
            return graph_meta
        return json.dumps(graph_meta, ensure_ascii=True)

    # ── Chunk Query ────────────────────────────────────────

    def get_chunk_entities(self, chunk_id: str) -> List[str]:
        """Query list of entity UIDs by chunk_id."""
        if not (self._has_field("chunk_id") and self._has_field("entity_uids")):
            return []
        try:
            rows = self.query(expr=f'chunk_id == "{chunk_id}"',
                              output_fields=["entity_uids"], limit=1)
        except Exception:
            return []
        if not rows: return []
        value = rows[0].get("entity_uids", "")
        if not isinstance(value, str) or not value: return []
        return [v for v in value.split(",") if v]

    def get_chunk_text(self, chunk_id: str) -> dict:
        """Get chunk text by chunk_id."""
        if not (self._has_field("chunk_id") and self._has_field("chunk_text")):
            return {"text": "", "ts": ""}
        output_fields = ["chunk_text"]
        if self._has_field("ts"): output_fields.append("ts")
        try:
            rows = self.query(expr=f'chunk_id == "{chunk_id}"',
                              output_fields=output_fields, limit=1)
        except Exception:
            return {"text": "", "ts": ""}
        if not rows: return {"text": "", "ts": ""}
        row = rows[0]
        return {"text": row.get("chunk_text", "") or "",
                "ts": row.get("ts", "") or ""}

    def get_chunk_graph_meta(self, chunk_id: str) -> dict:
        """Get graph metadata by chunk_id."""
        if not self._has_field("graph_meta"):
            return {}
        try:
            rows = self.query(expr=f'chunk_id == "{chunk_id}"',
                              output_fields=["graph_meta"], limit=1)
        except Exception:
            return {}
        if not rows: return {}
        value = rows[0].get("graph_meta")
        if isinstance(value, dict): return value
        if isinstance(value, str) and value:
            try: return json.loads(value)
            except json.JSONDecodeError: return {}
        return {}

    def _prepare_chunk_record(self, chunk_id, vector, entity_uids, chunk_text, graph_meta, timestamp):
        pk = abs(hash(chunk_id)) % (2**63 - 1)
        record = {"pk": pk, "vec": vector}
        if self._has_field("chunk_id"): record["chunk_id"] = chunk_id
        if self._has_field("chunk_text") and chunk_text is not None:
            record["chunk_text"] = chunk_text
        if self._has_field("entity_uids"):
            if entity_uids is None: entity_uids = []
            record["entity_uids"] = ",".join(str(uid) for uid in entity_uids if str(uid))
        if self._has_field("graph_meta") and graph_meta is not None:
            record["graph_meta"] = self._serialize_graph_meta(graph_meta)
        if self._has_field("ts"):
            if not timestamp:
                timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            record["ts"] = timestamp
        return record

    def insert_chunk(self, chunk_id, vector, entity_uids=None, chunk_text=None,
                     graph_meta=None, timestamp=None):
        db = self._get_db()
        record = self._prepare_chunk_record(chunk_id, vector, entity_uids, chunk_text, graph_meta, timestamp)
        db.insert([record])

    # ── Fact Collection ──────────────────────────────────

    def create_fact_collection(self, consistency_level="Strong"):
        """Create a fact index collection."""
        fields = {
            "fields": [
                {"name": "pk", "type": DataType.INT64, "is_primary": True},
                {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": self.embed_model.dim}},
                {"name": "fact_text", "type": DataType.VARCHAR, "params": {"max_length": 1024}},
                {"name": "subj_name", "type": DataType.VARCHAR, "params": {"max_length": 255}},
                {"name": "obj_name", "type": DataType.VARCHAR, "params": {"max_length": 255}},
                {"name": "relation", "type": DataType.VARCHAR, "params": {"max_length": 255}},
                {"name": "chunk_id", "type": DataType.VARCHAR, "params": {"max_length": 255}},
                {"name": "subj_uid", "type": DataType.INT64},
                {"name": "obj_uid", "type": DataType.INT64},
            ],
            "auto_id": False,
        }
        self._create_collection(fields, consistency_level, ttl_seconds=1209600)

    def insert_facts_batch(self, fact_list: list, vectors: list):
        """Batch insert fact records."""
        db = self._get_db()
        records = []
        for fact_data, vector in zip(fact_list, vectors):
            pk = abs(hash(fact_data["fact_text"])) % (2**63 - 1)
            records.append({
                "pk": pk,
                "vec": vector.tolist() if hasattr(vector, "tolist") else vector,
                "fact_text": fact_data.get("fact_text", ""),
                "subj_name": fact_data.get("subj_name", ""),
                "obj_name": fact_data.get("obj_name", ""),
                "relation": fact_data.get("relation", ""),
                "chunk_id": fact_data.get("chunk_id", ""),
                "subj_uid": fact_data.get("subj_uid", 0),
                "obj_uid": fact_data.get("obj_uid", 0),
            })
        db.insert(records)

    def search_facts(self, vector: list, topk: int = 100) -> list:
        """Vector search for facts, returns a list of complete metadata."""
        search_params = {"metric_type": self.metric, "params": {"nprobe": 10}}
        output_fields = ["fact_text", "subj_name", "obj_name",
                         "relation", "chunk_id", "subj_uid", "obj_uid"]
        try:
            result = self.search(vector, search_params, topk, output_fields=output_fields)
        except Exception:
            return []
        facts = []
        if result and len(result) > 0:
            for hit in result[0]:
                ent = hit.entity
                def _g(key, default=""):
                    try: return ent.get(key) or default
                    except Exception: return getattr(ent, key, default)
                facts.append({
                    "fact_text": _g("fact_text"),
                    "subj_name": _g("subj_name"),
                    "obj_name": _g("obj_name"),
                    "relation": _g("relation"),
                    "chunk_id": _g("chunk_id"),
                    "subj_uid": _g("subj_uid", 0),
                    "obj_uid": _g("obj_uid", 0),
                    "score": float(hit.distance),
                })
        return facts
