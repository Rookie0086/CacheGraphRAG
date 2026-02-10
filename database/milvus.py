import time
from typing import List  # Optional, Dict,

from pymilvus import (  # utility,; FieldSchema,; CollectionSchema,; MilvusClient,; Connections,
    Collection,
    DataType,
    Milvus,
    connections,
)

from utils.base import print_text
from utils.llm_env import EmbeddingEnv

fmt = "\n=== {:30} ===\n"


class myMilvus(Milvus):

    def __init__(self, host="127.0.0.1", port="19530", **kwargs):
        super().__init__(host=host, port=port, **kwargs)

    def show_all_collections(self):
        ret = self.list_collections()
        print(f"=== all collections name: {ret}")

    def show_collections_stats(self, db_name):
        ret = self.get_collection_stats(db_name)
        print(f"=== stat of {db_name}: {ret}")

    def show_collections_schema(self, db_name):
        ret = self.describe_collection(db_name)
        print(f"=== schema of {db_name}: {ret}")

    def drop(self, db_name):
        ret = self.drop_collection(db_name)
        print(f"=== clear collection {db_name}: {ret}")

    # def exist(self, db_name):
    #     return self.has_collection(db_name)

    def get_vector_count(self, db_name):
        ret = self.get_collection_stats(db_name)
        return ret["row_count"]


class MilvusDB:

    def __init__(
        self,
        db_name,
        model_name="/home/shuyurui/model/bge-large-en-v1.5",
        overwrite=False,
        server_ip="127.0.0.1",
        server_port="19530",
        metric="COSINE",
        verbose=False,
    ):
        self.db_name = db_name
        self.overwrite = overwrite
        self.server_ip = server_ip
        self.server_port = server_port
        self.client = myMilvus(server_ip, server_port)

        self.db = None
        self.verbose = verbose
        self.metric = metric

        self.embed_model = EmbeddingEnv(model_name=model_name, batch_size=20)

        connections.connect("default", host="localhost", port=server_port)

    def insert(self, entities):
        time_s = time.time()
        # self.client.insert(self.db_name, entities)
        self.db.insert(entities)
        time_e = time.time()
        if self.verbose:
            print(f"insert cost {time_e - time_s}")

        self.db.load()

    def load(self):
        if not self.db:
            self.db = Collection(self.db_name)
            self.db.load()

    def search(self, embedding, limit=3):
        # self.db.load()
        # time_s = time.time()
        # self.db.flush()
        # time_e = time.time()
        # print(f'time cost {time_e - time_s:.3f}')
        search_params = {
            "metric_type": self.metric,
            "params": {"nprobe": 10},
        }
        start_time = time.time()
        result = self.db.search(embedding, "vec", search_params, limit=limit)
        end_time = time.time()

        # logging.info(f'embedding {embedding}, search cost {end_time-start_time:.3f}')

        # print(type(result), type(result[0]), type(result[0][0]))

        distance = [hit.distance for hits in result for hit in hits]
        pk = [hit.pk for hits in result for hit in hits]

        if self.verbose:
            print(f"search cost {end_time-start_time:.3f}")

        # for hits in result:
        #     print(hits)
        #     for hit in hits:
        #         print(f"hit: {hit}, random field: {hit.entity.get('distance')}, {hit.distance}")

        return pk, distance

    def create(self, consistency_level="Session"):

        # connections.connect("default", host="localhost", port="19530")

        if self.overwrite and self.db_name in self.client.list_collections():
            self.client.drop_collection(self.db_name)

        # fields = [
        #     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        #     # FieldSchema(name="random", dtype=DataType.DOUBLE),
        #     FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        # ]

        fields = {
            "fields": [
                {"name": "pk", "type": DataType.INT64, "is_primary": True},
                {
                    "name": "vec",
                    "type": DataType.FLOAT_VECTOR,
                    "params": {"dim": self.embed_model.dim},
                },
            ],
            "auto_id": False,
        }

        self.client.create_collection(
            self.db_name,
            fields,
            consistency_level=consistency_level,
            # Strong, Bounded, Eventually, Session
        )

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": self.metric,
            "params": {"nlist": 128},
        }
        # self.db.create_index("embeddings", index)

        # schema = CollectionSchema(fields, "customize schema")

        # self.db = Collection(self.db_name, schema)

        self.client.create_index(self.db_name, "vec", index)

        # print(Connections().list_connections)

        # print('has', self.client.has_collection(self.db_name))

        # self.db = Collection(self.db_name, consistency_level=consistency_level)
        self.db = Collection(self.db_name)
        self.db.load()


def test_db(db_name):
    vector_db = MilvusDB("cache", overwrite=True, metric="COSINE")

    vector_db.create()
    print("create done!")

    print(f"schema {vector_db.db.schema}")

    vector_db.show_collections_stats()
    pk, dis = vector_db.search([[1, 1, 1, 1]], limit=10)
    print(f"pk {pk}, dis {dis}")

    vector_db.insert([[1], [[1, 1, 1, 1]]])
    vector_db.insert([[2], [[2, 2, 2, 2]]])
    vector_db.insert([[3], [[3, 3, 3, 3]]])
    pk, dis = vector_db.search([[1, 1, 1, 1]], limit=10)
    print(f"pk {pk}, dis {dis}")

    vector_db.insert(
        [[5, 6, 7, 8], [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]
    )
    pk, dis = vector_db.search([[7, 7, 7, 7]], limit=10)
    print(f"pk {pk}, dis {dis}")

    vector_db.show_collections_stats()


if __name__ == "__main__":

    # test_db('cache')

    db_name = "test"
    db_name = "rgb_entities"
    db_name = "rgb"

    dim = 1024

    # create vector database
    # create_datebase(db_name='crag_small')

    vector_db = MilvusDB(db_name=db_name, overwrite=False)
    client = myMilvus()

    client.show_collections_stats(db_name=db_name)  # dict {'row_count': 150600}
    count = client.get_vector_count(db_name=db_name)
    print(f"=== vector count: {count}")

    client.show_all_collections()  # list
