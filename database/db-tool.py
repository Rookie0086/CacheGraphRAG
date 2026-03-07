import argparse
import os

# from database.nebulagraph import NebulaDB
from database.milvus import MilvusDB, myMilvus
from database.nebulagraph import NebulaClient
from utils.base import file_exist


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="database info")
    parser.add_argument("--db", required=True, type=str, help="db name, e.g. test.")
    parser.add_argument(
        "--dim",
        # required=True,
        default=1024,
        type=str,
        help="vector dim, e.g. 1024",
    )

    parser.add_argument(
        "--clear", default=None, type=str, help="clear db. e.g. vector, kg, all"
    )

    parser.add_argument(
        "--create",
        default=None,
        type=str,
        choices=["vector", "kg", "all"],
        help="create a new database. e.g. vector, kg, all",
    )

    parser.add_argument(
        "--save_kg", default=None, type=str, help="dir to save the kg triplets."
    )

    parser.add_argument(
        "--delete",
        default=None,
        type=str,
        choices=["vector", "kg", "all"],
        help="delete a database. e.g. vector, kg, all",
    )

    parser.add_argument(
        "--info",
        default=None,
        type=str,
        choices=["vector", "kg"],
        help="print info of database. choose vector or kg",
    )

    args = parser.parse_args()
    db_name = args.db
    print(f"db_name: {db_name}")
    nebula_client = NebulaClient()
    milvus_client = myMilvus()
    nebula_client.show_space()
    milvus_client.show_all_collections()

    print("\n\n########################")
    print(f"db_name: {db_name}")
    print("########################")

    #### creat new database
    if args.create == "vector" or args.create == "all":
        milvus_db = MilvusDB(db_name=db_name, overwrite=True)
        milvus_db.create(consistency_level="Strong")
        milvus_client.show_all_collections()
        print(f"create milvus_db {db_name}")
    if args.create == "kg" or args.create == "all":
        nebula_client.create_space(db_name)
        nebula_client.show_space()
        print(f"create nebula_db {db_name}")
    #######################

    #### clear database
    if args.clear == "kg" or args.clear == "all":
        nebula_client.clear(db_name)
        print(f"clear nebula_db {db_name}")
    if args.clear == "vector" or args.clear == "all":
        milvus_client.drop(db_name)
        # milvus_db = MilvusDB(db_name=db_name, overwrite=True)
        print(f"clear milvus_db {db_name}")
    #######################

    #### delete database
    if args.delete == "kg" or args.delete == "all":
        nebula_client.drop_space(db_name)
        print(f"delete nebula_db {db_name}")
    if args.delete == "vector" or args.delete == "all":
        milvus_client.drop(db_name)
        print(f"delete milvus_db {db_name}")
    #######################

    #### show database info
    if args.info == "kg":
        nebula_client.count_edges(db_name)
        nebula_client.info(db_name)
    if args.info == "vector":
        collections = milvus_client.list_collections()
        if db_name not in collections:
            print(f"collection {db_name} does not exist. Existing collections: {collections}")
        else:
            milvus_client.show_collections_stats(db_name)
    #######################

    #### save kg triplets
    if args.save_kg:
        # dir_name = os.path.dirname(args.save_kg)
        assert file_exist(args.save_kg), f"{args.save_kg} not exist!"
        # file_path = os.path.join(args.save_kg, f"{args.db}.txt")
        # nebula_client.show_triplets(db_name, file_path)
        file_path = os.path.join(args.save_kg, f"{db_name}_triplets.json")
        nebula_client.save_triplets(db_name, file_path)
    #######################
