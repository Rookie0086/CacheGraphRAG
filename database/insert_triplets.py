import argparse

# import json
import multiprocessing
import time

from database.nebulagraph import NebulaDB
from utils.base import read_json


def insert_triple(pid, triplets, graph_db: NebulaDB, verbose=False, log=False):
    start_time = time.time()
    # for i, triplet in tqdm(enumerate(triplets), f"insert triplets in {db_names}"):
    for i, triplet in enumerate(triplets):
        if i and i % 10000 == 0 and verbose:
            print(
                f"processor {pid} insert {i}/{len(triplets)} triplets, cost {time.time() - start_time : .3f}s."
            )
        graph_db.upsert_triplet(triplet[0], triplet[1], triplet[2])
    end_time = time.time()
    print(
        f"pid {pid} insert {len(triplets)} triplets cost {end_time - start_time : .3f}s."
    )


def parallel_insert(triplets, db_name, nproc=5, reuse=False):
    start_time = time.time()
    processes = []
    n_triplets = len(triplets)
    if n_triplets < 100:
        nproc = 1
    step = (n_triplets + nproc - 1) // nproc

    print(n_triplets, db_name, nproc, step)
    print(f"\ninsert {n_triplets} triplets in {db_name}, nproc={nproc}")

    nebula_db = NebulaDB(db_name)
    if not reuse:
        nebula_db.clear()

    for i in range(nproc):
        start = i * step
        end = min(start + step, n_triplets)
        print(f"pid {i} take {start}-{end}")
        p = multiprocessing.Process(
            target=insert_triple,
            args=(
                i,
                triplets[start:end],
                nebula_db,
                True,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()

    print(f"insert_triple_parallel cost {end_time - start_time:.3f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", type=str, default=None, help="database name, e.g. test."
    )
    parser.add_argument("--reuse", action="store_true", help="reuse database")
    # parser.add_argument("--data", type=str, required=True, help="dataset name.")
    parser.add_argument(
        "--input", type=str, required=True, help="input triplet json file."
    )
    parser.add_argument(
        "--proc",
        type=int,
        # required=True,
        default=1,
        help="processor numbers, e.g. test.",
    )

    args = parser.parse_args()
    # if not args.db:
    #     args.db = args.data

    print(args)

    # filename = f"../triplet/triplets/{args.data}_triplets.json"
    filename = args.input
    loaded_triplets = read_json(filename)
    loaded_triplets = [(str(x), str(y), str(z)) for x, y, z in loaded_triplets]
    loaded_triplets = list(set(loaded_triplets))

    for triplet in loaded_triplets:
        assert len(triplet) == 3
        for x in triplet:
            assert len(x) > 0, triplet
    print(f"load {len(loaded_triplets)} triplets from {filename}.")

    parallel_insert(loaded_triplets, args.db, args.proc, args.reuse)
