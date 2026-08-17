"""Offline verification for isolated Nebula L2 snapshot preparation."""
import unittest

from scripts.experiments.nebula_clone import clone_l2


class Value:
    def __init__(self, value):
        self.value = value

    def cast(self):
        return self.value


class Result:
    def __init__(self, rows=None, error=""):
        self._rows = rows or []
        self._error = error

    def is_succeeded(self):
        return not self._error

    def error_msg(self):
        return self._error

    def keys(self):
        return list(self._rows[0]) if self._rows else []

    def column_values(self, key):
        return [Value(row[key]) for row in self._rows]


class Session:
    def __init__(self):
        self.target_vertices = 0
        self.target_edges = 0
        self.queries = []

    def execute(self, query):
        self.queries.append(query)
        if query == "SHOW SPACES;":
            return Result([{"Name": "source"}, {"Name": "target"}])
        if "SHOW TAGS" in query:
            return Result([{"Name": "entity"}])
        if "SHOW EDGES" in query:
            return Result([{"Name": "relationship"}])
        if query == "USE target;":
            return Result()
        if "CREATE TAG" in query or "CREATE EDGE" in query:
            return Result()
        if "USE source; MATCH (v) RETURN count(*)" in query:
            return Result([{"n": 2}])
        if "USE source; MATCH ()-[e]->() RETURN count(*)" in query:
            return Result([{"n": 1}])
        if "USE target; MATCH (v) RETURN count(*)" in query:
            return Result([{"n": self.target_vertices}])
        if "USE target; MATCH ()-[e]->() RETURN count(*)" in query:
            return Result([{"n": self.target_edges}])
        if "MATCH (v:entity)" in query and "SKIP 0 LIMIT 2" in query:
            return Result([
                {"vid": 11, "name": 'A "quoted" name', "type": "PERSON", "source_chunk": "c1"},
                {"vid": 22, "name": "B", "type": "PLACE", "source_chunk": "c2"},
            ])
        if "MATCH (v:entity)" in query:
            return Result()
        if "MATCH ()-[e:relationship]" in query and "SKIP 0 LIMIT 2" in query:
            return Result([{"src": 11, "dst": 22, "edge_rank": 7,
                            "relation": "KNOWS", "source_chunk": "c1"}])
        if "MATCH ()-[e:relationship]" in query:
            return Result()
        if "INSERT VERTEX" in query:
            self.target_vertices = 2
            return Result()
        if "INSERT EDGE" in query:
            self.target_edges = 1
            return Result()
        return Result(error=f"unexpected query: {query}")


class NebulaCloneTest(unittest.TestCase):
    def test_clone_preserves_counts_rank_and_escaping(self):
        session = Session()
        report = clone_l2(session, "source", "target", batch_size=2)
        self.assertTrue(report["verified"])
        self.assertEqual(report["target_after"], {"vertices": 2, "edges": 1})
        vertex_insert = next(q for q in session.queries if "INSERT VERTEX" in q)
        edge_insert = next(q for q in session.queries if "INSERT EDGE" in q)
        self.assertIn('A \\"quoted\\" name', vertex_insert)
        self.assertIn("11->22@7", edge_insert)

    def test_partial_target_is_rejected(self):
        session = Session()
        session.target_vertices = 1
        with self.assertRaisesRegex(RuntimeError, "partially populated"):
            clone_l2(session, "source", "target", batch_size=2)

    def test_empty_mode_creates_verified_empty_space(self):
        session = Session()
        report = clone_l2(session, "source", "target", empty=True)
        self.assertEqual(report["mode"], "empty")
        self.assertEqual(report["target_after"], {"vertices": 0, "edges": 0})


if __name__ == "__main__":
    unittest.main()
