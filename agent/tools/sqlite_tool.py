import os
import sqlite3
from typing import List, Tuple, Any, Dict

DEFAULT_DB_PATH = os.environ.get("NORTHWIND_DB", os.path.join(os.getcwd(), "data", "northwind.sqlite"))

class SQLiteTool:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path

    def _connect(self):
        return sqlite3.connect(self.path)

    def get_schema(self) -> Dict[str, List[str]]:
        con = self._connect()
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        schema: Dict[str, List[str]] = {}
        for t in tables:
            cur.execute(f"PRAGMA table_info('{t}');")
            cols = [r[1] for r in cur.fetchall()]
            schema[t] = cols
        con.close()
        return schema

    def execute(self, sql: str, params: Tuple = ()) -> List[Tuple[Any, ...]]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        con.close()
        return rows

    def explain(self, sql: str) -> List[Tuple]:
        con = self._connect()
        cur = con.cursor()
        cur.execute(f"EXPLAIN QUERY PLAN {sql}")
        r = cur.fetchall()
        con.close()
        return r
