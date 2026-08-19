import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class TourismDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_script(self, script: str) -> None:
        with self.connect() as conn:
            conn.executescript(script)
            conn.commit()

    def insert_dataframe(self, table: str, df: pd.DataFrame) -> None:
        with self.connect() as conn:
            df.to_sql(table, conn, if_exists="append", index=False)
            conn.commit()

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
