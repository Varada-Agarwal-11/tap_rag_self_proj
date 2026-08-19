from pathlib import Path

import pandas as pd

from src.db import TourismDB
from src.query_parser import parse_query
from src.retriever import SQLRetriever


def test_parser():
    intent = parse_query(
        "Find a hotel in Riyadh under 800 SAR for 3 people for 2 days"
    )
    assert intent.city == "Riyadh"
    assert intent.category == "hotels"
    assert intent.budget == 800
    assert intent.people == 3
    assert intent.days == 2


def test_sql_retrieval(tmp_path):
    db = TourismDB(str(tmp_path / "test.db"))
    schema = Path("src/schema.sql").read_text(encoding="utf-8")
    db.execute_script(schema)

    db.insert_dataframe(
        "hotels",
        pd.DataFrame([
            {
                "name": "Test Hotel",
                "price_range": "Budget",
                "min_price": 300,
                "max_price": 500,
                "rating": 4.5,
                "location": "Riyadh",
                "description": "Central hotel",
            }
        ]),
    )

    rows = SQLRetriever(db).retrieve_hotels(
        parse_query("hotel in Riyadh under 800 SAR")
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "Test Hotel"
