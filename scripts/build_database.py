from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DB_PATH
from src.db import TourismDB


def main():
    db = TourismDB(DB_PATH)
    schema = (ROOT / "src" / "schema.sql").read_text(encoding="utf-8")
    db.execute_script(schema)

    with db.connect() as conn:
        conn.execute("DELETE FROM places")
        conn.execute("DELETE FROM restaurants")
        conn.execute("DELETE FROM hotels")
        conn.commit()

    seed_dir = ROOT / "data" / "seed"
    db.insert_dataframe("places", pd.read_csv(seed_dir / "places.csv"))
    db.insert_dataframe("restaurants", pd.read_csv(seed_dir / "restaurants.csv"))
    db.insert_dataframe("hotels", pd.read_csv(seed_dir / "hotels.csv"))

    print(f"Database created at: {DB_PATH}")


if __name__ == "__main__":
    main()
