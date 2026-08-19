from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "tourist.db"))
MODEL_NAME = os.getenv("MODEL_NAME", "google-t5/t5-small")
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "12"))
TOP_K = int(os.getenv("TOP_K", "5"))
