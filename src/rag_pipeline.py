from __future__ import annotations

from typing import Any

from config import MODEL_NAME, TOP_K
from src.db import TourismDB
from src.query_parser import parse_query
from src.retriever import SQLRetriever, SemanticReranker, text_for_item
from src.t5_model import T5Engine


class TouristRAG:
    def __init__(self, db_path: str):
        self.db = TourismDB(db_path)
        self.model = T5Engine(MODEL_NAME)
        self.retriever = SQLRetriever(self.db)
        self.reranker = SemanticReranker(self.model)

    def retrieve(self, query: str) -> dict[str, Any]:
        intent = parse_query(query)
        sql_results = self.retriever.retrieve(intent)
        ranked = self.reranker.rank(query, sql_results, top_k=TOP_K)
        return {
            "intent": intent,
            "sql_results": sql_results,
            "ranked": ranked,
        }

    @staticmethod
    def _build_header(intent) -> str:
        return (
            f"HEADER:\n"
            f"city={intent.city or 'any'}\n"
            f"category={intent.category}\n"
            f"budget={intent.budget if intent.budget is not None else 'not specified'}\n"
            f"people={intent.people if intent.people is not None else 'not specified'}\n"
            f"days={intent.days if intent.days is not None else 'not specified'}\n"
            f"cuisine={intent.cuisine or 'not specified'}\n"
            f"preferences={', '.join(intent.preferences) if intent.preferences else 'none'}\n"
        )

    @staticmethod
    def _build_context(ranked: list[dict[str, Any]]) -> str:
        if not ranked:
            return "No matching database records were found."

        return "\n\n".join(
            f"[RESULT {i}]\n{text_for_item(item)}\n"
            f"retrieval_score={item.get('semantic_score', 0):.4f}"
            for i, item in enumerate(ranked, 1)
        )

    def build_generation_prompt(self, query: str, data: dict[str, Any]) -> str:
        intent = data["intent"]
        return f"""
You are a grounded tourist recommendation assistant.

Use ONLY the database information in CONTEXT.
Do not invent prices, ratings, locations, cuisines, hotel facilities, or attractions.
If the database does not contain enough information, say so.

{self._build_header(intent)}

USER QUERY:
{query}

CONTEXT:
{self._build_context(data["ranked"])}

TASK:
Give a useful, concise recommendation.
Start with a direct recommendation.
Then list the best matching options with their rating, location, and price/cuisine when available.
Briefly explain why each option matches the user's request.
For a multi-category trip, organize the answer into Hotels, Restaurants, and Places.
""".strip()

    def answer(self, query: str) -> dict[str, Any]:
        data = self.retrieve(query)
        prompt = self.build_generation_prompt(query, data)
        answer = self.model.generate(prompt)
        return {
            "answer": answer,
            "intent": data["intent"],
            "sql_results": data["sql_results"],
            "ranked": data["ranked"],
            "prompt": prompt,
        }
