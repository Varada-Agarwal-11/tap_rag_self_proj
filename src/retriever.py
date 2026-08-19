from __future__ import annotations

from typing import Any

import numpy as np

from config import MAX_CANDIDATES, TOP_K
from src.db import TourismDB
from src.query_parser import QueryIntent


class SQLRetriever:
    def __init__(self, db: TourismDB):
        self.db = db

    @staticmethod
    def _normalize_city(value: str | None):
        return value.lower().strip() if value else None

    def _location_clause(self, column: str, city: str | None):
        if not city:
            return "", []
        return f" AND LOWER({column}) LIKE ?", [f"%{self._normalize_city(city)}%"]

    def retrieve_hotels(self, intent: QueryIntent) -> list[dict[str, Any]]:
        sql = """
        SELECT id, name, price_range, min_price, max_price, rating,
               location, description, 'hotel' AS category
        FROM hotels
        WHERE 1=1
        """
        params: list[Any] = []

        clause, values = self._location_clause("location", intent.city)
        sql += clause
        params.extend(values)

        if intent.budget is not None:
            sql += " AND min_price <= ?"
            params.append(intent.budget)

        if "luxury" in intent.preferences:
            sql += " AND rating >= 4.3"

        sql += " ORDER BY rating DESC, min_price ASC LIMIT ?"
        params.append(MAX_CANDIDATES)
        return self.db.query(sql, tuple(params))

    def retrieve_restaurants(self, intent: QueryIntent) -> list[dict[str, Any]]:
        sql = """
        SELECT id, name, cuisine, price_range, min_price, max_price,
               rating, location, description, 'restaurant' AS category
        FROM restaurants
        WHERE 1=1
        """
        params: list[Any] = []

        clause, values = self._location_clause("location", intent.city)
        sql += clause
        params.extend(values)

        if intent.budget is not None:
            sql += " AND min_price <= ?"
            params.append(intent.budget)

        if intent.cuisine:
            sql += " AND LOWER(cuisine) LIKE ?"
            params.append(f"%{intent.cuisine}%")

        sql += " ORDER BY rating DESC, min_price ASC LIMIT ?"
        params.append(MAX_CANDIDATES)
        return self.db.query(sql, tuple(params))

    def retrieve_places(self, intent: QueryIntent) -> list[dict[str, Any]]:
        sql = """
        SELECT id, name, location, type, rating,
               famous_for, description, 'place' AS category
        FROM places
        WHERE 1=1
        """
        params: list[Any] = []

        clause, values = self._location_clause("location", intent.city)
        sql += clause
        params.extend(values)

        sql += " ORDER BY rating DESC LIMIT ?"
        params.append(MAX_CANDIDATES)
        return self.db.query(sql, tuple(params))

    def retrieve(self, intent: QueryIntent) -> dict[str, list[dict[str, Any]]]:
        result = {"hotels": [], "restaurants": [], "places": []}

        if intent.category in ("all", "hotels"):
            result["hotels"] = self.retrieve_hotels(intent)

        if intent.category in ("all", "restaurants"):
            result["restaurants"] = self.retrieve_restaurants(intent)

        if intent.category in ("all", "places"):
            result["places"] = self.retrieve_places(intent)

        return result


def text_for_item(item: dict[str, Any]) -> str:
    category = item["category"]

    if category == "hotel":
        return (
            f"Hotel {item['name']}. Location: {item['location']}. "
            f"Price: {item['price_range']} ({item['min_price']}-{item['max_price']} SAR per night). "
            f"Rating: {item['rating']} out of 5. {item.get('description', '')}"
        )

    if category == "restaurant":
        return (
            f"Restaurant {item['name']}. Cuisine: {item['cuisine']}. "
            f"Location: {item['location']}. Price: {item['price_range']} "
            f"({item['min_price']}-{item['max_price']} SAR). "
            f"Rating: {item['rating']} out of 5. {item.get('description', '')}"
        )

    return (
        f"Place {item['name']}. Type: {item['type']}. "
        f"Location: {item['location']}. Rating: {item['rating']} out of 5. "
        f"Famous for: {item.get('famous_for', '')}. {item.get('description', '')}"
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticReranker:
    def __init__(self, encoder):
        self.encoder = encoder

    def rank(
        self,
        query: str,
        grouped_candidates: dict[str, list[dict[str, Any]]],
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        all_items = [
            item
            for group in grouped_candidates.values()
            for item in group
        ]
        if not all_items:
            return []

        texts = [text_for_item(item) for item in all_items]
        query_vec = self.encoder.embed(query)
        item_vecs = self.encoder.embed(texts)

        scored = []
        for item, vector in zip(all_items, item_vecs):
            record = dict(item)
            record["semantic_score"] = cosine_similarity(query_vec, vector)
            scored.append(record)

        scored.sort(
            key=lambda x: (x["semantic_score"], x["rating"]),
            reverse=True,
        )
        return scored[:top_k]
