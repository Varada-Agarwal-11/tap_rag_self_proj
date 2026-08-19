import re
from dataclasses import dataclass, field


KNOWN_CITIES = [
    "Riyadh", "Jeddah", "Mecca", "Medina",
    "Dammam", "Dubai", "Abu Dhabi",
]

PLACE_WORDS = [
    "place", "places", "attraction", "attractions", "visit",
    "sightseeing", "museum", "park", "landmark", "tourist",
]

RESTAURANT_WORDS = [
    "restaurant", "restaurants", "food", "dining", "cuisine",
    "eat", "eating", "meal",
]

HOTEL_WORDS = [
    "hotel", "hotels", "stay", "lodging", "accommodation",
    "room", "rooms",
]


@dataclass
class QueryIntent:
    raw_query: str
    city: str | None = None
    category: str = "all"
    cuisine: str | None = None
    place_type: str | None = None
    budget: float | None = None
    people: int | None = None
    days: int | None = None
    preferences: list[str] = field(default_factory=list)


def extract_number(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, flags=re.I)
    if not match:
        return None
    try:
        return cast(match.group(1))
    except (TypeError, ValueError):
        return None


def parse_query(query: str) -> QueryIntent:
    q = query.lower()
    intent = QueryIntent(raw_query=query)

    for city in KNOWN_CITIES:
        if city.lower() in q:
            intent.city = city
            break

    place_score = sum(word in q for word in PLACE_WORDS)
    restaurant_score = sum(word in q for word in RESTAURANT_WORDS)
    hotel_score = sum(word in q for word in HOTEL_WORDS)

    if hotel_score > max(place_score, restaurant_score):
        intent.category = "hotels"
    elif restaurant_score > max(place_score, hotel_score):
        intent.category = "restaurants"
    elif place_score > max(hotel_score, restaurant_score):
        intent.category = "places"

    budget = extract_number(
        r"(?:under|below|less than|max(?:imum)?|budget(?: of)?|within)"
        r"\s*(?:sar|rs\.?|₹|\$)?\s*(\d+(?:\.\d+)?)",
        query,
        float,
    )
    if budget is None:
        budget = extract_number(
            r"(\d+(?:\.\d+)?)\s*(?:sar|rs\.?|₹|\$)",
            query,
            float,
        )
    intent.budget = budget

    intent.people = extract_number(
        r"(\d+)\s*(?:people|persons|guests|travellers|travelers)",
        query,
        int,
    )

    intent.days = extract_number(
        r"(\d+)\s*(?:day|days|night|nights)",
        query,
        int,
    )

    cuisine_match = re.search(
        r"\b(indian|arabic|saudi|italian|chinese|japanese|mexican|thai|lebanese|american|seafood)\b",
        q,
    )
    if cuisine_match:
        intent.cuisine = cuisine_match.group(1)

    for pref in [
        "cheap", "budget", "luxury", "cultural", "family",
        "romantic", "outdoor", "historical", "shopping",
        "highly rated", "best", "quiet", "central",
    ]:
        if pref in q:
            intent.preferences.append(pref)

    return intent
