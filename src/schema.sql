PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS places (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    location TEXT NOT NULL,
    type TEXT NOT NULL,
    rating REAL NOT NULL CHECK(rating >= 0 AND rating <= 5),
    famous_for TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    cuisine TEXT NOT NULL,
    price_range TEXT NOT NULL,
    min_price REAL NOT NULL,
    max_price REAL NOT NULL,
    rating REAL NOT NULL CHECK(rating >= 0 AND rating <= 5),
    location TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS hotels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    price_range TEXT NOT NULL,
    min_price REAL NOT NULL,
    max_price REAL NOT NULL,
    rating REAL NOT NULL CHECK(rating >= 0 AND rating <= 5),
    location TEXT NOT NULL,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_places_location ON places(location);
CREATE INDEX IF NOT EXISTS idx_places_type ON places(type);
CREATE INDEX IF NOT EXISTS idx_restaurants_location ON restaurants(location);
CREATE INDEX IF NOT EXISTS idx_restaurants_cuisine ON restaurants(cuisine);
CREATE INDEX IF NOT EXISTS idx_hotels_location ON hotels(location);
