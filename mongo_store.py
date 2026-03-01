from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, Iterable, List

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class MongoSettings:
    uri: str
    db_name: str


class MongoStore:
    """Simple persistence layer for AutoFlow pipeline artifacts."""
    def __init__(self, settings: MongoSettings) -> None:
        self._settings = settings
        self._client: MongoClient | None = None
        self._db: Database | None = None

    @property
    def db(self) -> Database:
        if self._db is None:
            self._client = MongoClient(self._settings.uri)
            self._db = self._client[self._settings.db_name]
        return self._db

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None

    def ensure_indexes(self) -> None:
        raw = self.db["raw_news"]
        merged = self.db["merged_news"]
        assets = self.db["assets"]

        raw.create_index([("dedup_hash", ASCENDING)], unique=True, name="uq_raw_news_dedup")
        raw.create_index([("scraped_at", DESCENDING)], name="idx_raw_news_scraped_at")

        merged.create_index([("merge_hash", ASCENDING)], unique=True, name="uq_merged_news_hash")
        merged.create_index([("status", ASCENDING)], name="idx_merged_news_status")
        merged.create_index([("updated_at", DESCENDING)], name="idx_merged_news_updated_at")

        assets.create_index([("asset_hash", ASCENDING)], unique=True, name="uq_assets_hash")
        assets.create_index([("status", ASCENDING)], name="idx_assets_status")
        assets.create_index([("updated_at", DESCENDING)], name="idx_assets_updated_at")
        assets.create_index([("instagram_media_id", ASCENDING)], name="idx_assets_instagram_media_id")
        assets.create_index([("instagram_posted", ASCENDING)], name="idx_assets_instagram_posted")

    def upsert_raw_news(self, items: Iterable[Dict[str, str]], run_id: str) -> List[Dict[str, Any]]:
        col: Collection = self.db["raw_news"]
        now = _utc_now()
        stored: List[Dict[str, Any]] = []

        for item in items:
            source_url = item.get("link", "").strip()
            heading = item.get("heading", "").strip()
            content = item.get("content", "").strip()
            if not heading or not content:
                continue

            dedup_hash = _hash_text(f"{heading.lower()}|{source_url.lower()}")
            doc = {
                "dedup_hash": dedup_hash,
                "source_url": source_url,
                "heading": heading,
                "content": content,
                "run_id": run_id,
                "scraped_at": now,
                "updated_at": now,
            }

            col.update_one(
                {"dedup_hash": dedup_hash},
                {
                    "$setOnInsert": {"created_at": now},
                    "$set": doc,
                },
                upsert=True,
            )
            stored.append(doc)

        return stored

    def upsert_merged_news(self, items: Iterable[Dict[str, Any]], run_id: str) -> List[Dict[str, Any]]:
        col: Collection = self.db["merged_news"]
        now = _utc_now()
        stored: List[Dict[str, Any]] = []

        for item in items:
            category = str(item.get("category", "")).strip()
            canonical_title = str(item.get("canonical_title", "")).strip()
            merged_content = str(item.get("merged_content", "")).strip()
            source_links = list(item.get("source_links", []))
            if not category or not canonical_title or not merged_content:
                continue

            merge_hash = _hash_text(f"{category.lower()}|{canonical_title.lower()}")
            doc = {
                "merge_hash": merge_hash,
                "category": category,
                "canonical_title": canonical_title,
                "merged_content": merged_content,
                "source_links": source_links,
                "status": "MERGED",
                "run_id": run_id,
                "updated_at": now,
            }

            col.update_one(
                {"merge_hash": merge_hash},
                {
                    "$setOnInsert": {"created_at": now},
                    "$set": doc,
                },
                upsert=True,
            )
            stored.append(doc)

        return stored

    def set_merged_status(self, *, category: str, canonical_title: str, status: str, run_id: str) -> None:
        col: Collection = self.db["merged_news"]
        now = _utc_now()
        merge_hash = _hash_text(f"{category.lower()}|{canonical_title.lower()}")
        col.update_one(
            {"merge_hash": merge_hash},
            {
                "$setOnInsert": {"created_at": now},
                "$set": {
                    "merge_hash": merge_hash,
                    "category": category,
                    "canonical_title": canonical_title,
                    "status": status,
                    "run_id": run_id,
                    "updated_at": now,
                },
            },
            upsert=True,
        )

    def upsert_asset(self, item: Dict[str, Any], run_id: str) -> Dict[str, Any] | None:
        col: Collection = self.db["assets"]
        now = _utc_now()

        category = str(item.get("category", "")).strip()
        canonical_title = str(item.get("canonical_title", "")).strip()
        local_path = item.get("local_path")
        status = str(item.get("status", "saved")).strip()
        source_links = list(item.get("source_links", []))

        if not canonical_title:
            return None

        asset_hash = _hash_text(f"{category.lower()}|{canonical_title.lower()}")
        doc = {
            "asset_hash": asset_hash,
            "category": category,
            "canonical_title": canonical_title,
            "local_path": local_path,
            "source_links": source_links,
            "status": status,
            "run_id": run_id,
            "updated_at": now,
        }

        col.update_one(
            {"asset_hash": asset_hash},
            {
                "$setOnInsert": {
                    "created_at": now,
                    "instagram_posted": False,
                },
                "$set": doc,
            },
            upsert=True,
        )
        return doc

    def set_asset_upload(
        self,
        *,
        category: str,
        canonical_title: str,
        cloudinary_url: str | None,
        cloudinary_public_id: str | None,
        status: str,
        run_id: str,
    ) -> None:
        col: Collection = self.db["assets"]
        now = _utc_now()
        asset_hash = _hash_text(f"{category.lower()}|{canonical_title.lower()}")
        col.update_one(
            {"asset_hash": asset_hash},
            {
                "$setOnInsert": {
                    "created_at": now,
                    "instagram_posted": False,
                },
                "$set": {
                    "asset_hash": asset_hash,
                    "category": category,
                    "canonical_title": canonical_title,
                    "cloudinary_url": cloudinary_url,
                    "cloudinary_public_id": cloudinary_public_id,
                    "status": status,
                    "run_id": run_id,
                    "updated_at": now,
                },
            },
            upsert=True,
        )

    def set_asset_instagram(
        self,
        *,
        category: str,
        canonical_title: str,
        caption: str | None,
        instagram_creation_id: str | None,
        instagram_media_id: str | None,
        instagram_permalink: str | None,
        status: str,
        error: str | None,
        run_id: str,
    ) -> None:
        col: Collection = self.db["assets"]
        now = _utc_now()
        asset_hash = _hash_text(f"{category.lower()}|{canonical_title.lower()}")
        is_posted = status == "INSTAGRAM_POSTED" or bool(instagram_media_id)
        set_fields: Dict[str, Any] = {
            "asset_hash": asset_hash,
            "category": category,
            "canonical_title": canonical_title,
            "instagram_caption": caption,
            "instagram_creation_id": instagram_creation_id,
            "instagram_media_id": instagram_media_id,
            "instagram_permalink": instagram_permalink,
            "instagram_error": error,
            "status": status,
            "run_id": run_id,
            "updated_at": now,
        }
        if is_posted:
            set_fields["instagram_posted"] = True

        col.update_one(
            {"asset_hash": asset_hash},
            {
                "$setOnInsert": {
                    "created_at": now,
                },
                "$set": set_fields,
            },
            upsert=True,
        )

    def get_recent_posted_news(self, limit: int = 200) -> List[Dict[str, Any]]:
        col: Collection = self.db["merged_news"]
        cursor = (
            col.find(
                {"status": "POSTED"},
                {
                    "_id": 0,
                    "category": 1,
                    "canonical_title": 1,
                    "source_links": 1,
                    "updated_at": 1,
                },
            )
            .sort("updated_at", DESCENDING)
            .limit(max(1, limit))
        )
        return list(cursor)

    def get_recent_posted_assets(self, limit: int = 200) -> List[Dict[str, Any]]:
        col: Collection = self.db["assets"]
        cursor = (
            col.find(
                {
                    "$or": [
                        {"status": "INSTAGRAM_POSTED"},
                        {"instagram_media_id": {"$exists": True, "$nin": [None, ""]}},
                    ]
                },
                {
                    "_id": 0,
                    "category": 1,
                    "canonical_title": 1,
                    "source_links": 1,
                    "updated_at": 1,
                },
            )
            .sort("updated_at", DESCENDING)
            .limit(max(1, limit))
        )
        return list(cursor)

    def get_pending_uploaded_assets(self, limit: int = 200) -> List[Dict[str, Any]]:
        col: Collection = self.db["assets"]
        cursor = (
            col.find(
                {
                    "cloudinary_url": {"$exists": True, "$type": "string", "$nin": ["", None]},
                    "instagram_posted": {"$ne": True},
                    "$and": [
                        {"status": {"$ne": "INSTAGRAM_POSTED"}},
                        {
                            "$or": [
                                {"instagram_media_id": {"$exists": False}},
                                {"instagram_media_id": None},
                                {"instagram_media_id": ""},
                            ]
                        },
                    ],
                },
                {
                    "_id": 0,
                    "category": 1,
                    "canonical_title": 1,
                    "cloudinary_url": 1,
                    "cloudinary_public_id": 1,
                    "source_links": 1,
                    "instagram_caption": 1,
                    "updated_at": 1,
                },
            )
            .sort("updated_at", DESCENDING)
            .limit(max(1, limit))
        )
        return list(cursor)

    def backfill_instagram_posted_flags(self) -> None:
        col: Collection = self.db["assets"]
        col.update_many(
            {
                "$or": [
                    {"status": "INSTAGRAM_POSTED"},
                    {"instagram_media_id": {"$exists": True, "$nin": [None, ""]}},
                ]
            },
            {"$set": {"instagram_posted": True}},
        )

    def is_asset_posted(self, *, category: str, canonical_title: str) -> bool:
        col: Collection = self.db["assets"]
        asset_hash = _hash_text(f"{category.lower()}|{canonical_title.lower()}")
        doc = col.find_one(
            {
                "asset_hash": asset_hash,
                "$or": [
                    {"status": "INSTAGRAM_POSTED"},
                    {"instagram_media_id": {"$exists": True, "$nin": [None, ""]}},
                ],
            },
            {"_id": 1},
        )
        return doc is not None

    def get_asset(self, *, category: str, canonical_title: str) -> Dict[str, Any] | None:
        col: Collection = self.db["assets"]
        asset_hash = _hash_text(f"{category.lower()}|{canonical_title.lower()}")
        doc = col.find_one(
            {"asset_hash": asset_hash},
            {
                "_id": 0,
                "category": 1,
                "canonical_title": 1,
                "local_path": 1,
                "source_links": 1,
                "status": 1,
                "cloudinary_url": 1,
                "cloudinary_public_id": 1,
                "instagram_media_id": 1,
                "instagram_permalink": 1,
            },
        )
        return doc
