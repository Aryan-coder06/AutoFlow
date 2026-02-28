from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict
from uuid import uuid4

import requests
from dotenv import load_dotenv

from mongo_store import MongoSettings, MongoStore


def summarize_error(error: Exception) -> str:
    compact = re.sub(r"\s+", " ", str(error)).strip()
    return compact[:300]


def parse_graph_response(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(f"Graph API HTTP {response.status_code}: {response.text[:300]}")

    if response.status_code >= 400 or "error" in payload:
        err = payload.get("error", {})
        raise RuntimeError(
            "Graph API error "
            f"type={err.get('type')} "
            f"code={err.get('code')} "
            f"subcode={err.get('error_subcode')} "
            f"message={err.get('message')} "
            f"trace={err.get('fbtrace_id')}"
        )
    return payload


def graph_post(api_version: str, path: str, access_token: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{api_version}/{path.lstrip('/')}"
    payload = {**data, "access_token": access_token}
    response = requests.post(url, data=payload, timeout=60)
    return parse_graph_response(response)


def graph_get(api_version: str, path: str, access_token: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{api_version}/{path.lstrip('/')}"
    query = {**params, "access_token": access_token}
    response = requests.get(url, params=query, timeout=60)
    return parse_graph_response(response)


def default_caption(category: str, title: str) -> str:
    category_text = (category or "IMPORTANT NOTICE").strip()
    lines = [
        f"{category_text}: {title}",
        "",
        "Follow for daily education and job updates.",
        "#EducationUpdate #StudentAlert #AutoFlow",
    ]
    return "\n".join(lines).strip()[:2200]


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Publish already-uploaded Cloudinary assets to Instagram using Mongo asset records "
            "(without running scraper/image generation)."
        )
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of assets to publish in this run.")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also retry assets currently marked as INSTAGRAM_FAILED.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise RuntimeError("--limit must be >= 1")

    mongo_uri = require_env("MONGODB_URI")
    mongo_db_name = require_env("MONGODB_DB_NAME")
    access_token = require_env("INSTAGRAM_ACCESS_TOKEN")
    ig_user_id = require_env("INSTAGRAM_IG_USER_ID")
    graph_api_version = os.getenv("INSTAGRAM_GRAPH_API_VERSION", "v25.0").strip() or "v25.0"

    run_id = os.getenv("PIPELINE_RUN_ID", uuid4().hex)

    store = MongoStore(MongoSettings(uri=mongo_uri, db_name=mongo_db_name))
    try:
        assets_col = store.db["assets"]
        query: Dict[str, Any] = {
            "cloudinary_url": {"$exists": True, "$type": "string", "$ne": ""},
            "$or": [
                {"instagram_media_id": {"$exists": False}},
                {"instagram_media_id": None},
                {"instagram_media_id": ""},
            ],
        }
        if not args.include_failed:
            query["status"] = {"$ne": "INSTAGRAM_FAILED"}

        candidates = list(
            assets_col.find(query).sort("updated_at", 1).limit(args.limit)
        )
        if not candidates:
            print("No pending uploaded assets found to publish.")
            return

        print(f"Found {len(candidates)} pending assets to publish.")

        published = 0
        failed = 0

        for doc in candidates:
            category = str(doc.get("category", "")).strip()
            canonical_title = str(doc.get("canonical_title", "")).strip()
            cloudinary_url = str(doc.get("cloudinary_url", "")).strip()
            if not canonical_title or not cloudinary_url:
                continue

            caption = str(doc.get("instagram_caption") or "").strip()
            if not caption:
                caption = default_caption(category, canonical_title)

            print(f"\nPosting: [{category}] {canonical_title}")
            try:
                create_resp = graph_post(
                    graph_api_version,
                    f"{ig_user_id}/media",
                    access_token,
                    {"image_url": cloudinary_url, "caption": caption},
                )
                creation_id = create_resp.get("id")
                if not creation_id:
                    raise RuntimeError("Missing creation id from /media response")

                publish_resp = graph_post(
                    graph_api_version,
                    f"{ig_user_id}/media_publish",
                    access_token,
                    {"creation_id": creation_id},
                )
                media_id = publish_resp.get("id")
                if not media_id:
                    raise RuntimeError("Missing media id from /media_publish response")

                permalink = None
                try:
                    details = graph_get(
                        graph_api_version,
                        str(media_id),
                        access_token,
                        {"fields": "id,permalink"},
                    )
                    permalink = details.get("permalink")
                except Exception as details_exc:
                    print(f"  Warning: permalink fetch failed: {summarize_error(details_exc)}")

                store.set_asset_instagram(
                    category=category,
                    canonical_title=canonical_title,
                    caption=caption,
                    instagram_creation_id=str(creation_id),
                    instagram_media_id=str(media_id),
                    instagram_permalink=permalink,
                    status="INSTAGRAM_POSTED",
                    error=None,
                    run_id=run_id,
                )
                store.set_merged_status(
                    category=category,
                    canonical_title=canonical_title,
                    status="POSTED",
                    run_id=run_id,
                )

                published += 1
                print(f"  Posted media_id={media_id}")
                if permalink:
                    print(f"  Permalink={permalink}")

            except Exception as exc:
                short_error = summarize_error(exc)
                failed += 1
                print(f"  Failed: {short_error}")

                store.set_asset_instagram(
                    category=category,
                    canonical_title=canonical_title,
                    caption=caption,
                    instagram_creation_id=None,
                    instagram_media_id=None,
                    instagram_permalink=None,
                    status="INSTAGRAM_FAILED",
                    error=short_error,
                    run_id=run_id,
                )
                store.set_merged_status(
                    category=category,
                    canonical_title=canonical_title,
                    status="POST_FAILED",
                    run_id=run_id,
                )

        print("\nDone.")
        print(f"Published: {published}")
        print(f"Failed   : {failed}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
