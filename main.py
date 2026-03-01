from __future__ import annotations

import base64
import json
import os
import re
from typing import Annotated, Any, Dict, List, TypedDict
from urllib.parse import urlparse
from uuid import uuid4

import cloudinary
import cloudinary.uploader
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from mongo_store import MongoSettings, MongoStore
from scraper import run

load_dotenv()

RUN_ID = os.getenv("PIPELINE_RUN_ID", uuid4().hex)


# ─────────────────────────────────────────────
# ENV HELPERS
# ─────────────────────────────────────────────


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default



def float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0 <= value <= 1 else default



def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "autoflow")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ENABLE_IMAGE_GENERATION = bool_env("ENABLE_IMAGE_GENERATION", True)

ENABLE_CLOUDINARY_UPLOAD = bool_env("ENABLE_CLOUDINARY_UPLOAD", True)
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "").strip()
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()
CLOUDINARY_UPLOAD_PRESET = os.getenv("CLOUDINARY_UPLOAD_PRESET", "").strip()
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "autoflow/generated").strip()

ENABLE_INSTAGRAM_PUBLISH = bool_env("ENABLE_INSTAGRAM_PUBLISH", True)
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip()
INSTAGRAM_IG_USER_ID = os.getenv("INSTAGRAM_IG_USER_ID", "").strip()
INSTAGRAM_GRAPH_API_VERSION = os.getenv("INSTAGRAM_GRAPH_API_VERSION", "v25.0").strip() or "v25.0"

DEFAULT_SCRAPE_TOPICS = int_env("DEFAULT_SCRAPE_TOPICS", 3)
DEFAULT_POST_LIMIT = int_env("DEFAULT_POST_LIMIT", 1)
SCRAPER_MAX_LINKS_PER_TOPIC = int_env("SCRAPER_MAX_LINKS_PER_TOPIC", 3)
SCRAPER_HEADLESS = bool_env("SCRAPER_HEADLESS", False)
DUPLICATE_BUFFER_SIZE = int_env("DUPLICATE_BUFFER_SIZE", 300)
DUPLICATE_TITLE_SIMILARITY = float_env("DUPLICATE_TITLE_SIMILARITY", 0.72)
PENDING_PUBLISH_SCAN_LIMIT = int_env("PENDING_PUBLISH_SCAN_LIMIT", 200)
DEDUPE_EXTRA_POSTED_TITLES = [
    title.strip()
    for title in os.getenv("DEDUPE_EXTRA_POSTED_TITLES", "").split("||")
    if title.strip()
]
DEDUPE_EXTRA_POSTED_KEYWORDS = [
    normalize.strip()
    for normalize in (
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", keyword.lower())).strip()
        for keyword in os.getenv("DEDUPE_EXTRA_POSTED_KEYWORDS", "").split("||")
    )
    if normalize
]

mongo_store = MongoStore(MongoSettings(uri=MONGO_URI, db_name=MONGO_DB_NAME))


# ─────────────────────────────────────────────
# STATE MODELS
# ─────────────────────────────────────────────


class NewsItem(BaseModel):
    link: str
    heading: str
    content: str


class MergedNews(BaseModel):
    category: str
    canonical_title: str
    merged_content: str
    source_links: List[str]


class NewsList(BaseModel):
    news: List[MergedNews] = Field(description="A list of merged and summarized news objects")



def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    return {**existing, **new}


class State(TypedDict):
    raw_news_items: List[NewsItem]
    merged_news: List[MergedNews]
    generated_assets: Annotated[Dict[str, Any], merge_dicts]
    scrape_topics: int
    post_limit: int
    force_remake: bool


class PendingPublishItem(TypedDict):
    category: str
    canonical_title: str
    cloudinary_url: str
    cloudinary_public_id: str | None
    source_links: List[str]
    instagram_caption: str | None


# ─────────────────────────────────────────────
# CATEGORY DEFINITIONS
# ─────────────────────────────────────────────

CATEGORIES = """
  🎓 RESULT DECLARED     → exam results, marksheets, scorecards
  📝 ADMIT CARD          → hall tickets, exam entry passes
  📅 EXAM SCHEDULE       → timetables, date sheets, exam dates
  🔔 IMPORTANT NOTICE    → rule changes, policy updates, announcements
  💼 JOB OPENING         → govt jobs, recruitment, vacancies, apply online
  📄 ANSWER KEY          → provisional/final answer keys released
  🏆 MERIT LIST          → toppers, cut-off lists, selection lists
  💰 SCHOLARSHIP         → fellowships, financial aid, stipends
  🎯 ADMISSION OPEN      → college admissions, counselling, seat allotment
  ⚠️  DEADLINE ALERT      → last date to apply, fee payment, form submission
"""

CATEGORY_THEMES = """
  RESULT DECLARED  → Accent: mint (#52b788)
  ADMIT CARD       → Accent: sky blue (#74b9e8)
  EXAM SCHEDULE    → Accent: aqua (#2ec4b6)
  IMPORTANT NOTICE → Accent: lavender (#9b8fe8)
  JOB OPENING      → Accent: gold (#e9c46a)
  ANSWER KEY       → Accent: cyan (#48cae4)
  MERIT LIST       → Accent: violet (#c77dff)
  SCHOLARSHIP      → Accent: lime (#a7c957)
  ADMISSION OPEN   → Accent: coral (#ff6b6b)
  DEADLINE ALERT   → Accent: orange (#f4a261)
"""

DARK_GRADIENT_VARIANTS = [
    "#0d1117 → #1f2937",
    "#0b1320 → #1b263b",
    "#111827 → #1f2937",
    "#0f172a → #1e293b",
    "#10131a → #262b36",
    "#1a1029 → #2c1f45",
    "#0f1a12 → #1f3a2e",
    "#1a120f → #3b2a1f",
    "#13121a → #2a2640",
    "#111418 → #28313b",
]


# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model=GEMINI_TEXT_MODEL,
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY or None,
)

image_llm = ChatGoogleGenerativeAI(
    model=GEMINI_IMAGE_MODEL,
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY or None,
)


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

MERGE_PROMPT_TEMPLATE = """
### Role
You are a Professional News Synthesis Editor specializing in the Indian education and employment sector.
Your goal is to transform raw scraped news data into high-quality, deduplicated news summaries.

### Task
Analyze the provided list of raw news items and perform the following steps:

1. **Strict Filtering**: Retain ONLY news related to:
   - Job Openings / Recruitment / Careers
   - Education / Admissions / Results
   - Competitive Exams / Notifications
   - Discard all other topics immediately.

2. **Deduplication & Clustering**:
   - Identify news items describing the same event.
   - Group them into a single MergedNews object for that event.

3. **Information Synthesis**:
   - **Canonical Title**: Clear, click-worthy, factual title for the merged story.
   - **Merged Content**: Concise summary (100–150 words), preserving key details.
   - **Category**: Classify into EXACTLY ONE category from this list:
{categories}
   - **Source Links**: All unique URLs from the clustered sources.

### Quality Constraints
- Professional, journalistic tone.
- No repetitive sentences.
- Discard incomplete or vague items.
- category field must EXACTLY match one of the names above.
"""

merge_prompt = PromptTemplate(
    template=MERGE_PROMPT_TEMPLATE,
    input_variables=["categories"],
)

IMAGE_PROMPT_TEMPLATE = """
Create a social media news card (1080x1080px) for Indian students, optimized for Instagram and WhatsApp.

STEP 1 — APPLY VISUAL THEME
The category for this card is: {category}
Use category accent guidance:
{category_themes}

MANDATORY DARK GRADIENT FOR THIS CARD
Use this exact background gradient (dark style):
{dark_gradient_style}

STEP 2 — BUILD THE CARD (top to bottom, strict vertical stacking)

TOP SECTION:
- Rounded pill badge top-left: category emoji + name.
- Main headline directly below (2 lines max):
  "{title}"
- Thin horizontal divider below headline.

MIDDLE SECTION — KEY HIGHLIGHTS:
Small section label: "KEY HIGHLIGHTS".
Extract the 4-5 most critical points from:
{content}

Bullets rules:
- Rewrite each point in 10 words or fewer.
- ONE LINE per bullet.
- Bold the single most critical word in each bullet.

BOTTOM BOX — ACTION STEPS:
- Label: "WHAT YOU SHOULD DO NOW"
- Numbered list with EXACTLY 3 short steps.
- Numbering must be strictly sequential and valid:
  1. first line starts with "1."
  2. second line starts with "2."
  3. third line starts with "3."
- Never skip numbers, never repeat numbers, never output "4.".
- Each step must be one line only and 4-8 words.
- Include the most relevant URL from:
{sources}

FOOTER:
- Left side: source domains only.
- Right side: "EDUCATION UPDATE".

ABSOLUTE RULES:
- No illustrations or human figures.
- No spelling mistakes.
- Zero text overlap.
- Keep an authoritative, student-friendly dark visual style.
- Do not repeat words or phrases accidentally.
- Never output malformed numbering like "1,2,4" or duplicated numbers.
"""

image_prompt = PromptTemplate(
    template=IMAGE_PROMPT_TEMPLATE,
    input_variables=["category", "title", "content", "sources", "category_themes", "dark_gradient_style"],
)


# ─────────────────────────────────────────────
# GENERIC HELPERS
# ─────────────────────────────────────────────


def summarize_error(error: Exception, limit: int = 300) -> str:
    compact = re.sub(r"\s+", " ", str(error)).strip()
    return compact[:limit]



def normalize_title(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()



def title_tokens(value: str) -> set[str]:
    norm = normalize_title(value)
    return {token for token in norm.split() if len(token) > 2}



def title_similarity(a: str, b: str) -> float:
    tokens_a = title_tokens(a)
    tokens_b = title_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)



def normalize_source_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return ""
    host = parsed.netloc.lower().strip()
    path = parsed.path.rstrip("/").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{path}"



def pick_dark_gradient_style(title: str, category: str, index: int) -> str:
    key = f"{RUN_ID}|{category}|{title}|{index}"
    pos = abs(hash(key)) % len(DARK_GRADIENT_VARIANTS)
    return DARK_GRADIENT_VARIANTS[pos]



def prompt_text(prompt: str, default: str | None = None) -> str:
    try:
        raw = input(prompt)
    except EOFError:
        return default or ""
    value = raw.strip()
    if not value and default is not None:
        return default
    return value



def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    value = prompt_text(f"{prompt} {suffix}: ", default="y" if default else "n").lower()
    if value in {"y", "yes"}:
        return True
    if value in {"n", "no"}:
        return False
    return default



def prompt_int(prompt: str, default: int, min_value: int = 1, max_value: int = 100) -> int:
    while True:
        value = prompt_text(f"{prompt} [{default}]: ", default=str(default))
        try:
            parsed = int(value)
        except ValueError:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        if parsed < min_value or parsed > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return parsed



def parse_selection(selection: str, total_items: int) -> List[int]:
    raw = (selection or "").strip().lower()
    if not raw or raw == "all":
        return list(range(total_items))
    if raw in {"none", "n", "0"}:
        return []

    selected: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            if not (a_str.strip().isdigit() and b_str.strip().isdigit()):
                continue
            a = int(a_str)
            b = int(b_str)
            start = min(a, b)
            end = max(a, b)
            for idx in range(start, end + 1):
                if 1 <= idx <= total_items:
                    selected.add(idx - 1)
            continue

        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= total_items:
                selected.add(idx - 1)

    return sorted(selected)


# ─────────────────────────────────────────────
# DUPLICATE BUFFER HELPERS
# ─────────────────────────────────────────────


def load_recent_posted_cache() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    try:
        docs.extend(mongo_store.get_recent_posted_news(limit=DUPLICATE_BUFFER_SIZE))
        docs.extend(mongo_store.get_recent_posted_assets(limit=DUPLICATE_BUFFER_SIZE))
    except Exception as exc:
        print(f"[dedupe] WARNING: failed to load posted buffer: {summarize_error(exc)}")
        docs = []

    cache: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for doc in docs:
        title = str(doc.get("canonical_title", "")).strip()
        if not title:
            continue
        key = normalize_title(title)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        category = str(doc.get("category", "")).strip()
        source_links = [normalize_source_url(link) for link in doc.get("source_links", []) if isinstance(link, str)]
        cache.append(
            {
                "category": category,
                "title": title,
                "sources": {link for link in source_links if link},
            }
        )

    for title in DEDUPE_EXTRA_POSTED_TITLES:
        key = normalize_title(title)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        cache.append({"category": "MANUAL", "title": title, "sources": set()})
    return cache



def matches_posted_news(*, title: str, source_links: List[str], posted_cache: List[Dict[str, Any]]) -> bool:
    normalized_title = normalize_title(title)

    if DEDUPE_EXTRA_POSTED_KEYWORDS:
        for keyword in DEDUPE_EXTRA_POSTED_KEYWORDS:
            if keyword in normalized_title:
                return True

    if not posted_cache:
        return False

    normalized_sources = {normalize_source_url(link) for link in source_links if link}
    normalized_sources.discard("")

    for posted in posted_cache:
        posted_title = str(posted.get("title", ""))
        posted_sources = posted.get("sources", set())

        if normalized_sources and posted_sources and (normalized_sources & posted_sources):
            return True

        if title_similarity(title, posted_title) >= DUPLICATE_TITLE_SIMILARITY:
            return True

    return False


# ─────────────────────────────────────────────
# PERSISTENCE HELPERS
# ─────────────────────────────────────────────


def init_persistence() -> None:
    try:
        mongo_store.ensure_indexes()
        mongo_store.backfill_instagram_posted_flags()
        print(f"[mongo] Connected: {MONGO_DB_NAME}")
    except Exception as exc:
        print(f"[mongo] WARNING: persistence unavailable - {summarize_error(exc)}")



def init_cloudinary() -> bool:
    if not ENABLE_CLOUDINARY_UPLOAD:
        print("[cloudinary] Upload disabled (ENABLE_CLOUDINARY_UPLOAD=false)")
        return False
    if not (CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET):
        print("[cloudinary] Upload disabled (missing Cloudinary credentials)")
        return False

    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
    )
    print(f"[cloudinary] Configured for cloud '{CLOUDINARY_CLOUD_NAME}'")
    return True



def init_instagram() -> bool:
    if not ENABLE_INSTAGRAM_PUBLISH:
        print("[instagram] Publish disabled (ENABLE_INSTAGRAM_PUBLISH=false)")
        return False
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_IG_USER_ID:
        print("[instagram] Publish disabled (missing access token or IG user id)")
        return False
    print(f"[instagram] Configured for IG user '{INSTAGRAM_IG_USER_ID}'")
    return True



def persist_raw_news_items(raw_news: List[NewsItem]) -> None:
    try:
        mongo_store.upsert_raw_news([item.model_dump() for item in raw_news], run_id=RUN_ID)
    except Exception as exc:
        print(f"[mongo] WARNING: raw_news persistence failed - {summarize_error(exc)}")



def persist_merged_news_items(merged_news: List[MergedNews]) -> None:
    try:
        mongo_store.upsert_merged_news([item.model_dump() for item in merged_news], run_id=RUN_ID)
    except Exception as exc:
        print(f"[mongo] WARNING: merged_news persistence failed - {summarize_error(exc)}")



def persist_asset_item(
    *,
    category: str,
    canonical_title: str,
    local_path: str | None,
    source_links: List[str],
    status: str,
) -> None:
    try:
        mongo_store.upsert_asset(
            {
                "category": category,
                "canonical_title": canonical_title,
                "local_path": local_path,
                "source_links": source_links,
                "status": status,
            },
            run_id=RUN_ID,
        )

        merged_status: str | None = None
        if status == "saved":
            merged_status = "IMAGE_GENERATED"
        elif status == "skipped_already_posted":
            merged_status = "SKIPPED_ALREADY_POSTED"
        elif status in {"skipped_image_generation", "skipped_user_declined_generation"}:
            merged_status = "IMAGE_SKIPPED"
        elif status.startswith("error"):
            merged_status = "IMAGE_FAILED"

        if merged_status:
            mongo_store.set_merged_status(
                category=category,
                canonical_title=canonical_title,
                status=merged_status,
                run_id=RUN_ID,
            )
    except Exception as exc:
        print(f"[mongo] WARNING: assets persistence failed - {summarize_error(exc)}")



def persist_asset_upload(
    *,
    category: str,
    canonical_title: str,
    cloudinary_url: str | None,
    cloudinary_public_id: str | None,
    status: str,
) -> None:
    try:
        mongo_store.set_asset_upload(
            category=category,
            canonical_title=canonical_title,
            cloudinary_url=cloudinary_url,
            cloudinary_public_id=cloudinary_public_id,
            status=status,
            run_id=RUN_ID,
        )
        if status == "UPLOADED":
            mongo_store.set_merged_status(
                category=category,
                canonical_title=canonical_title,
                status="UPLOADED",
                run_id=RUN_ID,
            )
        elif status == "UPLOAD_FAILED":
            mongo_store.set_merged_status(
                category=category,
                canonical_title=canonical_title,
                status="UPLOAD_FAILED",
                run_id=RUN_ID,
            )
    except Exception as exc:
        print(f"[mongo] WARNING: cloudinary persistence failed - {summarize_error(exc)}")



def persist_asset_instagram(
    *,
    category: str,
    canonical_title: str,
    caption: str | None,
    instagram_creation_id: str | None,
    instagram_media_id: str | None,
    instagram_permalink: str | None,
    status: str,
    error: str | None,
) -> None:
    try:
        mongo_store.set_asset_instagram(
            category=category,
            canonical_title=canonical_title,
            caption=caption,
            instagram_creation_id=instagram_creation_id,
            instagram_media_id=instagram_media_id,
            instagram_permalink=instagram_permalink,
            status=status,
            error=error,
            run_id=RUN_ID,
        )

        if status == "INSTAGRAM_POSTED":
            mongo_store.set_merged_status(
                category=category,
                canonical_title=canonical_title,
                status="POSTED",
                run_id=RUN_ID,
            )
        elif status == "INSTAGRAM_FAILED":
            mongo_store.set_merged_status(
                category=category,
                canonical_title=canonical_title,
                status="POST_FAILED",
                run_id=RUN_ID,
            )
    except Exception as exc:
        print(f"[mongo] WARNING: instagram persistence failed - {summarize_error(exc)}")


# ─────────────────────────────────────────────
# IMAGE / INSTAGRAM HELPERS
# ─────────────────────────────────────────────


def _get_image_base64(response: AIMessage) -> str:
    image_block = next(
        block for block in response.content if isinstance(block, dict) and block.get("image_url")
    )
    url = image_block["image_url"].get("url", "")
    return url.split(",")[-1]



def save_base64_as_image(base64_str: str, title: str, folder: str = "output_images") -> str:
    os.makedirs(folder, exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    file_path = os.path.join(folder, f"{safe_name}.png")

    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    image_data = base64.b64decode(base64_str)
    with open(file_path, "wb") as f:
        f.write(image_data)

    print(f"  ✓ Saved: {file_path}")
    return file_path



def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "..."



def _domain_only(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""



def build_instagram_caption(
    *,
    category: str,
    title: str,
    merged_content: str,
    source_links: List[str],
) -> str:
    domains = [_domain_only(link) for link in source_links]
    domains = [d for d in domains if d]
    source_line = " | ".join(domains[:3])

    hashtags_by_category = {
        "RESULT DECLARED": "#ExamResults #StudentUpdate #EducationNews",
        "ADMIT CARD": "#AdmitCard #ExamAlert #StudentUpdate",
        "EXAM SCHEDULE": "#ExamSchedule #DateSheet #EducationNews",
        "IMPORTANT NOTICE": "#ImportantNotice #StudentAlert #EducationNews",
        "JOB OPENING": "#JobOpening #Recruitment #CareerUpdate",
        "ANSWER KEY": "#AnswerKey #ExamUpdate #StudentAlert",
        "MERIT LIST": "#MeritList #SelectionUpdate #EducationNews",
        "SCHOLARSHIP": "#Scholarship #FinancialAid #StudentSupport",
        "ADMISSION OPEN": "#AdmissionOpen #CollegeAdmission #EducationUpdate",
        "DEADLINE ALERT": "#DeadlineAlert #ApplyNow #StudentNotice",
    }

    summary = _truncate_words(merged_content, 55)
    hashtags = hashtags_by_category.get(category, "#EducationUpdate #StudentAlert #AutoFlow")

    lines = [
        f"{category}: {title}",
        "",
        summary,
        "",
        "Follow for daily education and job updates.",
        hashtags,
    ]
    if source_line:
        lines.extend(["", f"Sources: {source_line}"])

    return "\n".join(lines).strip()[:2200]


def publish_via_graph(*, cloudinary_url: str, caption: str) -> Dict[str, Any]:
    create_resp = graph_post(
        f"{INSTAGRAM_IG_USER_ID}/media",
        {
            "image_url": cloudinary_url,
            "caption": caption,
        },
    )
    creation_id = create_resp.get("id")
    if not creation_id:
        raise RuntimeError("Missing creation id from /media response")

    publish_resp = graph_post(
        f"{INSTAGRAM_IG_USER_ID}/media_publish",
        {"creation_id": creation_id},
    )
    media_id = publish_resp.get("id")
    if not media_id:
        raise RuntimeError("Missing media id from /media_publish response")

    permalink = None
    try:
        media_details = graph_get(str(media_id), {"fields": "id,permalink"})
        permalink = media_details.get("permalink")
    except Exception as details_exc:
        print(
            f"  [instagram] Could not fetch permalink for media {media_id}: "
            f"{summarize_error(details_exc)}"
        )

    return {
        "creation_id": str(creation_id),
        "media_id": str(media_id),
        "permalink": permalink,
    }


def _category_or_default(value: str | None) -> str:
    category = str(value or "").strip()
    return category if category else "IMPORTANT NOTICE"


def startup_pending_publish_queue() -> None:
    try:
        docs = mongo_store.get_pending_uploaded_assets(limit=PENDING_PUBLISH_SCAN_LIMIT)
    except Exception as exc:
        print(f"[pending] WARNING: failed to load pending cloudinary assets: {summarize_error(exc)}")
        return

    queue_by_category: Dict[str, List[PendingPublishItem]] = {}
    for doc in docs:
        title = str(doc.get("canonical_title", "")).strip()
        cloudinary_url = str(doc.get("cloudinary_url", "")).strip()
        if not title or not cloudinary_url:
            continue

        raw_sources = doc.get("source_links", [])
        if isinstance(raw_sources, list):
            source_links = [str(link).strip() for link in raw_sources if str(link).strip()]
        elif raw_sources:
            source_links = [str(raw_sources).strip()]
        else:
            source_links = []

        item: PendingPublishItem = {
            "category": _category_or_default(doc.get("category")),
            "canonical_title": title,
            "cloudinary_url": cloudinary_url,
            "cloudinary_public_id": doc.get("cloudinary_public_id"),
            "source_links": source_links,
            "instagram_caption": doc.get("instagram_caption"),
        }
        queue_by_category.setdefault(item["category"], []).append(item)

    if not queue_by_category:
        return

    print("\n" + "=" * 50)
    print("PENDING CLOUDINARY ASSETS (NOT POSTED YET)")
    print("=" * 50)

    flat_queue: List[PendingPublishItem] = []
    counter = 1
    for category in sorted(queue_by_category.keys()):
        items = queue_by_category[category]
        print(f"\n[{category}] {len(items)} item(s)")
        for item in items:
            print(f"  [{counter}] {item['canonical_title']}")
            print(f"      URL: {item['cloudinary_url']}")
            flat_queue.append(item)
            counter += 1

    if not prompt_yes_no("Pending Cloudinary URLs found. Publish any now", default=False):
        return

    if not init_instagram():
        print("[pending] Skipping pending publish because Instagram is not configured.")
        return

    selection_text = prompt_text(
        "Select pending item numbers to post (example: 1,3 or 1-2 or all or none)",
        default="all",
    )
    selected_indices = parse_selection(selection_text, len(flat_queue))
    if not selected_indices:
        print("[pending] No pending items selected.")
        return

    for idx in selected_indices:
        item = flat_queue[idx]
        category = item["category"]
        title = item["canonical_title"]

        if mongo_store.is_asset_posted(category=category, canonical_title=title):
            print(f"  [pending] Already posted, skipping: [{category}] {title}")
            continue

        caption = str(item.get("instagram_caption") or "").strip()
        if not caption:
            caption = build_instagram_caption(
                category=category,
                title=title,
                merged_content="",
                source_links=item.get("source_links", []),
            )

        try:
            result = publish_via_graph(
                cloudinary_url=item["cloudinary_url"],
                caption=caption,
            )
            print(f"  📸 Pending posted: {title} -> {result['media_id']}")
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=caption,
                instagram_creation_id=result["creation_id"],
                instagram_media_id=result["media_id"],
                instagram_permalink=result["permalink"],
                status="INSTAGRAM_POSTED",
                error=None,
            )
        except Exception as exc:
            short_error = summarize_error(exc)
            print(f"  📸 Pending publish failed for '{title}': {short_error}")
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=caption,
                instagram_creation_id=None,
                instagram_media_id=None,
                instagram_permalink=None,
                status="INSTAGRAM_FAILED",
                error=short_error,
            )



def _parse_graph_response(response: requests.Response) -> Dict[str, Any]:
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



def graph_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{INSTAGRAM_GRAPH_API_VERSION}/{path.lstrip('/')}"
    payload = {**data, "access_token": INSTAGRAM_ACCESS_TOKEN}
    response = requests.post(url, data=payload, timeout=60)
    return _parse_graph_response(response)



def graph_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{INSTAGRAM_GRAPH_API_VERSION}/{path.lstrip('/')}"
    query = {**params, "access_token": INSTAGRAM_ACCESS_TOKEN}
    response = requests.get(url, params=query, timeout=60)
    return _parse_graph_response(response)


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────


def run_scraper(state: State) -> dict:
    scrape_topics = max(1, int(state.get("scrape_topics") or DEFAULT_SCRAPE_TOPICS))
    force_remake = bool(state.get("force_remake", False))

    print(
        f"[scraper] topics={scrape_topics}, links/topic={SCRAPER_MAX_LINKS_PER_TOPIC}, "
        f"headless={SCRAPER_HEADLESS}"
    )
    data = run(
        max_topics=scrape_topics,
        max_links_per_topic=SCRAPER_MAX_LINKS_PER_TOPIC,
        headless=SCRAPER_HEADLESS,
    )

    raw_news: List[NewsItem] = [
        NewsItem(link=item["source"], heading=item["heading"], content=item["content"])
        for item in data
    ]

    if not force_remake:
        posted_cache = load_recent_posted_cache()
        filtered: List[NewsItem] = []
        skipped = 0
        for item in raw_news:
            if matches_posted_news(title=item.heading, source_links=[item.link], posted_cache=posted_cache):
                skipped += 1
                continue
            filtered.append(item)
        raw_news = filtered
        if skipped:
            print(f"[dedupe] Skipped {skipped} raw items already posted earlier.")

    persist_raw_news_items(raw_news)
    return {"raw_news_items": raw_news}



def mergeNews(state: State) -> dict:
    if not state.get("raw_news_items"):
        print("[merge] No raw news available after dedupe.")
        return {"merged_news": []}

    merger = llm.with_structured_output(NewsList)
    formatted_prompt = merge_prompt.format(categories=CATEGORIES)
    raw_data_string = json.dumps([item.model_dump() for item in state["raw_news_items"]])

    result = merger.invoke(
        [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=f"Raw Data: {raw_data_string}"),
        ]
    )

    merged_items: List[MergedNews] = []
    seen_keys: set[str] = set()
    for item in result.news:
        key = f"{item.category.lower()}|{normalize_title(item.canonical_title)}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_items.append(item)

    force_remake = bool(state.get("force_remake", False))
    if not force_remake:
        posted_cache = load_recent_posted_cache()
        filtered_items: List[MergedNews] = []
        skipped = 0
        for item in merged_items:
            if matches_posted_news(
                title=item.canonical_title,
                source_links=item.source_links,
                posted_cache=posted_cache,
            ):
                skipped += 1
                mongo_store.set_merged_status(
                    category=item.category,
                    canonical_title=item.canonical_title,
                    status="SKIPPED_ALREADY_POSTED",
                    run_id=RUN_ID,
                )
                continue
            filtered_items.append(item)
        merged_items = filtered_items
        if skipped:
            print(f"[dedupe] Skipped {skipped} merged stories already posted earlier.")

    persist_merged_news_items(merged_items)
    return {"merged_news": merged_items}



def storeImages(state: State) -> dict:
    new_assets: Dict[str, Any] = {}
    merged_news = state.get("merged_news", [])
    force_remake = bool(state.get("force_remake", False))

    if not merged_news:
        return {"generated_assets": new_assets}

    if not ENABLE_IMAGE_GENERATION:
        print("[image] Skipped image generation (ENABLE_IMAGE_GENERATION=false)")
        for item in merged_news:
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "sources": item.source_links,
                "status": "skipped_image_generation",
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=None,
                source_links=item.source_links,
                status="skipped_image_generation",
            )
        return {"generated_assets": new_assets}

    posted_cache = [] if force_remake else load_recent_posted_cache()
    items_to_generate: List[tuple[int, MergedNews]] = []

    for idx, item in enumerate(merged_news):
        already_posted = False
        if not force_remake:
            already_posted = mongo_store.is_asset_posted(
                category=item.category,
                canonical_title=item.canonical_title,
            ) or matches_posted_news(
                title=item.canonical_title,
                source_links=item.source_links,
                posted_cache=posted_cache,
            )

        if already_posted:
            existing = mongo_store.get_asset(category=item.category, canonical_title=item.canonical_title) or {}
            print(f"\n🖼  Skipping image (already posted): [{item.category}] {item.canonical_title}")
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": existing.get("local_path"),
                "sources": item.source_links,
                "status": "skipped_already_posted",
                "cloudinary_url": existing.get("cloudinary_url"),
                "cloudinary_public_id": existing.get("cloudinary_public_id"),
                "instagram_media_id": existing.get("instagram_media_id"),
                "instagram_permalink": existing.get("instagram_permalink"),
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=existing.get("local_path"),
                source_links=item.source_links,
                status="skipped_already_posted",
            )
            continue

        items_to_generate.append((idx, item))

    if not items_to_generate:
        print("[image] No fresh non-posted stories available for image generation.")
        return {"generated_assets": new_assets}

    print("\n" + "=" * 50)
    print("FRESH STORIES (NOT POSTED) - READY FOR IMAGE GENERATION")
    print("=" * 50)
    for i, (_, item) in enumerate(items_to_generate, 1):
        print(f"[{i}] [{item.category}] {item.canonical_title}")

    should_generate = prompt_yes_no("Generate images for these fresh stories now", default=False)
    if not should_generate:
        for _, item in items_to_generate:
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "sources": item.source_links,
                "status": "skipped_user_declined_generation",
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=None,
                source_links=item.source_links,
                status="skipped_user_declined_generation",
            )
        print("[image] Generation skipped by user.")
        return {"generated_assets": new_assets}

    selection_text = prompt_text(
        "Select story numbers to generate images for (example: 1,3 or 1-2 or all or none)",
        default="all",
    )
    selected_indices = parse_selection(selection_text, len(items_to_generate))
    selected_set = set(selected_indices)

    if not selected_set:
        for _, item in items_to_generate:
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "sources": item.source_links,
                "status": "skipped_user_declined_generation",
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=None,
                source_links=item.source_links,
                status="skipped_user_declined_generation",
            )
        print("[image] No stories selected for generation.")
        return {"generated_assets": new_assets}

    for position, (idx, item) in enumerate(items_to_generate):
        if position not in selected_set:
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "sources": item.source_links,
                "status": "skipped_user_declined_generation",
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=None,
                source_links=item.source_links,
                status="skipped_user_declined_generation",
            )
            continue

        print(f"\n🖼  Generating image for: [{item.category}] {item.canonical_title}")
        dark_gradient_style = pick_dark_gradient_style(item.canonical_title, item.category, idx)
        formatted_image_prompt = image_prompt.format(
            category=item.category,
            title=item.canonical_title,
            content=item.merged_content,
            sources=", ".join(item.source_links),
            category_themes=CATEGORY_THEMES,
            dark_gradient_style=dark_gradient_style,
        )

        try:
            response = image_llm.invoke(formatted_image_prompt)
            base64_data = _get_image_base64(response)
            filename_key = f"{item.category}_{item.canonical_title}"
            path_on_disk = save_base64_as_image(base64_data, filename_key)

            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": path_on_disk,
                "sources": item.source_links,
                "status": "saved",
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=path_on_disk,
                source_links=item.source_links,
                status="saved",
            )
        except Exception as exc:
            short_error = summarize_error(exc)
            print(f"  ✗ Failed for '{item.canonical_title}': {short_error}")
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "sources": item.source_links,
                "status": f"error: {short_error}",
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=None,
                source_links=item.source_links,
                status=f"error: {short_error}",
            )

    return {"generated_assets": new_assets}



def uploadToCloudinary(state: State) -> dict:
    uploaded_assets: Dict[str, Any] = {}
    is_cloudinary_ready = init_cloudinary()

    for title, asset in state.get("generated_assets", {}).items():
        category = str(asset.get("category", "")).strip()
        local_path = asset.get("local_path")
        status = str(asset.get("status", "")).strip()

        if not is_cloudinary_ready:
            uploaded_assets[title] = {
                **asset,
                "cloudinary_url": asset.get("cloudinary_url"),
                "cloudinary_public_id": asset.get("cloudinary_public_id"),
                "status": "upload_skipped_cloudinary_not_configured",
            }
            persist_asset_upload(
                category=category,
                canonical_title=title,
                cloudinary_url=asset.get("cloudinary_url"),
                cloudinary_public_id=asset.get("cloudinary_public_id"),
                status="UPLOAD_SKIPPED",
            )
            continue

        if status != "saved" or not local_path:
            uploaded_assets[title] = {
                **asset,
                "cloudinary_url": asset.get("cloudinary_url"),
                "cloudinary_public_id": asset.get("cloudinary_public_id"),
                "status": "upload_skipped_no_new_local_asset",
            }
            persist_asset_upload(
                category=category,
                canonical_title=title,
                cloudinary_url=asset.get("cloudinary_url"),
                cloudinary_public_id=asset.get("cloudinary_public_id"),
                status="UPLOAD_SKIPPED",
            )
            continue

        try:
            upload_kwargs: Dict[str, Any] = {"folder": CLOUDINARY_FOLDER}
            if CLOUDINARY_UPLOAD_PRESET:
                upload_kwargs["upload_preset"] = CLOUDINARY_UPLOAD_PRESET

            response = cloudinary.uploader.upload(local_path, **upload_kwargs)
            cloudinary_url = response.get("secure_url")
            cloudinary_public_id = response.get("public_id")

            uploaded_assets[title] = {
                **asset,
                "cloudinary_url": cloudinary_url,
                "cloudinary_public_id": cloudinary_public_id,
                "status": "uploaded",
            }
            print(f"  ☁ Uploaded: {title} -> {cloudinary_public_id}")
            persist_asset_upload(
                category=category,
                canonical_title=title,
                cloudinary_url=cloudinary_url,
                cloudinary_public_id=cloudinary_public_id,
                status="UPLOADED",
            )
        except Exception as exc:
            short_error = summarize_error(exc)
            uploaded_assets[title] = {
                **asset,
                "cloudinary_url": None,
                "cloudinary_public_id": None,
                "status": f"upload_error: {short_error}",
            }
            print(f"  ☁ Upload failed for '{title}': {short_error}")
            persist_asset_upload(
                category=category,
                canonical_title=title,
                cloudinary_url=None,
                cloudinary_public_id=None,
                status="UPLOAD_FAILED",
            )

    return {"generated_assets": uploaded_assets}



def publishToInstagram(state: State) -> dict:
    final_assets: Dict[str, Any] = {}
    is_instagram_ready = init_instagram()
    force_remake = bool(state.get("force_remake", False))
    post_limit = max(1, int(state.get("post_limit") or DEFAULT_POST_LIMIT))
    merged_lookup = {item.canonical_title: item for item in state.get("merged_news", [])}

    publish_candidates: List[Dict[str, Any]] = []

    for title, asset in state.get("generated_assets", {}).items():
        category = str(asset.get("category", "")).strip()
        cloudinary_url = str(asset.get("cloudinary_url") or "").strip()

        if not is_instagram_ready:
            final_assets[title] = {
                **asset,
                "instagram_status": "skipped_not_configured",
                "instagram_creation_id": None,
                "instagram_media_id": asset.get("instagram_media_id"),
                "instagram_permalink": asset.get("instagram_permalink"),
            }
            continue

        if not cloudinary_url:
            final_assets[title] = {
                **asset,
                "instagram_status": "skipped_no_cloudinary_url",
                "instagram_creation_id": None,
                "instagram_media_id": asset.get("instagram_media_id"),
                "instagram_permalink": asset.get("instagram_permalink"),
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=None,
                instagram_creation_id=None,
                instagram_media_id=asset.get("instagram_media_id"),
                instagram_permalink=asset.get("instagram_permalink"),
                status="INSTAGRAM_SKIPPED",
                error="cloudinary_url_missing",
            )
            continue

        if not force_remake and mongo_store.is_asset_posted(category=category, canonical_title=title):
            existing = mongo_store.get_asset(category=category, canonical_title=title) or {}
            existing_media_id = existing.get("instagram_media_id") or asset.get("instagram_media_id")
            existing_permalink = existing.get("instagram_permalink") or asset.get("instagram_permalink")
            final_assets[title] = {
                **asset,
                "instagram_status": "skipped_already_posted",
                "instagram_creation_id": None,
                "instagram_media_id": existing_media_id,
                "instagram_permalink": existing_permalink,
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=None,
                instagram_creation_id=None,
                instagram_media_id=existing_media_id,
                instagram_permalink=existing_permalink,
                status="INSTAGRAM_POSTED",
                error=None,
            )
            continue

        merged_item = merged_lookup.get(title)
        merged_content = merged_item.merged_content if merged_item else ""
        source_links = merged_item.source_links if merged_item else list(asset.get("sources", []))
        caption = build_instagram_caption(
            category=category or "IMPORTANT NOTICE",
            title=title,
            merged_content=merged_content,
            source_links=source_links,
        )

        publish_candidates.append(
            {
                "title": title,
                "category": category,
                "asset": asset,
                "cloudinary_url": cloudinary_url,
                "caption": caption,
            }
        )

    if not publish_candidates:
        return {"generated_assets": final_assets}

    print("\n" + "=" * 50)
    print("INSTAGRAM PUBLISH CANDIDATES")
    print("=" * 50)
    for i, candidate in enumerate(publish_candidates, 1):
        print(f"[{i}] [{candidate['category']}] {candidate['title']}")
        print(f"    URL: {candidate['cloudinary_url']}")

    should_publish = prompt_yes_no("Publish to Instagram now", default=False)
    if not should_publish:
        for candidate in publish_candidates:
            title = candidate["title"]
            category = candidate["category"]
            asset = candidate["asset"]
            final_assets[title] = {
                **asset,
                "instagram_status": "skipped_user_declined_now",
                "instagram_creation_id": None,
                "instagram_media_id": asset.get("instagram_media_id"),
                "instagram_permalink": asset.get("instagram_permalink"),
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=candidate["caption"],
                instagram_creation_id=None,
                instagram_media_id=asset.get("instagram_media_id"),
                instagram_permalink=asset.get("instagram_permalink"),
                status="INSTAGRAM_SKIPPED",
                error="user_declined_publish_now",
            )
        return {"generated_assets": final_assets}

    selection_text = prompt_text(
        "Select image numbers to post (example: 1,3 or 1-2 or all or none)",
        default="all",
    )
    selected_indices = parse_selection(selection_text, len(publish_candidates))

    if len(selected_indices) > post_limit:
        print(f"[instagram] Limiting selected posts to {post_limit} as requested.")
        selected_indices = selected_indices[:post_limit]

    selected_set = set(selected_indices)

    for idx, candidate in enumerate(publish_candidates):
        title = candidate["title"]
        category = candidate["category"]
        asset = candidate["asset"]

        if idx not in selected_set:
            final_assets[title] = {
                **asset,
                "instagram_status": "skipped_user_not_selected",
                "instagram_creation_id": None,
                "instagram_media_id": asset.get("instagram_media_id"),
                "instagram_permalink": asset.get("instagram_permalink"),
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=candidate["caption"],
                instagram_creation_id=None,
                instagram_media_id=asset.get("instagram_media_id"),
                instagram_permalink=asset.get("instagram_permalink"),
                status="INSTAGRAM_SKIPPED",
                error="user_not_selected",
            )
            continue

        try:
            create_resp = graph_post(
                f"{INSTAGRAM_IG_USER_ID}/media",
                {
                    "image_url": candidate["cloudinary_url"],
                    "caption": candidate["caption"],
                },
            )
            creation_id = create_resp.get("id")
            if not creation_id:
                raise RuntimeError("Missing creation id from /media response")

            publish_resp = graph_post(
                f"{INSTAGRAM_IG_USER_ID}/media_publish",
                {"creation_id": creation_id},
            )
            media_id = publish_resp.get("id")
            if not media_id:
                raise RuntimeError("Missing media id from /media_publish response")

            permalink = None
            try:
                media_details = graph_get(str(media_id), {"fields": "id,permalink"})
                permalink = media_details.get("permalink")
            except Exception as details_exc:
                print(
                    f"  [instagram] Could not fetch permalink for media {media_id}: "
                    f"{summarize_error(details_exc)}"
                )

            print(f"  📸 Instagram posted: {title} -> {media_id}")
            final_assets[title] = {
                **asset,
                "instagram_status": "posted",
                "instagram_creation_id": str(creation_id),
                "instagram_media_id": str(media_id),
                "instagram_permalink": permalink,
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=candidate["caption"],
                instagram_creation_id=str(creation_id),
                instagram_media_id=str(media_id),
                instagram_permalink=permalink,
                status="INSTAGRAM_POSTED",
                error=None,
            )
        except Exception as exc:
            short_error = summarize_error(exc)
            print(f"  📸 Instagram publish failed for '{title}': {short_error}")
            final_assets[title] = {
                **asset,
                "instagram_status": f"failed: {short_error}",
                "instagram_creation_id": None,
                "instagram_media_id": None,
                "instagram_permalink": None,
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=candidate["caption"],
                instagram_creation_id=None,
                instagram_media_id=None,
                instagram_permalink=None,
                status="INSTAGRAM_FAILED",
                error=short_error,
            )

    return {"generated_assets": final_assets}


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

g = StateGraph(State)
g.add_node("scraper", run_scraper)
g.add_node("merger", mergeNews)
g.add_node("store", storeImages)
g.add_node("cloudinary", uploadToCloudinary)
g.add_node("instagram", publishToInstagram)

g.add_edge(START, "scraper")
g.add_edge("scraper", "merger")
g.add_edge("merger", "store")
g.add_edge("store", "cloudinary")
g.add_edge("cloudinary", "instagram")
g.add_edge("instagram", END)

app = g.compile()


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────


def run_app() -> None:
    init_persistence()
    startup_pending_publish_queue()

    scrape_topics = prompt_int(
        "How many trending topics to scrape this run",
        default=DEFAULT_SCRAPE_TOPICS,
        min_value=1,
        max_value=20,
    )
    post_limit = prompt_int(
        "How many posts max to push to Instagram this run",
        default=DEFAULT_POST_LIMIT,
        min_value=1,
        max_value=20,
    )
    force_remake = prompt_yes_no(
        "Remake mode (allow already posted news to regenerate/repost)",
        default=False,
    )

    print(
        "\n[run] "
        f"topics={scrape_topics}, post_limit={post_limit}, force_remake={force_remake}, "
        f"duplicate_buffer={DUPLICATE_BUFFER_SIZE}, similarity={DUPLICATE_TITLE_SIMILARITY}"
    )

    try:
        final_state = app.invoke(
            {
                "merged_news": [],
                "raw_news_items": [],
                "generated_assets": {},
                "scrape_topics": scrape_topics,
                "post_limit": post_limit,
                "force_remake": force_remake,
            }
        )

        print("\n" + "=" * 50)
        print("MERGED NEWS RESULTS")
        print("=" * 50)

        if not final_state.get("merged_news"):
            print("No relevant new news found.")
            return

        for i, news in enumerate(final_state["merged_news"], 1):
            print(f"\n[{i}] CATEGORY : {news.category}")
            print(f"    TITLE    : {news.canonical_title}")
            print(f"    SUMMARY  : {news.merged_content[:120]}...")
            print(f"    SOURCES  : {', '.join(news.source_links)}")

        print("\n" + "=" * 50)
        print("GENERATED ASSETS")
        print("=" * 50)

        for title, asset in final_state.get("generated_assets", {}).items():
            print(f"\n  [{asset.get('category', '')}] {title}")
            print(f"  Status : {asset.get('status')}")
            print(f"  Path   : {asset.get('local_path')}")
            if asset.get("cloudinary_url"):
                print(f"  URL    : {asset.get('cloudinary_url')}")
                print(f"  PubID  : {asset.get('cloudinary_public_id')}")
            if asset.get("instagram_status"):
                print(f"  IG     : {asset.get('instagram_status')}")
            if asset.get("instagram_media_id"):
                print(f"  IG_ID  : {asset.get('instagram_media_id')}")
            if asset.get("instagram_permalink"):
                print(f"  IG_URL : {asset.get('instagram_permalink')}")
    finally:
        mongo_store.close()


if __name__ == "__main__":
    run_app()
