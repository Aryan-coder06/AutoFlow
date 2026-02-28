from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Annotated
import json
import base64
import os
import re
from urllib.parse import urlparse
from uuid import uuid4
import cloudinary
import cloudinary.uploader
import requests
from scraper import run
from mongo_store import MongoStore, MongoSettings

load_dotenv()

RUN_ID = os.getenv("PIPELINE_RUN_ID", uuid4().hex)


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "autoflow")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ENABLE_IMAGE_GENERATION = os.getenv("ENABLE_IMAGE_GENERATION", "true").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_CLOUDINARY_UPLOAD = os.getenv("ENABLE_CLOUDINARY_UPLOAD", "true").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_INSTAGRAM_PUBLISH = os.getenv("ENABLE_INSTAGRAM_PUBLISH", "true").strip().lower() in {"1", "true", "yes", "y"}
INSTAGRAM_MAX_POSTS = int_env("INSTAGRAM_MAX_POSTS", 1)

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "").strip()
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()
CLOUDINARY_UPLOAD_PRESET = os.getenv("CLOUDINARY_UPLOAD_PRESET", "").strip()
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "autoflow/generated").strip()
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip()
INSTAGRAM_IG_USER_ID = os.getenv("INSTAGRAM_IG_USER_ID", "").strip()
INSTAGRAM_GRAPH_API_VERSION = os.getenv("INSTAGRAM_GRAPH_API_VERSION", "v25.0").strip() or "v25.0"

mongo_store = MongoStore(MongoSettings(uri=MONGO_URI, db_name=MONGO_DB_NAME))


# ─────────────────────────────────────────────
# STATE MODELS
# ─────────────────────────────────────────────

class NewsItem(BaseModel):
    link: str
    heading: str
    content: str

class MergedNews(BaseModel):
    category: str           # e.g. "RESULT DECLARED", "JOB OPENING"
    canonical_title: str
    merged_content: str
    source_links: List[str]

class NewsList(BaseModel):
    """Container for a list of merged news items."""
    news: List[MergedNews] = Field(description="A list of merged and summarized news objects")

def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merges two dictionaries. If keys collide, the new value wins."""
    return {**existing, **new}

class State(TypedDict):
    raw_news_items: List[NewsItem]
    merged_news: List[MergedNews]
    generated_assets: Annotated[Dict[str, Any], merge_dicts]


# ─────────────────────────────────────────────
# CATEGORY DEFINITIONS  (single source of truth)
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
  RESULT DECLARED  → Background: deep green  (#1a472a → #2d6a4f) | Accent: mint (#52b788)
  ADMIT CARD       → Background: deep navy   (#0d1b3e → #1a3a6b) | Accent: sky blue (#74b9e8)
  EXAM SCHEDULE    → Background: dark teal   (#0d3b38 → #1a5c57) | Accent: aqua (#2ec4b6)
  IMPORTANT NOTICE → Background: deep indigo (#1a1a4e → #2d2d7a) | Accent: lavender (#9b8fe8)
  JOB OPENING      → Background: dark maroon (#3b0d0d → #6b1a1a) | Accent: gold (#e9c46a)
  ANSWER KEY       → Background: dark slate  (#1a2535 → #2d3f5a) | Accent: cyan (#48cae4)
  MERIT LIST       → Background: deep purple (#2d0d3b → #4a1a6b) | Accent: violet (#c77dff)
  SCHOLARSHIP      → Background: dark olive  (#2b3a0d → #4a6320) | Accent: lime (#a7c957)
  ADMISSION OPEN   → Background: deep ocean  (#0d2b3b → #1a4f6b) | Accent: coral (#ff6b6b)
  DEADLINE ALERT   → Background: dark amber  (#3b2000 → #6b3a00) | Accent: orange (#f4a261)
"""


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
# MERGE PROMPT
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
   - Discard all other topics (general politics, sports, entertainment) immediately.

2. **Deduplication & Clustering**:
   - Identify news items describing the same event (e.g., multiple sources on the same UPSC result).
   - Group them into a single MergedNews object for that event.

3. **Information Synthesis**:
   - **Canonical Title**: Clear, click-worthy, factual title for the merged story.
   - **Merged Content**: Concise summary (100–150 words). Do NOT lose critical details such as
     deadlines, eligibility criteria, exam dates, or official website names.
   - **Category**: Classify into EXACTLY ONE category from this list (return name only, no emoji, no arrow):
{categories}
   - **Source Links**: All unique URLs from the clustered sources.

### Quality Constraints
- Professional, journalistic tone.
- No repetitive sentences.
- Discard incomplete or vague news items.
- All source links must be valid strings.
- category field must EXACTLY match one of the names above (e.g. "RESULT DECLARED", "JOB OPENING").
"""

merge_prompt = PromptTemplate(
    template=MERGE_PROMPT_TEMPLATE,
    input_variables=['categories']
)


# ─────────────────────────────────────────────
# IMAGE GENERATION PROMPT
# ─────────────────────────────────────────────

IMAGE_PROMPT_TEMPLATE = """
Create a social media news card (1080x1080px) for Indian students, optimized for Instagram and WhatsApp.

STEP 1 — APPLY VISUAL THEME:
The category for this card is: {category}

Match it to the visual theme below and apply those exact colors:
{category_themes}

STEP 2 — BUILD THE CARD (top to bottom, strict vertical stacking):

TOP SECTION:
- Rounded pill badge top-left: category emoji + name (e.g. "🎓 RESULT DECLARED")
  in accent color, uppercase, small bold, semi-transparent white border
- Main headline directly below in large bold white text (2 lines max):
  "{title}"
- Thin horizontal white divider (15% opacity) below the headline

MIDDLE SECTION — KEY HIGHLIGHTS:
Small section label: "KEY HIGHLIGHTS" in tiny accent-color uppercase letters.
Extract the 4-5 most critical points from the content below and display as bullets:

CONTENT:
{content}

Each bullet gets a colored circle icon on the left based on meaning:
  ✓  Accent color circle → confirmed facts, key announcements
  🌐 Sky blue circle     → website, portal, how to access
  ⚠  Yellow circle       → warnings, delays, cautions
  ✕  Red circle          → restrictions, not available
  📅 White circle        → dates, deadlines, schedules
  ↩  Mint circle         → next steps, what to do next

Bullet rules:
- Rewrite each point in 10 words or fewer — never copy-paste
- ONE LINE per bullet — no wrapping under any circumstance
- Bold the single most critical word in each bullet

BOTTOM BOX — ACTION STEPS:
Dark semi-transparent rounded box (black at 20% opacity), clear padding inside.
Label: "📋 WHAT YOU SHOULD DO NOW" in small accent-color uppercase.
Numbered list (accent-color circle numbers), 3-4 steps extracted from the content.
Each step: max 10 words, starts with a verb (Check / Visit / Download / Apply / Submit / Register).

Below the box: accent-color outlined pill badge showing the most relevant official URL from:
{sources}

FOOTER:
Thin white divider (10% opacity).
Left side: small italic muted text → domain names only from the sources
  (e.g. "timesofindia.com · ndtv.com · mathrubhumi.com")
Right side: small bold uppercase → "EDUCATION UPDATE"

ABSOLUTE DESIGN RULES — NEVER VIOLATE:
— NO illustrations, human figures, characters, or decorative images anywhere
— Every single word must be perfectly spelled — proofread before rendering
— Zero text overlapping — all elements stacked vertically with clear spacing
— Font: clean geometric sans-serif (Poppins or equivalent) throughout
— Use ONLY the background gradient and accent color for the detected category
— Additional allowed colors: white (#ffffff) for body text,
  yellow (#e9c46a) for warnings, red-orange (#e76f51) for alerts
— Minimum 28px padding on all four sides of the card
— Card must be information-rich but never visually cluttered
— Flat solid color on all text — zero text gradients
— Mood: authoritative, urgent, trustworthy, student-friendly
"""

image_prompt = PromptTemplate(
    template=IMAGE_PROMPT_TEMPLATE,
    input_variables=['category', 'title', 'content', 'sources', 'category_themes']
)


# ─────────────────────────────────────────────
# PERSISTENCE HELPERS
# ─────────────────────────────────────────────

def init_persistence() -> None:
    try:
        mongo_store.ensure_indexes()
        print(f"[mongo] Connected: {MONGO_DB_NAME}")
    except Exception as exc:
        print(f"[mongo] WARNING: persistence unavailable - {exc}")


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


def persist_raw_news_items(raw_news: List[NewsItem]) -> None:
    try:
        mongo_store.upsert_raw_news(
            [item.model_dump() for item in raw_news],
            run_id=RUN_ID,
        )
    except Exception as exc:
        print(f"[mongo] WARNING: raw_news persistence failed - {exc}")


def persist_merged_news_items(merged_news: List[MergedNews]) -> None:
    try:
        mongo_store.upsert_merged_news(
            [item.model_dump() for item in merged_news],
            run_id=RUN_ID,
        )
    except Exception as exc:
        print(f"[mongo] WARNING: merged_news persistence failed - {exc}")


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
        mongo_store.set_merged_status(
            category=category,
            canonical_title=canonical_title,
            status="IMAGE_GENERATED" if status == "saved" else "IMAGE_FAILED",
            run_id=RUN_ID,
        )
    except Exception as exc:
        print(f"[mongo] WARNING: assets persistence failed - {exc}")


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
        mongo_store.set_merged_status(
            category=category,
            canonical_title=canonical_title,
            status="UPLOADED" if status == "UPLOADED" else "UPLOAD_FAILED",
            run_id=RUN_ID,
        )
    except Exception as exc:
        print(f"[mongo] WARNING: cloudinary persistence failed - {exc}")


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
        print(f"[mongo] WARNING: instagram persistence failed - {exc}")


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def run_scraper(state: State) -> dict:
    data = run()
    raw_news: List[NewsItem] = []
    for item in data:
        news = NewsItem(
            link=item['source'],
            heading=item['heading'],
            content=item['content']
        )
        raw_news.append(news)

    persist_raw_news_items(raw_news)
    return {"raw_news_items": raw_news}


def mergeNews(state: State) -> dict:
    merger = llm.with_structured_output(NewsList)

    # Inject categories into the merge prompt
    formatted_prompt = merge_prompt.format(categories=CATEGORIES)

    raw_data_string = json.dumps([item.model_dump() for item in state['raw_news_items']])

    result = merger.invoke([
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Raw Data: {raw_data_string}"),
    ])

    persist_merged_news_items(result.news)
    return {'merged_news': result.news}


# ─────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────

def _get_image_base64(response: AIMessage) -> str:
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    url = image_block["image_url"].get("url", "")
    return url.split(",")[-1]


def save_base64_as_image(base64_str: str, title: str, folder: str = "output_images") -> str:
    """Decodes base64 string and saves PNG to local folder. Returns the file path."""
    os.makedirs(folder, exist_ok=True)

    safe_name = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
    file_path = os.path.join(folder, f"{safe_name}.png")

    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    with open(file_path, "wb") as f:
        f.write(image_data)

    print(f"  ✓ Saved: {file_path}")
    return file_path


def summarize_model_error(error: Exception) -> str:
    message = str(error)
    if "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
        return "quota_exhausted: Gemini image quota exceeded. Enable billing or change model/key."
    compact = re.sub(r"\s+", " ", message).strip()
    return compact[:300]


def summarize_cloudinary_error(error: Exception) -> str:
    compact = re.sub(r"\s+", " ", str(error)).strip()
    return compact[:300]


def summarize_instagram_error(error: Exception) -> str:
    compact = re.sub(r"\s+", " ", str(error)).strip()
    return compact[:300]


def init_instagram() -> bool:
    if not ENABLE_INSTAGRAM_PUBLISH:
        print("[instagram] Publish disabled (ENABLE_INSTAGRAM_PUBLISH=false)")
        return False
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_IG_USER_ID:
        print("[instagram] Publish disabled (missing access token or IG user id)")
        return False
    print(f"[instagram] Configured for IG user '{INSTAGRAM_IG_USER_ID}'")
    return True


def _domain_only(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""




def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "..."


def build_instagram_caption(
    *,
    category: str,
    title: str,
    merged_content: str,
    source_links: List[str],
) -> str:
    domain_parts = [_domain_only(link) for link in source_links]
    domains = [d for d in domain_parts if d]
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

    caption = "\n".join(lines).strip()
    return caption[:2200]


def _parse_graph_response(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        text = response.text[:300]
        raise RuntimeError(f"Graph API HTTP {response.status_code}: {text}")

    if response.status_code >= 400 or "error" in payload:
        err = payload.get("error", {})
        message = err.get("message", "Unknown Graph API error")
        code = err.get("code")
        subcode = err.get("error_subcode")
        err_type = err.get("type")
        fbtrace = err.get("fbtrace_id")
        raise RuntimeError(
            f"Graph API error type={err_type} code={code} subcode={subcode} "
            f"message={message} trace={fbtrace}"
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
# IMAGE GENERATION NODE
# ─────────────────────────────────────────────

def storeImages(state: State) -> dict:
    new_assets = {}

    if not ENABLE_IMAGE_GENERATION:
        print("[image] Skipped image generation (ENABLE_IMAGE_GENERATION=false)")
        for item in state['merged_news']:
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "sources": item.source_links,
                "status": "skipped_image_generation"
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=None,
                source_links=item.source_links,
                status="skipped_image_generation",
            )
        return {"generated_assets": new_assets}

    for item in state['merged_news']:
        print(f"\n🖼  Generating image for: [{item.category}] {item.canonical_title}")

        # Build the fully formatted prompt
        formatted_image_prompt = image_prompt.format(
            category=item.category,
            title=item.canonical_title,
            content=item.merged_content,
            sources=", ".join(item.source_links),
            category_themes=CATEGORY_THEMES
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
                "status": "saved"
            }
            persist_asset_item(
                category=item.category,
                canonical_title=item.canonical_title,
                local_path=path_on_disk,
                source_links=item.source_links,
                status="saved",
            )

        except Exception as e:
            short_error = summarize_model_error(e)
            print(f"  ✗ Failed for '{item.canonical_title}': {short_error}")
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "status": f"error: {short_error}"
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
        category = asset.get("category", "")
        source_links = asset.get("sources", [])
        local_path = asset.get("local_path")
        status = str(asset.get("status", ""))

        if not is_cloudinary_ready:
            uploaded_assets[title] = {
                **asset,
                "cloudinary_url": None,
                "cloudinary_public_id": None,
                "status": "upload_skipped_cloudinary_not_configured",
            }
            persist_asset_upload(
                category=category,
                canonical_title=title,
                cloudinary_url=None,
                cloudinary_public_id=None,
                status="UPLOAD_FAILED",
            )
            continue

        if not local_path or status != "saved":
            uploaded_assets[title] = {
                **asset,
                "cloudinary_url": None,
                "cloudinary_public_id": None,
                "status": "upload_skipped_no_local_asset",
            }
            persist_asset_upload(
                category=category,
                canonical_title=title,
                cloudinary_url=None,
                cloudinary_public_id=None,
                status="UPLOAD_FAILED",
            )
            continue

        try:
            upload_kwargs: Dict[str, Any] = {
                "folder": CLOUDINARY_FOLDER,
            }
            if CLOUDINARY_UPLOAD_PRESET:
                upload_kwargs["upload_preset"] = CLOUDINARY_UPLOAD_PRESET

            response = cloudinary.uploader.upload(local_path, **upload_kwargs)
            cloudinary_url = response.get("secure_url")
            cloudinary_public_id = response.get("public_id")

            uploaded_assets[title] = {
                **asset,
                "source_links": source_links,
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
            short_error = summarize_cloudinary_error(exc)
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
    posted_assets: Dict[str, Any] = {}
    is_instagram_ready = init_instagram()
    merged_lookup = {item.canonical_title: item for item in state.get("merged_news", [])}
    posted_count = 0

    for title, asset in state.get("generated_assets", {}).items():
        category = str(asset.get("category", "")).strip()
        cloudinary_url = str(asset.get("cloudinary_url") or "").strip()

        if not is_instagram_ready:
            posted_assets[title] = {
                **asset,
                "instagram_status": "skipped_not_configured",
                "instagram_creation_id": None,
                "instagram_media_id": None,
                "instagram_permalink": None,
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=None,
                instagram_creation_id=None,
                instagram_media_id=None,
                instagram_permalink=None,
                status="INSTAGRAM_SKIPPED",
                error="instagram_not_configured",
            )
            continue

        if posted_count >= INSTAGRAM_MAX_POSTS:
            posted_assets[title] = {
                **asset,
                "instagram_status": "skipped_max_posts_reached",
                "instagram_creation_id": None,
                "instagram_media_id": None,
                "instagram_permalink": None,
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=None,
                instagram_creation_id=None,
                instagram_media_id=None,
                instagram_permalink=None,
                status="INSTAGRAM_SKIPPED",
                error="instagram_max_posts_reached",
            )
            continue

        if not cloudinary_url:
            posted_assets[title] = {
                **asset,
                "instagram_status": "skipped_no_cloudinary_url",
                "instagram_creation_id": None,
                "instagram_media_id": None,
                "instagram_permalink": None,
            }
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=None,
                instagram_creation_id=None,
                instagram_media_id=None,
                instagram_permalink=None,
                status="INSTAGRAM_SKIPPED",
                error="cloudinary_url_missing",
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

        try:
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
                print(f"  [instagram] Could not fetch permalink for media {media_id}: {summarize_instagram_error(details_exc)}")

            posted_count += 1
            posted_assets[title] = {
                **asset,
                "instagram_status": "posted",
                "instagram_creation_id": creation_id,
                "instagram_media_id": media_id,
                "instagram_permalink": permalink,
            }
            print(f"  📸 Instagram posted: {title} -> {media_id}")
            persist_asset_instagram(
                category=category,
                canonical_title=title,
                caption=caption,
                instagram_creation_id=str(creation_id),
                instagram_media_id=str(media_id),
                instagram_permalink=permalink,
                status="INSTAGRAM_POSTED",
                error=None,
            )
        except Exception as exc:
            short_error = summarize_instagram_error(exc)
            print(f"  📸 Instagram publish failed for '{title}': {short_error}")
            posted_assets[title] = {
                **asset,
                "instagram_status": f"failed: {short_error}",
                "instagram_creation_id": None,
                "instagram_media_id": None,
                "instagram_permalink": None,
            }
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

    return {"generated_assets": posted_assets}


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

g = StateGraph(State)
g.add_node('scraper', run_scraper)
g.add_node('merger', mergeNews)
g.add_node('store', storeImages)
g.add_node('cloudinary', uploadToCloudinary)
g.add_node('instagram', publishToInstagram)

g.add_edge(START, 'scraper')
g.add_edge('scraper', 'merger')
g.add_edge('merger', 'store')
g.add_edge('store', 'cloudinary')
g.add_edge('cloudinary', 'instagram')
g.add_edge('instagram', END)

app = g.compile()


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

def run_app():
    init_persistence()
    try:
        final_state = app.invoke({
            "merged_news": [],
            "raw_news_items": [],
            "generated_assets": {}
        })

        print("\n" + "="*50)
        print("MERGED NEWS RESULTS")
        print("="*50)

        if not final_state.get("merged_news"):
            print("No relevant news found.")
            return

        for i, news in enumerate(final_state["merged_news"], 1):
            print(f"\n[{i}] CATEGORY : {news.category}")
            print(f"    TITLE    : {news.canonical_title}")
            print(f"    SUMMARY  : {news.merged_content[:120]}...")
            print(f"    SOURCES  : {', '.join(news.source_links)}")

        print("\n" + "="*50)
        print("GENERATED ASSETS")
        print("="*50)

        for title, asset in final_state.get("generated_assets", {}).items():
            print(f"\n  [{asset['category']}] {title}")
            print(f"  Status : {asset['status']}")
            print(f"  Path   : {asset['local_path']}")
            if asset.get("cloudinary_url"):
                print(f"  URL    : {asset['cloudinary_url']}")
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
