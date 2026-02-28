from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Annotated
import operator
import json
import base64
import os
import re
from scraper import run

load_dotenv()


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
    model="gemini-2.5-flash",
    temperature=0.3
)

image_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image",
    temperature=0.3,
    google_api_key="AIzaSyDRCPu3r3sTKDcfh9kdlRBIolfkXJX0c2k"
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
Extract the 4–5 most critical points from the content below and display as bullets:

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
Numbered list (accent-color circle numbers), 3–4 steps extracted from the content.
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

    # Strip data URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    with open(file_path, "wb") as f:
        f.write(image_data)

    print(f"  ✓ Saved: {file_path}")
    return file_path


# ─────────────────────────────────────────────
# IMAGE GENERATION NODE
# ─────────────────────────────────────────────

def storeImages(state: State) -> dict:
    new_assets = {}

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

            # Use category + title slug as filename for uniqueness
            filename_key = f"{item.category}_{item.canonical_title}"
            path_on_disk = save_base64_as_image(base64_data, filename_key)

            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": path_on_disk,
                "sources": item.source_links,
                "status": "saved"
            }

        except Exception as e:
            print(f"  ✗ Failed for '{item.canonical_title}': {e}")
            new_assets[item.canonical_title] = {
                "category": item.category,
                "local_path": None,
                "status": f"error: {str(e)}"
            }

    return {"generated_assets": new_assets}


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

g = StateGraph(State)
g.add_node('scraper', run_scraper)
g.add_node('merger', mergeNews)
g.add_node('store', storeImages)

g.add_edge(START, 'scraper')
g.add_edge('scraper', 'merger')
g.add_edge('merger', 'store')
g.add_edge('store', END)

app = g.compile()


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

def run_app():
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


if __name__ == "__main__":
    run_app()