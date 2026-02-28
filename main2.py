from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.types import Send
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any,Annotated
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
import operator
import time
import random
import json
import base64
import os
import re
from scraper import run

load_dotenv()


# ---------- STATE MODELS ----------

class NewsItem(BaseModel):
    link: str
    heading: str
    content: str

class MergedNews(BaseModel):
    topic_name: str
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
    # input
    # scraped_batches: List[Dict[str, Any]]   # your JSON

    # flattened
    raw_news_items: List[NewsItem]

    # DB layer
    # fresh_news_items: List[NewsItem]
    # discarded_links: List[str]

    # merged per topic
    merged_news: List[MergedNews]

    # generation
    generated_assets: Annotated[Dict[str, Any], merge_dicts]

    # errors: List[str]



# ---------- LANGGRAPH NODE ----------

def run_scraper(state: State) -> dict:
   data=run()
   rawNews:List[NewsItem]=[]
   for item in data:
       news=NewsItem(
       link=item['source'],    
       heading=item['heading'],    
       content=item['content']    
       )
       rawNews.append(news)
       

   return {"raw_news_items":rawNews}

llm=ChatGoogleGenerativeAI(
     model="gemini-2.5-flash",
     temperature=0.3)   

MERGE_PROMPT="""
### Role
You are a Professional News Synthesis Editor specializing in the Indian education and employment sector. Your goal is to transform raw scraped news data into high-quality, deduplicated news summaries.

### Task
Analyze the provided list of raw news items and perform the following steps:

1. **Strict Filtering**: Retain ONLY news related to:
   - Job Openings / Recruitment / Careers
   - Education / Admissions / Results
   - Competitive Exams / Notifications
   - Discard all other topics (e.g., general politics, sports, entertainment) immediately.

2. **Deduplication & Clustering**: 
   - Identify news items that describe the same event (e.g., multiple sources reporting on the same UPSC result or a specific Bank recruitment). 
   - Group these items together to create a single 'MergedNews' object for that event.

3. **Information Synthesis**:
   - **Canonical Title**: Create a clear, click-worthy, yet factual title for the merged story.
   - **Merged Content**: Write a concise summary (100-150 words). Do NOT lose critical details such as deadlines, eligibility criteria, exam dates, or official website names. Combine unique facts from all sources in the cluster.
   - **Topic Name**: Categorize into "Jobs", "Education", or "Exams".
   - **Source Links**: Collect all unique URLs from the sources used to create that specific merged object.

### Quality Constraints
- Use a professional, journalistic tone.
- Avoid repetitive sentences.
- If a news item is incomplete or vague, discard it.
- Ensure all source links are valid strings."""


def mergeNews(state: State) -> dict:
    # Use the container class instead of List[MergedNews]
    merger = llm.with_structured_output(NewsList)
    
    # It's better to pass the data as a clean string
    raw_data_string = json.dumps([item.model_dump() for item in state['raw_news_items']])
    
    result = merger.invoke(
        [
            SystemMessage(content=MERGE_PROMPT),
            HumanMessage(content=f"Raw Data: {raw_data_string}"),
        ]
    )
    
    # Return the 'news' list from the container object
    return {'merged_news': result.news}

IMAGE_PROMPT = """
Create a social media news card (1080x1080px) styled like an educational infographic post.

STEP 1 — DETECT CATEGORY:
Read the title and content carefully and classify it into EXACTLY ONE of these categories:
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

STEP 2 — SET VISUAL THEME based on detected category:
  RESULT DECLARED   → Background: deep green  (#1a472a → #2d6a4f)  | Accent: mint (#52b788)
  ADMIT CARD        → Background: deep navy   (#0d1b3e → #1a3a6b)  | Accent: sky blue (#74b9e8)
  EXAM SCHEDULE     → Background: dark teal   (#0d3b38 → #1a5c57)  | Accent: aqua (#2ec4b6)
  IMPORTANT NOTICE  → Background: deep indigo (#1a1a4e → #2d2d7a)  | Accent: lavender (#9b8fe8)
  JOB OPENING       → Background: dark maroon (#3b0d0d → #6b1a1a)  | Accent: gold (#e9c46a)
  ANSWER KEY        → Background: dark slate  (#1a2535 → #2d3f5a)  | Accent: cyan (#48cae4)
  MERIT LIST        → Background: deep purple (#2d0d3b → #4a1a6b)  | Accent: violet (#c77dff)
  SCHOLARSHIP       → Background: dark olive  (#2b3a0d → #4a6320)  | Accent: lime (#a7c957)
  ADMISSION OPEN    → Background: deep ocean  (#0d2b3b → #1a4f6b)  | Accent: coral (#ff6b6b)
  DEADLINE ALERT    → Background: dark amber  (#3b2000 → #6b3a00)  | Accent: orange (#f4a261)

STEP 3 — BUILD THE CARD:

TOP SECTION:
- Rounded pill badge top-left using detected category emoji + name (e.g. "🎓 RESULT DECLARED"), 
  in accent color, uppercase, small bold text, semi-transparent white border
- Main headline in large bold white text (2 lines max): "{title}"
- Thin horizontal white divider (15% opacity) below headline

MIDDLE SECTION — KEY HIGHLIGHTS:
Label: "KEY HIGHLIGHTS" in tiny accent-color uppercase.
Read the content below and extract the 4–5 most critical points for a student:

CONTENT: {content}

Assign each bullet a colored circle icon based on what the point conveys:
  ✓  Accent color circle  → confirmed facts, key announcements
  🌐 Sky blue circle      → website, portal, where to access
  ⚠  Yellow circle        → warnings, delays, conditions, cautions  
  ✕  Red circle           → restrictions, not available, rejected
  📅 White circle         → important dates, deadlines, schedules
  ↩  Mint circle          → next steps, what to do after

Rules for bullets:
- Rewrite each point in 10 words or fewer — never copy paste from content
- ONE LINE per bullet — absolutely no text wrapping
- Bold the single most important word in each bullet

BOTTOM BOX — ACTION STEPS:
Dark semi-transparent rounded box (black 20% opacity).
Label: "📋 WHAT YOU SHOULD DO NOW" in small accent-color uppercase.
Extract 3–4 clear action steps from the content, numbered with accent-color circle numbers.
Each step: max 10 words, student-friendly, starts with a verb (Check / Visit / Download / Apply / Submit).

Below box: accent-color outlined pill → most relevant official URL from {sources}

FOOTER:
Thin white divider (10% opacity).
Left: small italic muted text → domain names only from {sources} (e.g. "ndtv.com · timesofindia.com")
Right: small uppercase → "EDUCATION UPDATE"

STRICT DESIGN RULES:
— NO illustrations, characters, stock images, or decorative graphics anywhere
— Every word must be perfectly spelled — zero spelling errors allowed
— No text overlapping any other element — strict vertical stacking only
— Font: clean geometric sans-serif (Poppins or equivalent)
— Use ONLY the background and accent colors assigned to the detected category
— Secondary colors allowed: white (text), yellow (#e9c46a for warnings), red (#e76f51 for alerts)
— Minimum 28px padding on all sides
— Card must feel information-rich but never cluttered
— Flat solid color on all text — zero gradients on text
— Overall mood: trustworthy, urgent, student-friendly, mobile-optimized
"""

template = PromptTemplate(
    template=IMAGE_PROMPT,
    input_variables=['title', 'content', 'sources']
)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image", 
    google_api_key="AIzaSyDRCPu3r3sTKDcfh9kdlRBIolfkXJX0c2k"
)
# response = model.invoke(IMAGE_PROMPT)

def _get_image_base64(response: AIMessage) -> None:
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    return image_block["image_url"].get("url").split(",")[-1]


# image_base64 = _get_image_base64(response)
# display(Image(data=base64.b64decode(image_base64), width=300))

def save_base64_as_image(base64_str: str, title: str, folder: str = "output_images") -> str:
    """Decodes base64 and saves to a local folder. Returns the file path."""
    # 1. Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 2. Clean the title to make it a safe filename
    safe_name = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
    file_path = os.path.join(folder, f"{safe_name}.png")

    # 3. Strip metadata if present (e.g., 'data:image/png;base64,')
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # 4. Decode and Write
    image_data = base64.b64decode(base64_str)
    with open(file_path, "wb") as f:
        f.write(image_data)

    return file_path

def storeImages(state: State) -> dict:
    new_assets = {}
    
    for item in state['merged_news']:
        # 1. Generate the image content
        prompt = template.invoke({
            'title': item.canonical_title,
            'content': item.merged_content,
            'sources':item.source_links,
            'topic':item.topic_name
        })
        response = model.invoke(prompt)
        
        # 2. Extract Base64 from your model response
        base64_data = _get_image_base64(response)
        
        # 3. Save to local disk
        path_on_disk = save_base64_as_image(base64_data, item.topic_name)
        
        # 4. Update the dictionary with the path
        new_assets[item.topic_name] = {
            "local_path": path_on_disk,
            "status": "saved"
        }

    # This returns to the LangGraph reducer we set up earlier
    return {"generated_assets": new_assets}


g=StateGraph(State)
g.add_node('scraper',run_scraper)
g.add_node('merger',mergeNews)
g.add_node('store',storeImages)

g.add_edge(START,'scraper')
g.add_edge('scraper','merger')
g.add_edge('merger','store')
g.add_edge('store',END)
app=g.compile()

def run_app(): # Renamed to avoid conflict with the 'run' import from scraper
    final_state = app.invoke(
        {
            "merged_news": [],
            "raw_news_items": []
        }
    )
    
    print("\n--- MERGED NEWS RESULTS ---\n")
    
    # Check if any news was merged
    if not final_state.get("merged_news"):
        print("No relevant news found for Jobs, Education, or Exams.")
        return

    for i, news in enumerate(final_state["merged_news"], 1):
        print(f"[{i}] TOPIC: {news.topic_name}")
        print(f"TITLE: {news.canonical_title}")
        print(f"SUMMARY: {news.merged_content}")
        print(f"SOURCES: {', '.join(news.source_links)}")
        print("-" * 30)

if __name__ == "__main__":
    run_app()
