# AutoFlow - Current Pipeline Understanding and Next Steps

## 0) Latest Implementation Update (Done)

### MongoDB persistence layer added
- New file: `mongo_store.py`
- Collections now persisted by pipeline:
  - `raw_news`
  - `merged_news`
  - `assets`
- Upsert + dedup strategy:
  - `raw_news` dedup via `dedup_hash` (heading + source URL hash)
  - `merged_news` dedup via `merge_hash` (category + canonical title hash)
  - `assets` dedup via `asset_hash` (category + canonical title hash)
- Index bootstrap is automatic at runtime (`ensure_indexes`).

### `main.py` integration completed
- Removed hardcoded Gemini key usage.
- Added env-based settings for:
  - `GOOGLE_API_KEY`
  - `GEMINI_TEXT_MODEL`
  - `GEMINI_IMAGE_MODEL`
  - `MONGODB_URI`
  - `MONGODB_DB_NAME`
- Node-level persistence now happens automatically:
  - scraper node -> persists `raw_news`
  - merger node -> persists `merged_news`
  - image node -> persists `assets` and updates merged status
  - cloudinary node -> uploads generated assets and persists `cloudinary_url/public_id`

### Environment files added
- `.env` created (for local runtime values)
- `.env.example` created (shareable template)
- `requirements.txt` added with `pymongo` + core runtime dependencies
- Instagram + Cloudinary placeholders included so you can fill:
  - `INSTAGRAM_ACCESS_TOKEN`
  - `INSTAGRAM_IG_USER_ID`
  - `INSTAGRAM_PAGE_ID`
  - `INSTAGRAM_APP_ID`
  - `INSTAGRAM_APP_SECRET`
  - `INSTAGRAM_GRAPH_API_VERSION`
  - `CLOUDINARY_CLOUD_NAME`
  - `CLOUDINARY_API_KEY`
  - `CLOUDINARY_API_SECRET`
  - `CLOUDINARY_UPLOAD_PRESET`

## 1) Current codebase understanding

Current files:
- `main.py`: primary LangGraph pipeline (scrape -> merge -> generate image -> save local file)
- `main2.py`: older variant of same pipeline logic (partially duplicated)
- `scraper.py`: Playwright-based scraper using Google Trends + article page extraction

### Current pipeline flow (`main.py`)
1. `run_scraper` node
   - Calls `scraper.run()`.
   - Converts raw objects into `NewsItem` (`link`, `heading`, `content`).

2. `mergeNews` node
   - Uses `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) with structured output (`NewsList`).
   - Filters and merges relevant job/exam/education stories.
   - Produces `MergedNews` list with:
     - `category`
     - `canonical_title`
     - `merged_content`
     - `source_links`

3. `storeImages` node
   - Uses `gemini-2.5-flash-image` prompt to generate card image per merged news item.
   - Extracts returned image base64.
   - Saves PNG to `output_images/`.
   - Stores in-memory metadata in `generated_assets`.

4. Graph wiring
   - `START -> scraper -> merger -> store -> END`

### Current scraper behavior (`scraper.py`)
- Launches Chromium via Playwright (currently `headless=False`).
- Opens Google Trends India category page.
- Expands topic row(s) and collects non-Google external article links.
- Visits each article URL.
- Extracts:
  - first H1 as heading (fallback "No Heading")
  - long paragraph text blocks as content
- Returns list of objects:
  - `source`, `heading`, `content`

## 2) Current strengths
- End-to-end proof of concept is working.
- LangGraph orchestration is already established.
- Structured merge output is used before image generation.
- Category-based creative prompting is implemented.

## 3) Immediate technical gaps before Instagram automation
- Mongo persistence now exists for `raw_news`, `merged_news`, and `assets`, but publish queue persistence is pending.
- Cloudinary upload is now integrated and persisted in Mongo (`assets` + merged status).
- No publish queue / approval gate / post lifecycle states yet.
- `main.py` and `main2.py` duplicate logic and should be consolidated.
- Scraper reliability controls are still basic (no retry policy, no robust rate limiting beyond Playwright timing).

## 4) MongoDB target architecture (next)

### Collections to introduce
1. `raw_news`
   - `_id`
   - `source_url`
   - `heading`
   - `content`
   - `scraped_at`
   - `dedup_hash`

2. `merged_news`
   - `_id`
   - `category`
   - `canonical_title`
   - `merged_content`
   - `source_links`
   - `created_at`
   - `status` (`MERGED`, `IMAGE_GENERATED`, `UPLOADED`, `READY_FOR_POST`, `POSTED`, `FAILED`)

3. `assets`
   - `_id`
   - `merged_news_id`
   - `local_path`
   - `cloudinary_url`
   - `cloudinary_public_id`
   - `created_at`

4. `publish_jobs`
   - `_id`
   - `merged_news_id`
   - `asset_id`
   - `caption`
   - `scheduled_for`
   - `approval_status` (`UNDER_REVIEW`, `APPROVED`, `REJECTED`)
   - `publish_status` (`PENDING`, `POSTED`, `FAILED`)
   - `instagram_media_id`
   - `error`

### Required indexes
- `raw_news.dedup_hash` unique
- `merged_news.status`
- `publish_jobs.approval_status`
- `publish_jobs.publish_status`
- `publish_jobs.scheduled_for`

## 5) Cloudinary integration (implemented)
- Added dedicated LangGraph node after image generation:
  - `uploadToCloudinary`
- Input: local image path from `generated_assets`
- Output persisted in Mongo:
  - `cloudinary_url`
  - `cloudinary_public_id`
- Merged status is updated to:
  - `UPLOADED` on success
  - `UPLOAD_FAILED` on failure

## 6) Instagram automation integration step
- Add publish workflow node(s):
  - `caption_node` (LLM caption + hashtags)
  - `approval_gate_node` (manual or API-controlled)
  - `instagram_publish_node`
- Instagram publish via Meta Graph API flow:
  1. Create media container (`image_url` + `caption`)
  2. Publish container
  3. Persist returned post IDs + timestamps
- Retry policy and dead-letter behavior for failed publishes.

## 7) Recommended execution plan (in order)
1. Consolidate to single pipeline file (`main.py`) and archive/remove `main2.py`.
2. Move hardcoded secrets to `.env`; remove key from source code.
3. Add Mongo client + migration/index setup.
4. Persist `raw_news` during scrape node with dedup.
5. Persist merged results in `merged_news`.
6. Persist generated image metadata in `assets`.
7. Add Cloudinary upload node and persistence.
8. Add publish job collection + approval state machine.
9. Integrate Instagram posting node.
10. Add structured logging + run IDs + failure tracking.

## 8) Definition of completion for next milestone
Milestone is complete when one full run does this:
- Scrape -> Merge -> Generate Image -> Upload to Cloudinary -> Create publish job in Mongo (`UNDER_REVIEW`).
- Nothing gets posted until explicit approval.
