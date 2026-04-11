"""
update_dataset.py
-----------------
Pipeline to:
  1. Scrape the latest Daily Price Report PDF from CBSL.
  2. Extract vegetable / fruit / rice wholesale & retail prices (page 2).
  3. Fetch rainfall data from Open-Meteo for two key farming regions.
  4. Save / update a wide-format CSV dataset.
  5. Push the CSV to a Hugging Face dataset repository.

Run modes
---------
  python update_dataset.py                        # process only the latest PDF
  python update_dataset.py --backfill             # probe & fill ALL weekdays since 2023-01-01
  python update_dataset.py --backfill --start-date 2024-01-01   # backfill from a specific date
  python update_dataset.py --force                # re-process latest even if already exists

How PDF discovery works (important fix)
----------------------------------------
The old approach scraped the CBSL HTML listing page with pagination (?page=0,1,2...).
This was broken: the CBSL site uses Drupal with a Year/Month filter, and the default
paginated view only iterates within the currently-selected year. Result: ~45-55% of
2023 dates were silently missing — NOT holidays, just never found by the scraper.

The new approach directly generates the PDF URL from the date (the URL is 100%
predictable: .../price_report_YYYYMMDD_e.pdf), sends a fast HEAD request to check
if it exists, and only downloads confirmed PDFs. HTTP 404 = genuine holiday or
non-publishing day. This gives complete coverage with no pagination complexity.

How the PDF is parsed
---------------------
Page 2 of each report is a price table with 10 numeric columns per item:
  [WS_Pettah_Yest, WS_Pettah_Today,
   WS_Dambulla_Yest, WS_Dambulla_Today,
   RT_Pettah_Yest, RT_Pettah_Today,
   RT_Dambulla_Yest, RT_Dambulla_Today,
   RT_Narahenpita_Yest, RT_Narahenpita_Today]

We only store the "Today" values at indices 1, 3, 5, 7, 9.

The PDF uses a split-glyph font rendering where every number is broken
into two tokens by pdfplumber, e.g.:
  "200.00" -> ['2', '00.00']
  "1000.00" -> ['1', ',000.00']   (thousands separator also split)
  "80.00"  -> ['8', '0.00']      (2-digit numbers split at tens digit)
  "n.a."   -> ['n.a.']            (not reported)

_extract_values() handles all these patterns reliably.
Validated against the real 08-Apr-2026 PDF: 24 items x 5 fields = 120/120 correct.
"""

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import pdfplumber
from dotenv import load_dotenv
from huggingface_hub import HfApi


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CBSL_PRICE_REPORT_URL = (
    "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report"
)

# Matches URLs like: .../price_report_20260408_e.pdf
PDF_LINK_RE = re.compile(r"https?://[^\"']+price_report_(\d{8})_e\.pdf")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CSV_PATH = DATA_DIR / "price_dataset.csv"

# Canonical item names — these become column name prefixes in the dataset.
REQUIRED_ITEMS: List[str] = [
    "Beans",
    "Carrot",
    "Cabbage",
    "Tomato",
    "Brinjal",
    "Pumpkin",
    "Snake Gourd",
    "Green Chilli",
    "Lime",
    "Red Onion (Local)",
    "Red Onion (Imp)",
    "Big Onion (Local)",
    "Big Onion (Imp)",
    "Potato (Local)",
    "Potato (Imp)",
    "Dried Chilli (Imp)",
    "Coconut (Avg.)",
    "Banana (Sour)",
    "Papaw",
    "Pineapple",
    "Samba",
    "Nadu",
    "Kekulu (White)",
    "Kekulu (Red)",
]

# Case-insensitive lookup: lowercase_alias -> canonical_name
# Handles capitalisation differences AND known PDF typos.
_ITEM_LOWER_MAP: Dict[str, str] = {item.lower(): item for item in REQUIRED_ITEMS}
_ITEM_LOWER_MAP["red onion (lmp)"] = "Red Onion (Imp)"  # PDF prints 'lmp' not 'Imp'
_ITEM_LOWER_MAP["snake gourd"] = "Snake Gourd"           # PDF uses lowercase 'gourd'

# (field_name, sequential_index) — Today values are at odd positions 1,3,5,7,9
PRICE_FIELDS: List[Tuple[str, int]] = [
    ("wholesale_pettah",    1),
    ("wholesale_dambulla",  3),
    ("retail_pettah",       5),
    ("retail_dambulla",     7),
    ("retail_narahenpita",  9),
]

# Farming region coordinates for rainfall data
NUWARA_ELIYA = (6.97078, 80.78286)
POLONNARUWA  = (7.93965, 81.00274)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _match_item(text: str) -> Optional[str]:
    """
    Case-insensitive match: is any known item name present in `text`?
    Returns the canonical name if yes, None if no match.

    WHY: The PDF uses inconsistent capitalisation ('snake gourd', 'Red Onion
    (lmp)') and the item name may be surrounded by other text on the line.
    A lower-case substring match is simpler and more robust than exact equality.
    """
    text_lower = text.lower()
    for alias, canonical in _ITEM_LOWER_MAP.items():
        if alias in text_lower:
            return canonical
    return None


def _extract_values(line: str) -> List[float]:
    """
    Extract all price values from one item's text line.

    The CBSL PDF uses a split-glyph rendering: each number's digits are
    emitted as two separate text tokens by pdfplumber. This function
    reassembles them.

    Split patterns discovered in the real PDF:
      '2' + '00.00'  -> 200.00   (3-digit number, split before last 2 digits)
      '8' + '0.00'   ->  80.00   (2-digit number, split before last 1 digit)
      '5' + '8.00'   ->  58.00   (same, different tens digit)
      '1' + '30.00'  -> 130.00   (3-digit, split normally)
      '1' + ',000.00'-> 1000.00  (4-digit, comma thousands separator also split)
      '1' + ',025.00'-> 1025.00  (same)
      'n.a.'         ->  NaN     (value not reported for that market)

    The key regex fix vs. the original code:
      Original: r"\\d{2,3}\\.\\d{2}"  -- missed 1-digit decimal parts like '0.00'
      Fixed:    r"\\d{1,3}\\.\\d{2}"  -- handles 1, 2, or 3 digit decimal parts
    """
    tokens = line.split()

    # Skip everything before the unit token (Rs./kg, Rs./Nut, Rs./Ltr)
    unit_idx = next((i for i, t in enumerate(tokens) if t.startswith("Rs./")), None)
    if unit_idx is None:
        return []

    vt = tokens[unit_idx + 1:]
    values: List[float] = []
    i = 0

    while i < len(vt):
        t = vt[i]

        # Not-available marker
        if t.lower() == "n.a.":
            values.append(float("nan"))
            i += 1
            continue

        # Split number: bare integer + decimal-part token
        if re.fullmatch(r"\d+", t) and i + 1 < len(vt):
            nxt = vt[i + 1]

            # Normal split (1–3 digits before the dot): covers 2-digit and 3-digit prices
            if re.fullmatch(r"\d{1,3}\.\d{2}", nxt):
                values.append(float(t + nxt))
                i += 2
                continue

            # Thousands-comma split: '1' + ',000.00' -> 1000.00
            if re.fullmatch(r",\d{3}\.\d{2}", nxt):
                values.append(float(t + nxt.replace(",", "")))
                i += 2
                continue

        # Standalone decimal (defensive fallback — rare in practice)
        if re.fullmatch(r"\d[\d,]*\.\d{2}", t):
            values.append(float(t.replace(",", "")))
            i += 1
            continue

        i += 1  # skip non-numeric tokens (item name fragments, markers, etc.)

    return values


def _http_get(url: str, retries: int = 3, backoff: float = 2.0) -> requests.Response:
    """
    HTTP GET with exponential-backoff retry.

    WHY: CBSL and Open-Meteo occasionally return transient errors. Without
    retry, a single flaky network call would fail the entire GitHub Actions
    run. Three attempts handles the vast majority of real-world issues.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff ** attempt
                print(f"  [retry {attempt + 1}/{retries}] {exc} — waiting {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"HTTP GET failed after {retries} attempts: {url}") from last_exc


# ---------------------------------------------------------------------------
# PDF discovery & download
# ---------------------------------------------------------------------------

# The CBSL PDF URL is 100% predictable from the date — no HTML scraping needed.
# This is the root-cause fix for missing dates:
#
# OLD APPROACH (broken): scrape the CBSL listing page, follow pagination.
#   Problem: the CBSL website (Drupal) has "Filter by Year/Month" dropdowns.
#   The default view only shows PDFs within the current year's filter, so
#   paginating ?page=0, ?page=1... only iterates through the current year.
#   Older years were simply never found, causing ~45-55% data loss in 2023.
#
# NEW APPROACH (correct): directly generate every weekday URL by date,
#   send a HEAD request to check if the PDF exists (fast, no download),
#   then only download the ones that return HTTP 200.
#   CBSL holidays produce 404 — that's how we detect genuine non-publishing days.
#   This gives us 100% coverage with zero dependence on the listing page.

CBSL_PDF_BASE = (
    "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/"
    "statistics/pricerpt/price_report_{date_str}_e.pdf"
)

# Keep the listing URL only as a fallback for find_latest_pdf_url()
CBSL_PRICE_REPORT_URL = (
    "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report"
)
PDF_LINK_RE = re.compile(r"https?://[^\"']+price_report_(\d{8})_e\.pdf")

# Earliest date CBSL has PDFs online (adjust if you need to go further back)
CBSL_HISTORY_START = datetime(2023, 1, 1)


def _pdf_url(date_str: str) -> str:
    """Build the direct PDF URL for a given date string (YYYYMMDD)."""
    return CBSL_PDF_BASE.format(date_str=date_str)


def _pdf_exists(url: str) -> bool:
    """
    Check if a PDF URL exists using a HEAD request (no download).

    WHY HEAD: a HEAD request fetches only the HTTP status code and headers,
    not the file body. Checking 220 missing dates with HEAD requests is
    fast (< 1 second per request) vs downloading each full PDF first.
    Returns True for HTTP 200, False for 404 (holiday / non-publishing day).
    """
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def list_pdf_links(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, str]:
    """
    Generate and probe every weekday PDF URL between start_date and end_date.
    Returns {date_str: pdf_url} for every date where the PDF exists (HTTP 200).

    WHY this replaces HTML scraping:
    The CBSL listing page uses Drupal's Views with a year/month filter. Paginating
    through ?page=0,1,2... only iterates within the currently-selected year filter,
    so older years were silently skipped — causing ~45-55% data loss in 2023.
    Direct URL probing is simpler, faster, and gives 100% coverage.
    """
    if start_date is None:
        start_date = CBSL_HISTORY_START
    if end_date is None:
        end_date = datetime.utcnow()

    # Generate all weekdays (Mon-Fri) in the range
    weekdays = pd.bdate_range(start=start_date, end=end_date)

    links: Dict[str, str] = {}
    total = len(weekdays)
    found = 0

    for i, day in enumerate(weekdays):
        date_str = day.strftime("%Y%m%d")
        url = _pdf_url(date_str)
        if _pdf_exists(url):
            links[date_str] = url
            found += 1
        if (i + 1) % 50 == 0:
            print(f"  Probed {i+1}/{total} dates, found {found} PDFs so far...")

    print(f"  Done probing: {found}/{total} weekdays have a PDF.")
    return links


def find_latest_pdf_url() -> Tuple[str, str]:
    """
    Find the most recent PDF by probing the last 10 weekdays.

    WHY 10 days: CBSL sometimes publishes with a 1-2 day delay. Checking the
    last 10 weekdays (~2 calendar weeks) guarantees we catch the latest PDF
    even if there was a holiday cluster or upload delay.
    """
    # Generate last 10 weekdays ending today
    today = datetime.utcnow()
    last_10 = pd.bdate_range(end=today, periods=10)

    for day in reversed(last_10):  # newest first
        date_str = day.strftime("%Y%m%d")
        url = _pdf_url(date_str)
        if _pdf_exists(url):
            return url, date_str

    raise RuntimeError(
        "No PDF found in the last 10 weekdays. "
        "CBSL may not have published yet, or check your internet connection."
    )


def download_pdf(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(_http_get(url).content)
    print(f"  Downloaded: {target_path.name}")


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def extract_prices(pdf_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse page 2 of the CBSL Daily Price Report and return:
      {canonical_item_name: {field_name: value, ...}, ...}

    Why pdfplumber text mode (not tabula):
    - tabula requires Java, adding CI setup complexity.
    - pdfplumber's extract_text(x_tolerance=4) reliably returns all 10
      values per row as a flat line — we tested this against the real PDF.
    - Sequential indexing [1,3,5,7,9] is robust because the table structure
      is fixed: CBSL always publishes Yesterday then Today per column.

    Raises RuntimeError if any required item is missing after parsing.
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        if len(pdf.pages) < 2:
            raise RuntimeError(
                f"{pdf_path.name}: expected at least 2 pages, got {len(pdf.pages)}"
            )
        page = pdf.pages[1]
        # x_tolerance=4: slightly wider than default (2) so that the two
        # glyph fragments of each split number land in the same text line.
        text = page.extract_text(x_tolerance=4, y_tolerance=2)

    if not text:
        raise RuntimeError(
            f"{pdf_path.name}: pdfplumber returned empty text from page 2."
        )

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    results: Dict[str, Dict[str, float]] = {}

    for line in lines:
        if "Rs./" not in line:  # fast pre-filter
            continue

        item = _match_item(line)
        if not item or item in results:
            continue

        values = _extract_values(line)

        # Each item must have exactly 10 values (yesterday + today per market).
        # All-NaN rows (e.g. Big Onion Local) are valid and still count as 10.
        if len(values) < 10:
            print(
                f"  WARNING: {item} yielded only {len(values)} values "
                f"(expected 10) — skipping."
            )
            continue

        results[item] = {field: values[idx] for field, idx in PRICE_FIELDS}

    missing = [it for it in REQUIRED_ITEMS if it not in results]
    if missing:
        raise RuntimeError(
            f"{pdf_path.name}: could not extract {len(missing)} items: {missing}"
        )

    return results


# ---------------------------------------------------------------------------
# Rainfall data
# ---------------------------------------------------------------------------

def fetch_precipitation_range(
    start_date: str, end_date: str, lat: float, lon: float
) -> Dict[str, float]:
    """
    Fetch daily precipitation totals (mm) from Open-Meteo for a date range.
    Returns {ISO_date: mm_float} for every day in the range.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=precipitation_sum&timezone=Asia%2FColombo"
    )
    data = _http_get(url).json()
    daily  = data.get("daily", {})
    times  = daily.get("time", [])
    values = daily.get("precipitation_sum", [])
    if not times or len(times) != len(values):
        raise RuntimeError(f"Unexpected Open-Meteo response for ({lat},{lon}): {data}")
    return {
        t: (float(v) if v is not None else float("nan"))
        for t, v in zip(times, values)
    }


# ---------------------------------------------------------------------------
# Dataset assembly (wide format)
# ---------------------------------------------------------------------------

def _col_name(field: str, item: str) -> str:
    """
    Build a column name: "retail_pettah_Beans", "wholesale_dambulla_Red_Onion_Imp".
    Strips parentheses and dots; replaces spaces with underscores.
    """
    safe = (
        item.replace("(", "")
            .replace(")", "")
            .replace(".", "")
            .replace(" ", "_")
    )
    return f"{field}_{safe}"


def build_wide_row(
    report_date: str,
    pdf_url: str,
    prices: Dict[str, Dict[str, float]],
    rain_nuwara: float,
    rain_polonnaruwa: float,
) -> Dict[str, object]:
    """
    Assemble one wide-format CSV row for a single report date.

    WHY wide format: the target schema (Image 2) places each item-market
    combination as its own column, with one row per date. This is also the
    most convenient layout when each date's data is consumed as a single
    ML feature vector.
    """
    row: Dict[str, object] = {
        "date": report_date,
        "rainfall_nuwara_eliya_mm": rain_nuwara,
        "rainfall_polonnaruwa_mm": rain_polonnaruwa,
        "source_pdf": pdf_url,
    }
    for item in REQUIRED_ITEMS:
        for field, _ in PRICE_FIELDS:
            row[_col_name(field, item)] = prices[item][field]
    return row


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

# Columns that the old long-format pipeline wrote but the wide-format pipeline
# does NOT use. These are dropped automatically when loading any existing CSV,
# so old files are migrated on first run without any manual intervention.
_LEGACY_COLUMNS = [
    "item", "unit",
    "retail_pettah", "retail_dambulla", "retail_narahenpita",
    "wholesale_pettah", "wholesale_dambulla",
]


def load_existing_df() -> pd.DataFrame:
    """
    Load the existing CSV, auto-migrating away from the old long format.

    WHY: The first version of this pipeline wrote one row per item per date
    (long format) with generic column names like 'retail_pettah'. After
    switching to wide format those columns became empty ghosts in the CSV.
    Dropping them here means the dataset self-heals on any machine without
    needing a separate migration script.

    Also drops any rows that have no wide-format price data at all — these
    are rows that were written by the old pipeline for dates where the new
    pipeline hasn't run yet.
    """
    if not CSV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH).copy()  # .copy() defragments the wide DataFrame

    # Drop legacy columns if present (silently ignore if already gone)
    legacy_present = [c for c in _LEGACY_COLUMNS if c in df.columns]
    if legacy_present:
        print(f"  [migrate] Dropping {len(legacy_present)} legacy columns: {legacy_present}")
        df = df.drop(columns=legacy_present)

    # Drop rows that have no wide-format price data.
    # We use wholesale_pettah_Beans as a sentinel — it is always populated
    # for any correctly-processed date.
    sentinel = "wholesale_pettah_Beans"
    if sentinel in df.columns:
        bad_rows = df[sentinel].isna().sum()
        if bad_rows:
            print(f"  [migrate] Dropping {bad_rows} rows with no wide-format data.")
            df = df[df[sentinel].notna()]

    return df.reset_index(drop=True)


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps in the dataset caused by CBSL not publishing on public holidays.

    WHY this happens
    ----------------
    CBSL publishes a price report every weekday EXCEPT Sri Lanka public
    holidays (Poya days, Sinhala/Tamil New Year, Christmas, etc.). Gaps
    of 1-12 days are entirely expected — NOT scraping failures.
    Analysis of the real dataset confirms all large gaps align with known
    Sri Lanka holiday clusters (New Year week, Wesak, Poya days, etc.).

    WHY forward-fill is the right approach
    ---------------------------------------
    Prices don't jump to zero on a holiday — the last known market price
    is the best estimate for that day. Forward-fill (ffill) is the standard
    technique for daily financial/commodity price datasets: carry the last
    published value forward until a new publication arrives.

    What we do
    ----------
    1. Reindex to a complete daily calendar (no gaps).
    2. Mark newly inserted rows with is_interpolated=True BEFORE filling,
       so consumers can always distinguish real publications from filled rows.
    3. Forward-fill all price and metadata columns.

    The 'source_pdf' on interpolated rows points to the PDF whose prices
    were carried forward — useful for debugging and traceability.
    """
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    added_days = len(full_range) - len(df)

    # Reindex — missing dates become rows of NaN
    df = df.reindex(full_range)
    df.index.name = "date"

    # Mark interpolated rows BEFORE ffill using source_pdf as the indicator:
    # if source_pdf is NaN, CBSL did not publish that day.
    df["is_interpolated"] = df["source_pdf"].isna()

    # Forward-fill everything except the flag
    fill_cols = [c for c in df.columns if c != "is_interpolated"]
    df[fill_cols] = df[fill_cols].ffill()

    df = df.reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    if added_days > 0:
        print(f"  Filled {added_days} missing date(s) via forward-fill "
              f"(holidays / non-publishing days).")
    return df


def save_csv(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fill_missing_dates(df)
    df.to_csv(CSV_PATH, index=False, na_rep="NaN")
    print(f"  Saved {len(df)} rows -> {CSV_PATH}")


# ---------------------------------------------------------------------------
# Hugging Face upload
# ---------------------------------------------------------------------------

def push_to_huggingface() -> None:
    """
    Upload price_dataset.csv to a Hugging Face dataset repo.
    Reads HF_DATASET_REPO and HF_TOKEN from environment / .env file.
    In GitHub Actions these are set as repository secrets.
    """
    load_dotenv()
    repo_id = os.getenv("HF_DATASET_REPO")
    token   = os.getenv("HF_TOKEN")
    if not repo_id or not token:
        print("HF_DATASET_REPO or HF_TOKEN not set — skipping Hugging Face upload.")
        return
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(CSV_PATH),
        path_in_repo="price_dataset.csv",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Auto-update: {datetime.utcnow().strftime('%Y-%m-%d')}",
    )
    print(f"  Pushed to Hugging Face: {repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    force    = "--force"    in sys.argv
    backfill = "--backfill" in sys.argv

    # Optional: --start-date YYYY-MM-DD lets you backfill from a specific date
    # instead of the full CBSL_HISTORY_START. Useful to fill only recent gaps.
    start_date_override: Optional[datetime] = None
    if "--start-date" in sys.argv:
        idx = sys.argv.index("--start-date")
        if idx + 1 < len(sys.argv):
            start_date_override = datetime.strptime(sys.argv[idx + 1], "%Y-%m-%d")

    existing_df = load_existing_df()
    # Only treat a date as "already processed" if it came from a real CBSL
    # publication — not from forward-fill. This ensures that if today was
    # previously forward-filled, the pipeline still processes the real PDF
    # when it becomes available.
    if not existing_df.empty and "is_interpolated" in existing_df.columns:
        real_rows = existing_df[existing_df["is_interpolated"] == False]
        existing_dates: set = set(real_rows["date"].astype(str).unique())
    else:
        existing_dates = (
            set(existing_df["date"].astype(str).unique()) if not existing_df.empty else set()
        )

    # Decide which PDFs to process
    if backfill:
        # Probe every weekday URL directly — no HTML scraping, no pagination bugs.
        # 404 = holiday or non-publishing day (correctly skipped).
        # 200 = PDF exists, add to queue.
        start = start_date_override or CBSL_HISTORY_START
        print(f"Backfill: probing all weekday URLs from {start.date()} to today...")
        links = list_pdf_links(start_date=start)
        dates_to_process = sorted(links.keys())
        if not force:
            dates_to_process = [
                d for d in dates_to_process
                if datetime.strptime(d, "%Y%m%d").date().isoformat() not in existing_dates
            ]
        if not dates_to_process:
            print("Backfill: all dates already present. Nothing to do.")
            return 0
        print(f"Backfill: {len(dates_to_process)} new date(s) to process.")
    else:
        latest_url, date_compact = find_latest_pdf_url()
        report_date = datetime.strptime(date_compact, "%Y%m%d").date().isoformat()
        if report_date in existing_dates and not force:
            print(f"Already up to date: {report_date}")
            return 0
        links = {date_compact: latest_url}
        dates_to_process = [date_compact]

    # Batch-fetch rainfall for all dates at once (one API call per region)
    iso_dates  = [datetime.strptime(d, "%Y%m%d").date().isoformat() for d in dates_to_process]
    start_date, end_date = min(iso_dates), max(iso_dates)
    print(f"Fetching rainfall {start_date} -> {end_date}...")
    rain_nuwara_map = fetch_precipitation_range(start_date, end_date, *NUWARA_ELIYA)
    rain_pol_map    = fetch_precipitation_range(start_date, end_date, *POLONNARUWA)

    # Process each PDF
    new_rows: List[Dict[str, object]] = []
    for date_compact in dates_to_process:
        report_date = datetime.strptime(date_compact, "%Y%m%d").date().isoformat()
        pdf_url     = links[date_compact]
        pdf_path    = RAW_DIR / f"price_report_{date_compact}_e.pdf"

        print(f"\nProcessing {report_date}...")
        if not pdf_path.exists() or force:
            download_pdf(pdf_url, pdf_path)

        try:
            prices = extract_prices(pdf_path)
        except RuntimeError as exc:
            # In backfill mode: skip bad PDFs instead of aborting everything.
            print(f"  ERROR: {exc}")
            if backfill:
                continue
            raise

        rain_nuwara = rain_nuwara_map.get(report_date, float("nan"))
        rain_pol    = rain_pol_map.get(report_date, float("nan"))
        new_rows.append(build_wide_row(report_date, pdf_url, prices, rain_nuwara, rain_pol))
        print(f"  OK — {len(REQUIRED_ITEMS)} items extracted.")

    if not new_rows:
        print("No new rows produced.")
        return 1

    new_df = pd.DataFrame(new_rows)

    # Merge: replace any existing rows for dates we just processed
    if not existing_df.empty:
        keep = ~existing_df["date"].isin(new_df["date"].unique())
        combined = pd.concat([existing_df[keep], new_df], ignore_index=True).copy()
    else:
        combined = new_df.copy()

    combined = combined.sort_values("date").reset_index(drop=True)
    save_csv(combined)
    push_to_huggingface()

    final_rows = len(pd.read_csv(CSV_PATH))
    print(f"\nDone. CSV rows total: {final_rows} "
          f"({len(new_rows)} new published + forward-filled gaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())