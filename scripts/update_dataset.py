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
  python update_dataset.py                      # process only the latest PDF
  python update_dataset.py --backfill           # process all pages (all history)
  python update_dataset.py --backfill --max-pages 3
  python update_dataset.py --force              # re-process even if date exists

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

def _fetch_page_html(page_num: Optional[int]) -> str:
    url = (
        CBSL_PRICE_REPORT_URL
        if page_num is None
        else f"{CBSL_PRICE_REPORT_URL}?page={page_num}"
    )
    return _http_get(url).text


def list_pdf_links(max_pages: Optional[int] = None) -> Dict[str, str]:
    """
    Scrape the CBSL price-report listing and return {date_str: pdf_url}.
    Paginates until no more PDF links are found.

    date_str format: 'YYYYMMDD' (from the PDF filename).
    """
    links: Dict[str, str] = {}
    page = 0
    while True:
        if max_pages is not None and page >= max_pages:
            break
        html = _fetch_page_html(None if page == 0 else page)
        matches = list(PDF_LINK_RE.finditer(html))
        if not matches:
            if page == 0:
                raise RuntimeError(
                    "No price_report PDF links found on the CBSL page. "
                    "The website structure or PDF_LINK_RE may need updating."
                )
            break  # end of pagination
        for m in matches:
            links[m.group(1)] = m.group(0)
        page += 1
    return links


def find_latest_pdf_url() -> Tuple[str, str]:
    """Return (pdf_url, date_str) for the most recent report."""
    links = list_pdf_links(max_pages=1)
    latest = max(links.keys())
    return links[latest], latest


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

def load_existing_df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()


def save_csv(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    max_pages: Optional[int] = None
    if "--max-pages" in sys.argv:
        idx = sys.argv.index("--max-pages")
        if idx + 1 < len(sys.argv):
            max_pages = int(sys.argv[idx + 1])

    existing_df    = load_existing_df()
    existing_dates: set = (
        set(existing_df["date"].astype(str).unique()) if not existing_df.empty else set()
    )

    # Decide which PDFs to process
    if backfill:
        links = list_pdf_links(max_pages=max_pages)
        dates_to_process = sorted(links.keys())
        if not force:
            dates_to_process = [
                d for d in dates_to_process
                if datetime.strptime(d, "%Y%m%d").date().isoformat() not in existing_dates
            ]
        if not dates_to_process:
            print("Backfill: all dates already present. Nothing to do.")
            return 0
        print(f"Backfill: {len(dates_to_process)} date(s) to process.")
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
            # A single corrupt/missing PDF shouldn't prevent all others.
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

    # Merge: replace any existing rows for dates we just re-processed
    if not existing_df.empty:
        keep = ~existing_df["date"].isin(new_df["date"].unique())
        combined = pd.concat([existing_df[keep], new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values("date").reset_index(drop=True)
    save_csv(combined)
    push_to_huggingface()

    print(f"\nDone. Total rows: {len(combined)} | Added: {len(new_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())