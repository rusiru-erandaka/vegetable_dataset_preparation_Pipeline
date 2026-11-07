"""
update_dataset.py
-----------------
Pipeline to:
  1. Find the latest Daily Price Report PDF from CBSL (by probing predictable URLs).
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

PDF discovery
-------------
The PDF URL is predictable from the date:
  .../price_report_YYYYMMDD_e.pdf
We probe each weekday URL. HTTP 404 = genuine holiday / non-publishing day.

FIX (this version): the probe previously used a bare `requests.head()` with the
default python-requests User-Agent and treated ANY non-200 / any exception as
"PDF does not exist". From GitHub Actions runners the CBSL server (or its WAF)
rejects such requests (403 / 405 / timeouts), so every date looked missing and
the run failed with "No PDF found in the last 10 weekdays".
The probe now:
  * sends browser-like headers,
  * treats only a real 404/410 as "does not exist",
  * falls back to a streamed GET (checking the %PDF magic bytes) when HEAD is
    rejected or inconclusive,
  * retries transient errors, and
  * reports the real reason when nothing is found.

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
from huggingface_hub import HfApi, hf_hub_download


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
HF_CSV_NAME = "price_dataset.csv"

# Browser-like headers. Many government sites / WAFs reject the default
# "python-requests/x.y" User-Agent, especially from cloud (datacenter) IPs.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": CBSL_PRICE_REPORT_URL,
}

SESSION = requests.Session()
SESSION.headers.update(HTTP_HEADERS)

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
    """Case-insensitive match: is any known item name present in `text`?"""
    text_lower = text.lower()
    for alias, canonical in _ITEM_LOWER_MAP.items():
        if alias in text_lower:
            return canonical
    return None


def _extract_values(line: str) -> List[float]:
    """
    Extract all price values from one item's text line, re-assembling the
    split-glyph tokens produced by pdfplumber (see module docstring).
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

            # Normal split (1–3 digits before the dot)
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

        i += 1  # skip non-numeric tokens

    return values


def _http_get(url: str, retries: int = 3, backoff: float = 2.0) -> requests.Response:
    """HTTP GET with exponential-backoff retry and browser-like headers."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=30)
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

CBSL_PDF_BASE = (
    "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/"
    "statistics/pricerpt/price_report_{date_str}_e.pdf"
)

# Earliest date CBSL has PDFs online with the same format we can parse.
CBSL_HISTORY_START = datetime(2023, 1, 1)


def _pdf_url(date_str: str) -> str:
    """Build the direct PDF URL for a given date string (YYYYMMDD)."""
    return CBSL_PDF_BASE.format(date_str=date_str)


def _probe_pdf(url: str, retries: int = 3, backoff: float = 2.0) -> Tuple[Optional[bool], str]:
    """
    Check whether a PDF exists at `url`.

    Returns (exists, reason):
      (True,  "...")  -> PDF is there (HTTP 200 and genuinely a PDF)
      (False, "HTTP 404") -> definitively not published (holiday etc.)
      (None,  "...")  -> INCONCLUSIVE (blocked / network error / server error).
                         Callers must NOT treat this as "no PDF published".

    Strategy:
      1. HEAD (cheap). 200 with a non-HTML content-type -> exists. 404/410 -> gone.
      2. Anything else (403, 405, 5xx, HTML 200 'soft block' page...) -> streamed
         GET, and check the first bytes are the '%PDF' magic number. Streaming
         means we read only a few bytes, not the whole file.
      3. Network exceptions are retried with backoff.
    """
    last_reason = "no attempt made"

    for attempt in range(retries):
        try:
            head = SESSION.head(url, timeout=20, allow_redirects=True)
            ctype = head.headers.get("Content-Type", "").lower()
            if head.status_code == 200 and "html" not in ctype:
                return True, "HTTP 200 (HEAD)"
            if head.status_code in (404, 410):
                return False, f"HTTP {head.status_code}"
            last_reason = f"HEAD returned HTTP {head.status_code} ({ctype or 'no content-type'})"

            # HEAD was rejected or inconclusive -> verify with a streamed GET.
            with SESSION.get(url, timeout=30, stream=True, allow_redirects=True) as resp:
                if resp.status_code in (404, 410):
                    return False, f"HTTP {resp.status_code}"
                if resp.status_code == 200:
                    first = next(resp.iter_content(chunk_size=8), b"")
                    if first.startswith(b"%PDF"):
                        return True, "HTTP 200 (GET)"
                    return None, "HTTP 200 but body is not a PDF (blocked / soft-404 page?)"
                last_reason = f"GET returned HTTP {resp.status_code} (HEAD was {head.status_code})"
        except requests.RequestException as exc:
            last_reason = f"{type(exc).__name__}: {exc}"

        if attempt < retries - 1:
            time.sleep(backoff ** attempt)

    return None, last_reason


def _pdf_exists(url: str) -> bool:
    """True only if the PDF is confirmed to exist."""
    exists, _ = _probe_pdf(url)
    return exists is True


def list_pdf_links(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, str]:
    """
    Probe every weekday PDF URL between start_date and end_date.
    Returns {date_str: pdf_url} for every date where the PDF exists.
    """
    if start_date is None:
        start_date = CBSL_HISTORY_START
    if end_date is None:
        end_date = datetime.utcnow()

    weekdays = pd.bdate_range(start=start_date, end=end_date)

    links: Dict[str, str] = {}
    inconclusive = 0
    total = len(weekdays)

    for i, day in enumerate(weekdays):
        date_str = day.strftime("%Y%m%d")
        url = _pdf_url(date_str)
        exists, reason = _probe_pdf(url)
        if exists:
            links[date_str] = url
        elif exists is None:
            inconclusive += 1
            print(f"  [warn] {date_str}: probe inconclusive — {reason}")
        if (i + 1) % 50 == 0:
            print(f"  Probed {i+1}/{total} dates, found {len(links)} PDFs so far...")

    print(f"  Done probing: {len(links)}/{total} weekdays have a PDF "
          f"({inconclusive} inconclusive).")
    if inconclusive:
        print("  NOTE: inconclusive dates are NOT treated as holidays; "
              "re-run the backfill later to pick them up.")
    return links


def find_latest_pdf_url(lookback: int = 10) -> Tuple[str, str]:
    """
    Find the most recent PDF by probing the last `lookback` weekdays (newest first).
    Raises with the real per-date reasons if nothing is found.
    """
    today = datetime.utcnow()
    days = pd.bdate_range(end=today, periods=lookback)

    reasons: List[str] = []
    for day in reversed(days):  # newest first
        date_str = day.strftime("%Y%m%d")
        url = _pdf_url(date_str)
        exists, reason = _probe_pdf(url)
        if exists:
            return url, date_str
        reasons.append(f"{date_str}: {reason}")

    detail = "\n    ".join(reasons)
    blocked = any("404" not in r and "410" not in r for r in reasons)
    hint = (
        "Probes were blocked or failed (not plain 404s) — the CBSL server is likely "
        "rejecting requests from this runner."
        if blocked else
        "Every date returned 404 — CBSL has not published in this window."
    )
    raise RuntimeError(f"No PDF found in the last {lookback} weekdays. {hint}\n    {detail}")


def download_pdf(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = _http_get(url).content
    if not content.startswith(b"%PDF"):
        raise RuntimeError(f"{url}: downloaded content is not a PDF (blocked?).")
    target_path.write_bytes(content)
    print(f"  Downloaded: {target_path.name}")


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

_PDF_DATE_RE = re.compile(
    r"\b(\d{1,2})\s*"
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "january":1, "february":2, "march":3,    "april":4,
    "may":5,     "june":6,     "july":7,     "august":8,
    "september":9,"october":10,"november":11,"december":12,
}


def _extract_date_from_text(text: str) -> Optional[str]:
    """Extract the report date from PDF text. Returns ISO date string or None."""
    m = _PDF_DATE_RE.search(text)
    if not m:
        return None
    day   = int(m.group(1))
    month = _MONTH_MAP[m.group(2).lower()]
    year  = int(m.group(3))
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None


def extract_pdf(pdf_path: Path, url_date: str) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Parse a CBSL Daily Price Report PDF and return (report_date, prices).

    - Item found with 10 values   -> extract normally
    - Item found with < 10 values -> all fields NaN (partial columns can't be mapped)
    - Item absent                 -> all fields NaN
    - Page 2 unreadable           -> all NaN, warning logged
    Raises RuntimeError only for a truly corrupt file (< 2 pages / cannot open).
    """
    _nan_prices: Dict[str, float] = {field: float("nan") for field, _ in PRICE_FIELDS}

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if len(pdf.pages) < 2:
                raise RuntimeError(
                    f"{pdf_path.name}: only {len(pdf.pages)} page(s) — expected at least 2."
                )
            page2_text = pdf.pages[1].extract_text(x_tolerance=4, y_tolerance=2)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{pdf_path.name}: could not open PDF — {exc}") from exc

    if not page2_text:
        print(f"  WARNING: {pdf_path.name}: page 2 returned no text. "
              f"All prices set to NaN.")
        return url_date, {item: dict(_nan_prices) for item in REQUIRED_ITEMS}

    pdf_date = _extract_date_from_text(page2_text)
    if pdf_date is None:
        print(f"  WARNING: {pdf_path.name}: date not found in PDF text — "
              f"using URL date {url_date}.")
        report_date = url_date
    elif pdf_date != url_date:
        print(f"  WARNING: {pdf_path.name}: PDF date ({pdf_date}) != "
              f"URL date ({url_date}). Using PDF date.")
        report_date = pdf_date
    else:
        report_date = pdf_date

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in page2_text.splitlines() if ln.strip()]
    extracted: Dict[str, Dict[str, float]] = {}

    for line in lines:
        if "Rs./" not in line:
            continue
        item = _match_item(line)
        if not item or item in extracted:
            continue
        values = _extract_values(line)
        if len(values) < 10:
            print(f"  INFO: {item}: {len(values)} values (truncated row) — storing NaN.")
            extracted[item] = dict(_nan_prices)
        else:
            extracted[item] = {field: values[idx] for field, idx in PRICE_FIELDS}

    prices: Dict[str, Dict[str, float]] = {}
    missing = []
    for item in REQUIRED_ITEMS:
        if item in extracted:
            prices[item] = extracted[item]
        else:
            prices[item] = dict(_nan_prices)
            missing.append(item)

    if missing:
        print(f"  INFO: {pdf_path.name}: {len(missing)} item(s) absent from PDF "
              f"(stored as NaN): {missing}")

    return report_date, prices


# ---------------------------------------------------------------------------
# Rainfall data
# ---------------------------------------------------------------------------

def fetch_precipitation_range(
    start_date: str, end_date: str, lat: float, lon: float
) -> Dict[str, float]:
    """Fetch daily precipitation totals (mm) from Open-Meteo for a date range."""
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
    """Build a column name like "retail_pettah_Beans"."""
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
    """Assemble one wide-format CSV row for a single report date."""
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

_LEGACY_COLUMNS = [
    "item", "unit",
    "retail_pettah", "retail_dambulla", "retail_narahenpita",
    "wholesale_pettah", "wholesale_dambulla",
]


def restore_from_huggingface() -> bool:
    """
    If the local CSV is missing (e.g. the GitHub Actions cache was evicted),
    pull the last published copy from Hugging Face so we never overwrite the
    full history with a one-row file.
    """
    load_dotenv()
    repo_id = os.getenv("HF_DATASET_REPO")
    token   = os.getenv("HF_TOKEN")
    if not repo_id or not token:
        return False
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=HF_CSV_NAME,
            repo_type="dataset",
            token=token,
            local_dir=str(DATA_DIR),
        )
        print(f"  [restore] Local CSV was missing — restored from Hugging Face ({repo_id}).")
        return CSV_PATH.exists()
    except Exception as exc:  # repo/file may not exist yet on a first-ever run
        print(f"  [restore] Could not restore from Hugging Face: {exc}")
        return False


def load_existing_df() -> pd.DataFrame:
    """
    Load the existing CSV and normalise it.

    - Restores from Hugging Face if the local file is missing.
    - Drops legacy long-format columns.
    - Drops forward-filled rows (is_interpolated == True). They are regenerated
      by fill_missing_dates(); keeping them would make the pipeline believe a
      holiday was a real publication and block real data from replacing it.
    - Drops rows with no price data at all.
    """
    if not CSV_PATH.exists():
        if not restore_from_huggingface():
            return pd.DataFrame()

    df = pd.read_csv(CSV_PATH).copy()

    legacy_present = [c for c in _LEGACY_COLUMNS if c in df.columns]
    if legacy_present:
        print(f"  [migrate] Dropping {len(legacy_present)} legacy columns: {legacy_present}")
        df = df.drop(columns=legacy_present)

    if "is_interpolated" in df.columns:
        interp = df["is_interpolated"].astype(str).str.lower().eq("true")
        if interp.any():
            print(f"  [load] Dropping {int(interp.sum())} forward-filled rows "
                  f"(they are regenerated on save).")
        df = df[~interp].drop(columns=["is_interpolated"])

    price_cols = [c for c in df.columns if c.startswith(("wholesale_", "retail_"))]
    if price_cols:
        no_data = df[price_cols].isna().all(axis=1)
        if no_data.any():
            print(f"  [migrate] Dropping {int(no_data.sum())} rows with no price data.")
            df = df[~no_data]

    return df.reset_index(drop=True)


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps caused by CBSL not publishing on public holidays by forward-filling
    the last published values, and flag those rows with is_interpolated=True.
    `df` must contain only REAL publications (load_existing_df guarantees this).
    """
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    added_days = len(full_range) - len(df)

    df = df.reindex(full_range)
    df.index.name = "date"

    # source_pdf is NaN only on rows that did not exist before reindexing.
    df["is_interpolated"] = df["source_pdf"].isna()

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
    """Upload price_dataset.csv to a Hugging Face dataset repo."""
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
        path_in_repo=HF_CSV_NAME,
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

    start_date_override: Optional[datetime] = None
    if "--start-date" in sys.argv:
        idx = sys.argv.index("--start-date")
        if idx + 1 < len(sys.argv):
            start_date_override = datetime.strptime(sys.argv[idx + 1], "%Y-%m-%d")

    # load_existing_df() returns REAL publications only (forward-filled rows
    # are dropped and regenerated on save).
    existing_df = load_existing_df()
    existing_dates: set = (
        set(existing_df["date"].astype(str).unique()) if not existing_df.empty else set()
    )

    if backfill:
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

    iso_dates  = [datetime.strptime(d, "%Y%m%d").date().isoformat() for d in dates_to_process]
    start_date, end_date = min(iso_dates), max(iso_dates)
    print(f"Fetching rainfall {start_date} -> {end_date}...")
    rain_nuwara_map = fetch_precipitation_range(start_date, end_date, *NUWARA_ELIYA)
    rain_pol_map    = fetch_precipitation_range(start_date, end_date, *POLONNARUWA)

    new_rows: List[Dict[str, object]] = []
    for date_compact in dates_to_process:
        url_date    = datetime.strptime(date_compact, "%Y%m%d").date().isoformat()
        pdf_url     = links[date_compact]
        pdf_path    = RAW_DIR / f"price_report_{date_compact}_e.pdf"

        print(f"\nProcessing {url_date}...")
        try:
            if not pdf_path.exists() or force:
                download_pdf(pdf_url, pdf_path)
            report_date, prices = extract_pdf(pdf_path, url_date)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}")
            if backfill:
                continue
            raise

        rain_nuwara = rain_nuwara_map.get(report_date, float("nan"))
        rain_pol    = rain_pol_map.get(report_date, float("nan"))
        new_rows.append(build_wide_row(report_date, pdf_url, prices, rain_nuwara, rain_pol))
        print(f"  OK — {len(REQUIRED_ITEMS)} items extracted (PDF date: {report_date}).")

    if not new_rows:
        print("No new rows produced.")
        return 1

    new_df = pd.DataFrame(new_rows)

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
