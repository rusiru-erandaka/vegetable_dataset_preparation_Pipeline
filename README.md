# Sri Lanka Daily Price Dataset Pipeline

Automates extraction of selected items from the CBSL Daily Price Report PDF and appends them to a long-format CSV. It also enriches each date with rainfall for Nuwara Eliya and Polonnaruwa using Open-Meteo.

## Output Schema

Columns in `data/price_dataset.csv`:
- `date` (YYYY-MM-DD)
- `item`
- `unit`
- `retail_pettah`
- `retail_dambulla`
- `retail_narahenpita`
- `wholesale_pettah`
- `wholesale_dambulla`
- `rainfall_nuwara_eliya_mm`
- `rainfall_polonnaruwa_mm`
- `source_pdf`

## Local Run

```bash
python -m pip install -r requirements.txt
python scripts/update_dataset.py
```

Force reprocess even if the date already exists:

```bash
python scripts/update_dataset.py --force
```

Backfill all historical pages (optionally limit pages):

```bash
python scripts/update_dataset.py --backfill
python scripts/update_dataset.py --backfill --max-pages 3
```

## Hugging Face Upload

Create a `.env` file (do not commit it) with:

```
HF_DATASET_REPO=your-username/your-dataset-repo
HF_TOKEN=hf_your_token_here
```

## Notes

- Tabula (Java) is used as the primary table extractor when available. If Java is not installed, the script falls back to PDF text parsing with pdfplumber.

## GitHub Actions

The workflow runs daily and pushes the CSV to your Hugging Face dataset repo.
Store your token and repo in GitHub Secrets and write them to `.env` at runtime.
