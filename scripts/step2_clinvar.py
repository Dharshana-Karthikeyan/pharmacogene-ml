"""
Step 2: Pull ClinVar, filtered to the gene panel.
"""

import pandas as pd
import requests
from pathlib import Path

from config import GENE_PANEL, CLINVAR_URL, CLINVAR_LOCAL, PROCESSED_DIR


def _remote_file_size(url: str) -> int | None:
    try:
        resp = requests.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        size = resp.headers.get("Content-Length")
        return int(size) if size is not None else None
    except requests.RequestException:
        return None


def download_clinvar(force: bool = False, max_retries: int = 8) -> Path:
    if force and CLINVAR_LOCAL.exists():
        CLINVAR_LOCAL.unlink()

    expected_size = _remote_file_size(CLINVAR_URL)

    if CLINVAR_LOCAL.exists():
        current_size = CLINVAR_LOCAL.stat().st_size
        if expected_size is not None and current_size >= expected_size:
            print(f"Using cached, complete {CLINVAR_LOCAL} ({current_size} bytes)")
            return CLINVAR_LOCAL
        print(f"Found partial download ({current_size} bytes) -- resuming.")
    else:
        current_size = 0

    for attempt in range(1, max_retries + 1):
        current_size = CLINVAR_LOCAL.stat().st_size if CLINVAR_LOCAL.exists() else 0
        headers = {"Range": f"bytes={current_size}-"} if current_size else {}
        mode = "ab" if current_size else "wb"

        try:
            print(f"Attempt {attempt}/{max_retries}: downloading from byte {current_size}...")
            resp = requests.get(CLINVAR_URL, headers=headers, stream=True, timeout=300)
            resp.raise_for_status()

            with open(CLINVAR_LOCAL, mode) as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

            final_size = CLINVAR_LOCAL.stat().st_size
            if expected_size is not None and final_size < expected_size:
                raise IOError(
                    f"Download incomplete: got {final_size} bytes, "
                    f"expected {expected_size}. Retrying..."
                )

            print(f"Download complete: {CLINVAR_LOCAL} ({final_size} bytes)")
            return CLINVAR_LOCAL

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download ClinVar file after {max_retries} attempts. "
                    f"The partial file has been kept at {CLINVAR_LOCAL} so the next "
                    f"run can resume from here -- just re-run the script."
                ) from e

    raise RuntimeError("Unreachable")


def load_clinvar_for_panel(gene_panel: list[str] = GENE_PANEL) -> pd.DataFrame:
    path = download_clinvar()

    usecols = [
        "GeneSymbol", "ClinicalSignificance", "VariationID", "Type", "Name",
        "ReviewStatus", "RS# (dbSNP)", "Chromosome", "Start", "Stop",
        "ReferenceAlleleVCF", "AlternateAlleleVCF", "Assembly",
    ]

    chunks = []
    reader = pd.read_csv(
        path, sep="\t", compression="gzip", usecols=lambda c: c in usecols,
        chunksize=200_000, dtype=str, low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[chunk["GeneSymbol"].isin(gene_panel)]
        chunk = chunk[chunk["Assembly"] == "GRCh38"]
        if not chunk.empty:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    print(f"ClinVar: {len(df)} variant rows across {df['GeneSymbol'].nunique()} panel genes")

    out_path = PROCESSED_DIR / "clinvar_panel_filtered.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df
