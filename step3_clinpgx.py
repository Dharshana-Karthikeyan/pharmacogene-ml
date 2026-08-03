"""
Step 3: Pull ClinPGx/PharmGKB allele definitions and function labels.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from config import (
    GENE_PA_IDS,
    CLINPGX_HAPLOTYPE_URL,
    USABLE_FUNCTION_LABELS,
    ALLELE_DEF_DIR,
    RAW_DIR,
    PROCESSED_DIR,
)


def load_clinpgx_bulk(filename: str, gene_panel: list[str]) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download from ClinPGx downloads page, "
            f"unzip, and place here before running this function."
        )

    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    print(f"{filename}: columns = {list(df.columns)}")

    gene_col_candidates = [c for c in df.columns if "gene" in c.lower()]
    if not gene_col_candidates:
        raise ValueError(
            f"No column containing 'gene' found in {filename}; "
            f"inspect columns manually and adjust filter logic."
        )
    gene_col = gene_col_candidates[0]
    df_filtered = df[df[gene_col].astype(str).apply(
        lambda x: any(g in x for g in gene_panel)
    )]
    print(f"{filename}: {len(df_filtered)}/{len(df)} rows retained after gene filter")
    return df_filtered


def download_allele_definition_files(
    gene_pa_ids: dict[str, str] = GENE_PA_IDS,
    delay_seconds: float = 2.0,
) -> dict[str, Path]:
    saved_paths = {}

    for gene, pa_id in gene_pa_ids.items():
        url = CLINPGX_HAPLOTYPE_URL.format(pa_id=pa_id)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", {})
        except requests.RequestException as e:
            print(f"{gene}: failed to fetch metadata: {e}")
            time.sleep(delay_seconds)
            continue

        file_url = data.get("cpicS3File")
        if not file_url:
            print(f"{gene}: no cpicS3File link found (alleleType={data.get('alleleType')!r})")
            time.sleep(delay_seconds)
            continue

        local_path = ALLELE_DEF_DIR / f"{gene}_allele_definitions.xlsx"
        try:
            file_resp = requests.get(file_url, timeout=60)
            file_resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(file_resp.content)
            print(f"{gene}: saved allele definitions -> {local_path}")
            saved_paths[gene] = local_path
        except requests.RequestException as e:
            print(f"{gene}: failed to download definition file: {e}")

        time.sleep(delay_seconds)

    return saved_paths


def inspect_allele_definition_file(path: Path, max_rows: int = 15) -> None:
    xls = pd.ExcelFile(path)
    print(f"\n{path.name} -- sheets: {xls.sheet_names}")
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None, nrows=max_rows)
        print(f"\n--- Sheet: {sheet} (first {max_rows} rows) ---")
        print(df)


def parse_allele_definition_file(path: Path, gene: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Alleles", header=None)

    rsid_row_idx = None
    allele_header_row_idx = None
    for i in range(len(raw)):
        col0 = str(raw.iat[i, 0]).strip().lower()
        if col0 == "rsid":
            rsid_row_idx = i
        if col0.endswith("allele") and "effect" not in col0 and rsid_row_idx is not None:
            allele_header_row_idx = i
            break

    if rsid_row_idx is None or allele_header_row_idx is None:
        raise ValueError(
            f"{path.name}: could not locate rsID row and/or allele-header row "
            f"by the expected pattern -- layout differs from CYP2D6, inspect manually."
        )

    rsid_row = raw.iloc[rsid_row_idx]
    variant_label_row = raw.iloc[1]

    data_start = allele_header_row_idx + 1
    records = []

    for i in range(data_start, len(raw)):
        allele_name = raw.iat[i, 0]
        if pd.isna(allele_name) or str(allele_name).strip() == "":
            break

        for col in range(1, raw.shape[1]):
            value = raw.iat[i, col]
            if pd.isna(value):
                continue
            rsid = rsid_row.iloc[col] if col < len(rsid_row) else None
            variant_label = variant_label_row.iloc[col] if col < len(variant_label_row) else None
            records.append({
                "gene": gene,
                "allele_name": str(allele_name).strip(),
                "variant_label": variant_label,
                "rsid": rsid if pd.notna(rsid) else None,
                "allele_nt_value": value,
            })

    return pd.DataFrame(records)


def pull_clinpgx_allele_functions(
    gene_pa_ids: dict[str, str] = GENE_PA_IDS,
    delay_seconds: float = 2.0,
    max_retries: int = 5,
) -> pd.DataFrame:
    all_rows = []

    for expected_gene, pa_id in gene_pa_ids.items():
        url = CLINPGX_HAPLOTYPE_URL.format(pa_id=pa_id)

        payload = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", delay_seconds * attempt * 2))
                    print(f"{expected_gene}: rate limited (429), waiting {wait:.1f}s "
                          f"(attempt {attempt}/{max_retries})...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            except requests.RequestException as e:
                print(f"{expected_gene}: request error on attempt {attempt}: {e}")
                time.sleep(delay_seconds * attempt)

        if payload is None:
            print(f"FAILED to pull {expected_gene} ({pa_id}) after {max_retries} attempts")
            time.sleep(delay_seconds)
            continue

        data = payload.get("data", {})
        returned_gene = data.get("geneSymbol", "<missing>")

        if returned_gene != expected_gene:
            print(
                f"*** MISMATCH WARNING *** expected {expected_gene} for {pa_id}, "
                f"but API returned geneSymbol={returned_gene}. "
                f"SKIPPING this gene -- verify the PA ID manually before retrying."
            )
            time.sleep(delay_seconds)
            continue

        haplotypes = data.get("haplotypes", [])
        n_labeled = 0
        seen_function_terms = set()
        for h in haplotypes:
            function_term = h.get("functionTerm")
            seen_function_terms.add(function_term)
            row = {
                "gene": expected_gene,
                "pa_id": pa_id,
                "allele_name": h.get("name"),
                "function_term": function_term,
                "pharm_var_id": h.get("pharmVarId"),
                "amp_tier": h.get("ampTier"),
                "usable_label": function_term in USABLE_FUNCTION_LABELS,
            }
            all_rows.append(row)
            if row["usable_label"]:
                n_labeled += 1

        print(
            f"{expected_gene} ({pa_id}): verified OK, {len(haplotypes)} alleles, "
            f"{n_labeled} with usable function labels"
        )

        if len(haplotypes) == 0 or n_labeled == 0:
            print(
                f"  >>> DIAGNOSTIC for {expected_gene}: "
                f"alleleType={data.get('alleleType')!r}, "
                f"alleleFunctionSource={data.get('alleleFunctionSource')!r}, "
                f"distinct functionTerm values seen={seen_function_terms or 'none (no haplotypes)'}"
            )

        time.sleep(delay_seconds)

    df = pd.DataFrame(all_rows)

    out_path = PROCESSED_DIR / "clinpgx_allele_function_labels.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if not df.empty:
        summary = (
            df.groupby("gene")["usable_label"]
            .agg(total="count", usable="sum")
            .assign(excluded=lambda d: d["total"] - d["usable"])
        )
        print("\nPer-gene label availability:")
        print(summary)

    return df