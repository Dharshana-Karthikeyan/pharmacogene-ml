"""
Step 10: Annotate the unlabeled variant pool (Phase 2) with VEP features
and conservation scores, using the same logic as steps 5 and 6.

Input:  data/processed/unlabeled_pool.csv  (1224 variants, 9 locked genes)
Output: data/processed/unlabeled_pool_annotated.csv

No functional labels are assigned here -- this step produces features only.
Pseudo-labeling happens in step 11 (self-training).
"""

import time
from pathlib import Path

import pandas as pd
import requests

from config import VEP_REGION_URL, PROCESSED_DIR

UCSC_PHYLOP_URL = "https://api.genome.ucsc.edu/getData/track"
UCSC_GERP_URL   = "https://api.genome.ucsc.edu/getData/track"
ENSEMBL_LIFTOVER_URL = "https://rest.ensembl.org/map/human/GRCh38/{region}/GRCh37"

DATA_PATH = PROCESSED_DIR / "unlabeled_pool.csv"
OUT_PATH  = PROCESSED_DIR / "unlabeled_pool_annotated.csv"


# ── VEP ──────────────────────────────────────────────────────────────────────

def _row_to_region_string(row: pd.Series) -> str | None:
    try:
        chrom = str(row["Chromosome"]).replace("chr", "")
        pos   = str(int(float(row["Start"])))
        vid   = row.get("rsid_norm", ".")
        vid   = vid if pd.notna(vid) else "."
        ref   = str(row["ReferenceAlleleVCF"])
        alt   = str(row["AlternateAlleleVCF"])
        if ref in ("", "nan") or alt in ("", "nan"):
            return None
        return f"{chrom} {pos} {vid} {ref} {alt} . . ."
    except (KeyError, ValueError, TypeError):
        return None


def run_vep(df: pd.DataFrame, batch_size: int = 200, delay: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df["region_string"] = df.apply(_row_to_region_string, axis=1)

    skipped = df["region_string"].isna().sum()
    if skipped:
        print(f"VEP: {skipped} variants skipped (missing coords/alleles)")

    annotatable = df[df["region_string"].notna()].copy()
    all_results = {}

    total_batches = (len(annotatable) + batch_size - 1) // batch_size
    for batch_num, start in enumerate(range(0, len(annotatable), batch_size), 1):
        batch   = annotatable.iloc[start:start + batch_size]
        regions = batch["region_string"].tolist()
        print(f"  VEP batch {batch_num}/{total_batches} ({len(regions)} variants)...")
        try:
            resp = requests.post(
                VEP_REGION_URL,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json"},
                json={"variants": regions},
                timeout=120,
            )
            resp.raise_for_status()
            for r in resp.json():
                all_results[r.get("input")] = r
        except requests.RequestException as e:
            print(f"  VEP batch {batch_num} failed: {e}")
        time.sleep(delay)

    def _extract(region_str, field):
        r = all_results.get(region_str)
        if r is None:
            return None
        if field == "most_severe_consequence":
            return r.get("most_severe_consequence")
        if field == "impact":
            tc = r.get("transcript_consequences", [])
            return tc[0].get("impact") if tc else None
        if field == "gnomad_af":
            for cv in r.get("colocated_variants", []):
                freqs = cv.get("frequencies")
                if freqs:
                    for allele_freqs in freqs.values():
                        if "gnomade" in allele_freqs:
                            return allele_freqs["gnomade"]
                        if "gnomadg" in allele_freqs:
                            return allele_freqs["gnomadg"]
            return None
        return None

    annotatable["most_severe_consequence"] = annotatable["region_string"].apply(
        lambda s: _extract(s, "most_severe_consequence"))
    annotatable["vep_impact"] = annotatable["region_string"].apply(
        lambda s: _extract(s, "impact"))
    annotatable["gnomad_af"] = annotatable["region_string"].apply(
        lambda s: _extract(s, "gnomad_af"))

    n_ann = annotatable["most_severe_consequence"].notna().sum()
    n_af  = annotatable["gnomad_af"].notna().sum()
    print(f"VEP: {n_ann}/{len(annotatable)} annotated, "
          f"{n_af}/{len(annotatable)} with gnomAD AF")
    return annotatable


# ── LIFTOVER + CONSERVATION ───────────────────────────────────────────────────

def _liftover_one(chrom: str, pos: int, delay: float = 0.3) -> tuple[str, int] | None:
    region = f"{chrom}:{pos}..{pos}"
    try:
        resp = requests.get(
            ENSEMBL_LIFTOVER_URL.format(region=region),
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        time.sleep(delay)
        if resp.status_code != 200:
            return None
        mappings = resp.json().get("mappings", [])
        if not mappings:
            return None
        m = mappings[0]["mapped"]
        return str(m["seq_region_name"]), int(m["start"])
    except Exception:
        return None


def _get_phylop(chrom: str, pos: int, delay: float = 0.3) -> float | None:
    try:
        resp = requests.get(
            UCSC_PHYLOP_URL,
            params={"genome": "hg38", "track": "phyloP100way",
                    "chrom": f"chr{chrom}", "start": pos - 1, "end": pos},
            timeout=20,
        )
        time.sleep(delay)
        if resp.status_code != 200:
            return None
        data = resp.json()
        items = data.get("phyloP100way", [])
        if isinstance(items, list) and items:
            return float(items[0].get("value"))
        return None
    except Exception:
        return None


def _get_gerp(chrom19: str, pos19: int, delay: float = 0.3) -> float | None:
    try:
        resp = requests.get(
            UCSC_GERP_URL,
            params={"genome": "hg19", "track": "allHg19RS_BW",
                    "chrom": f"chr{chrom19}",
                    "start": pos19 - 1, "end": pos19},
            timeout=20,
        )
        time.sleep(delay)
        if resp.status_code != 200:
            return None
        data = resp.json()
        items = data.get("allHg19RS_BW", [])
        if isinstance(items, list) and items:
            return float(items[0].get("value"))
        return None
    except Exception:
        return None


def run_conservation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    phylop_scores, gerp_scores = [], []
    hg19_chroms, hg19_starts = [], []

    total = len(df)
    print(f"Conservation scoring: {total} variants...")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            print(f"  ...{i}/{total}")
        try:
            chrom = str(row["Chromosome"]).replace("chr", "")
            pos   = int(float(row["Start"]))
        except (ValueError, TypeError):
            phylop_scores.append(None)
            gerp_scores.append(None)
            hg19_chroms.append(None)
            hg19_starts.append(None)
            continue

        phylop = _get_phylop(chrom, pos)
        phylop_scores.append(phylop)

        lifted = _liftover_one(chrom, pos)
        if lifted:
            chrom19, pos19 = lifted
            hg19_chroms.append(chrom19)
            hg19_starts.append(pos19)
            gerp = _get_gerp(chrom19, pos19)
            gerp_scores.append(gerp)
        else:
            hg19_chroms.append(None)
            hg19_starts.append(None)
            gerp_scores.append(None)

    df["phylop100way"] = phylop_scores
    df["gerp_score"]   = gerp_scores
    df["hg19_chrom"]   = hg19_chroms
    df["hg19_start"]   = hg19_starts

    n_phylop = sum(x is not None for x in phylop_scores)
    n_gerp   = sum(x is not None for x in gerp_scores)
    print(f"phyloP: {n_phylop}/{total} scored")
    print(f"GERP:   {n_gerp}/{total} scored")
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded unlabeled pool: {len(df)} variants")

    print("\n--- VEP annotation ---")
    df = run_vep(df)

    print("\n--- Conservation scores (phyloP + GERP) ---")
    df = run_conservation(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
