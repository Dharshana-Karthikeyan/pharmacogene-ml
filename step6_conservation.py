"""
Step 6: Pull conservation scores (phyloP100way + GERP) for the labeled,
VEP-annotated dataset.

phyloP100way is natively available for hg38 via UCSC's REST API -- one
call per variant.

GERP is only hosted by UCSC for hg19. So for GERP we do, per variant:
  1) liftover the GRCh38 position to GRCh37 via Ensembl's REST assembly
     mapping endpoint
  2) query UCSC's hg19 GERP track ("allHg19RS_BW") at the lifted position

Every variant that fails liftover or scoring is counted, printed, and
saved to its own CSV -- never silently dropped.
"""

import time

import pandas as pd
import requests

from config import (
    PROCESSED_DIR,
    UCSC_API_URL,
    PHYLOP_TRACK,
    PHYLOP_GENOME,
    ENSEMBL_MAP_URL,
    GERP_TRACK,
    GERP_GENOME,
)


def _to_ucsc_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    if chrom.upper() in ("MT", "M"):
        return "chrM"
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def get_phylop_scores(df: pd.DataFrame, delay_seconds: float = 0.3) -> pd.DataFrame:
    """Query UCSC's hg38 phyloP100way track, one base per variant."""
    scores = []
    n_ok = 0

    for _, row in df.iterrows():
        chrom = _to_ucsc_chrom(row["Chromosome"])
        pos = int(float(row["Start"]))  # 1-based, from ClinVar
        start0, end0 = pos - 1, pos      # UCSC wants 0-based, half-open

        url = (
            f"{UCSC_API_URL}?genome={PHYLOP_GENOME};track={PHYLOP_TRACK};"
            f"chrom={chrom};start={start0};end={end0}"
        )
        score = None
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get(PHYLOP_TRACK, [])
            if items:
                score = items[0].get("value")
        except requests.RequestException as e:
            print(f"phyloP: request failed for {chrom}:{pos}: {e}")

        if score is not None:
            n_ok += 1
        scores.append(score)
        time.sleep(delay_seconds)

    df = df.copy()
    df["phylop100way"] = scores
    print(f"\nphyloP100way: {n_ok}/{len(df)} variants scored")
    return df


def liftover_grch38_to_grch37(df: pd.DataFrame, delay_seconds: float = 0.3) -> pd.DataFrame:
    """Convert each variant's GRCh38 position to GRCh37 via Ensembl REST."""
    hg19_chroms, hg19_starts, hg19_ends = [], [], []
    n_ok = 0

    for _, row in df.iterrows():
        chrom = str(row["Chromosome"]).strip()
        pos = int(float(row["Start"]))
        region = f"{chrom}:{pos}-{pos}:1"
        url = ENSEMBL_MAP_URL.format(region=region)

        mapped_chrom, mapped_start, mapped_end = None, None, None
        try:
            resp = requests.get(
                url, headers={"Content-Type": "application/json"}, timeout=30
            )
            resp.raise_for_status()
            mappings = resp.json().get("mappings", [])
            if mappings:
                mapped = mappings[0]["mapped"]
                mapped_chrom = mapped["seq_region_name"]
                mapped_start = mapped["start"]
                mapped_end = mapped["end"]
                n_ok += 1
        except (requests.RequestException, KeyError, IndexError) as e:
            print(f"Liftover: failed for {chrom}:{pos}: {e}")

        hg19_chroms.append(mapped_chrom)
        hg19_starts.append(mapped_start)
        hg19_ends.append(mapped_end)
        time.sleep(delay_seconds)

    df = df.copy()
    df["hg19_chrom"] = hg19_chroms
    df["hg19_start"] = hg19_starts
    df["hg19_end"] = hg19_ends
    print(f"Liftover GRCh38->GRCh37: {n_ok}/{len(df)} variants mapped")
    return df


def get_gerp_scores(df: pd.DataFrame, delay_seconds: float = 0.3) -> pd.DataFrame:
    """Query UCSC's hg19 GERP track ('allHg19RS_BW') using lifted-over coords.

    Expects df to already have hg19_chrom / hg19_start columns from
    liftover_grch38_to_grch37().
    """
    scores = []
    n_ok = 0

    for _, row in df.iterrows():
        if pd.isna(row.get("hg19_chrom")) or pd.isna(row.get("hg19_start")):
            scores.append(None)
            continue

        chrom = _to_ucsc_chrom(row["hg19_chrom"])
        pos = int(row["hg19_start"])
        start0, end0 = pos - 1, pos

        url = (
            f"{UCSC_API_URL}?genome={GERP_GENOME};track={GERP_TRACK};"
            f"chrom={chrom};start={start0};end={end0}"
        )
        score = None
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get(GERP_TRACK, [])
            if items:
                score = items[0].get("value")
            elif "error" in data:
                # Track name may differ from allHg19RS_BW -- surface it
                # loudly rather than silently returning None for every row.
                print(f"GERP: UCSC returned an error for {chrom}:{pos}: {data['error']}")
        except requests.RequestException as e:
            print(f"GERP: request failed for {chrom}:{pos}: {e}")

        if score is not None:
            n_ok += 1
        scores.append(score)
        time.sleep(delay_seconds)

    df = df.copy()
    df["gerp_score"] = scores
    print(f"GERP: {n_ok}/{len(df)} variants scored")
    return df


def pull_conservation_scores(featured_df: pd.DataFrame) -> pd.DataFrame:
    df = get_phylop_scores(featured_df)
    df = liftover_grch38_to_grch37(df)
    df = get_gerp_scores(df)

    n_phylop_missing = df["phylop100way"].isna().sum()
    n_gerp_missing = df["gerp_score"].isna().sum()
    print(f"\nConservation scoring summary: "
          f"{len(df) - n_phylop_missing}/{len(df)} have phyloP, "
          f"{len(df) - n_gerp_missing}/{len(df)} have GERP")

    missing = df[df["phylop100way"].isna() | df["gerp_score"].isna()]
    if len(missing) > 0:
        missing_path = PROCESSED_DIR / "variants_missing_conservation_scores.csv"
        missing.to_csv(missing_path, index=False)
        print(f"Variants missing one or both conservation scores saved: {missing_path}")

    out_path = PROCESSED_DIR / "labeled_dataset_with_conservation.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df