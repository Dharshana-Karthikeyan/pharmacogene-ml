"""
Step 5: Pull VEP annotations (most severe consequence, impact, gnomAD AF)
for the labeled dataset -- feature engineering.
"""

import time

import pandas as pd
import requests

from config import VEP_REGION_URL, PROCESSED_DIR


def _clinvar_row_to_vep_region_string(row: pd.Series) -> str | None:
    try:
        chrom = str(row["Chromosome"]).replace("chr", "")
        pos = str(int(float(row["Start"])))
        vid = row["rsid_norm"] if pd.notna(row.get("rsid_norm")) else "."
        ref = str(row["ReferenceAlleleVCF"]) if pd.notna(row["ReferenceAlleleVCF"]) else ""
        alt = str(row["AlternateAlleleVCF"]) if pd.notna(row["AlternateAlleleVCF"]) else ""

        if ref in ("", "nan") or alt in ("", "nan"):
            return None

        return f"{chrom} {pos} {vid} {ref} {alt} . . ."
    except (KeyError, ValueError, TypeError):
        return None


def pull_vep_annotations(
    labeled_df: pd.DataFrame,
    batch_size: int = 200,
    delay_seconds: float = 1.0,
) -> pd.DataFrame:
    df = labeled_df.copy()
    df["region_string"] = df.apply(_clinvar_row_to_vep_region_string, axis=1)

    skipped = df[df["region_string"].isna()]
    if len(skipped) > 0:
        print(f"{len(skipped)} variants skipped (not clean SNVs -- indels/other, "
              f"need separate handling): {skipped['Name'].tolist()}")

    annotatable = df[df["region_string"].notna()].copy()
    all_results = {}

    for start in range(0, len(annotatable), batch_size):
        batch = annotatable.iloc[start:start + batch_size]
        regions = batch["region_string"].tolist()
        try:
            resp = requests.post(
                VEP_REGION_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json={"variants": regions},
                timeout=60,
            )
            resp.raise_for_status()
            results = resp.json()
        except requests.RequestException as e:
            print(f"VEP batch request failed: {e}")
            continue

        for r in results:
            all_results[r.get("input")] = r
        time.sleep(delay_seconds)

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
            # VEP returns combined-population keys as "gnomade" (exomes)
            # and "gnomadg" (genomes), NOT "gnomad". Prefer exome value,
            # fall back to genome value.
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

    n_annotated = annotatable["most_severe_consequence"].notna().sum()
    n_with_gnomad = annotatable["gnomad_af"].notna().sum()
    print(f"\nVEP annotation: {n_annotated}/{len(annotatable)} variants successfully annotated")
    print(f"gnomAD allele frequency: {n_with_gnomad}/{len(annotatable)} variants have a value "
          f"({len(annotatable) - n_with_gnomad} are None/missing)")

    out_path = PROCESSED_DIR / "labeled_dataset_with_vep_features.csv"
    annotatable.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if len(skipped) > 0:
        skipped_path = PROCESSED_DIR / "variants_skipped_vep_indels.csv"
        skipped.to_csv(skipped_path, index=False)
        print(f"Skipped (non-SNV) variants saved separately: {skipped_path}")

    return annotatable