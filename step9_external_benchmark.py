"""
Step 9: External benchmark against REVEL/CADD (via MyVariant.info).

Scope (Phase 1): descriptive + correlation comparison only.
Full threshold-based classifier benchmarking against REVEL/CADD is
scoped as Phase 2 future work (see project log) -- this step does NOT
attempt to convert REVEL/CADD into a competing classifier.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from scipy.stats import spearmanr

DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "labeled_dataset_expanded.csv"
OUT_DIR = Path(__file__).resolve().parent / "data" / "processed"
MYVARIANT_URL = "https://myvariant.info/v1/variant/{hgvs}"

# Map function_term to a numeric ordinal for correlation purposes only
# (NOT used as a training label -- purely for computing Spearman correlation
# against continuous REVEL/CADD scores).
FUNCTION_ORDINAL = {
    "No function": 0,
    "Decreased function": 1,
    "Normal function": 2,
}


def build_hgvs(row: pd.Series) -> str | None:
    try:
        chrom = str(row["Chromosome"]).replace("chr", "")
        pos = int(float(row["Start"]))
        ref = str(row["ReferenceAlleleVCF"])
        alt = str(row["AlternateAlleleVCF"])
        if ref in ("", "nan") or alt in ("", "nan"):
            return None
        return f"chr{chrom}:g.{pos}{ref}>{alt}"
    except (KeyError, ValueError, TypeError):
        return None


def fetch_revel_cadd(hgvs: str, delay_seconds: float = 0.3) -> dict:
    try:
        resp = requests.get(
            MYVARIANT_URL.format(hgvs=hgvs),
            params={
                "fields": "dbnsfp.revel.score,dbnsfp.cadd.phred,cadd.phred",
                "assembly": "hg38",
            },
            timeout=20,
        )
        time.sleep(delay_seconds)
        if resp.status_code == 404:
            return {"revel": None, "cadd": None}
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  {hgvs}: request failed: {e}")
        return {"revel": None, "cadd": None}

    dbnsfp = data.get("dbnsfp", {})
    revel = dbnsfp.get("revel", {}).get("score") if isinstance(dbnsfp.get("revel"), dict) else None

    cadd_phred = None
    if isinstance(dbnsfp.get("cadd"), dict):
        cadd_phred = dbnsfp["cadd"].get("phred")
    if cadd_phred is None and isinstance(data.get("cadd"), dict):
        cadd_phred = data["cadd"].get("phred")

    # Some fields return a list if multiple transcripts match -- take first
    if isinstance(revel, list):
        revel = revel[0] if revel else None
    if isinstance(cadd_phred, list):
        cadd_phred = cadd_phred[0] if cadd_phred else None

    return {"revel": revel, "cadd": cadd_phred}


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df["function_term"] != "Increased function"].copy()
    df["hgvs"] = df.apply(build_hgvs, axis=1)

    n_no_hgvs = df["hgvs"].isna().sum()
    print(f"{n_no_hgvs}/{len(df)} variants could not be converted to HGVS format (skipped).")

    revel_scores, cadd_scores = [], []
    print(f"\nQuerying MyVariant.info for {df['hgvs'].notna().sum()} variants...")
    for i, hgvs in enumerate(df["hgvs"], start=1):
        if hgvs is None:
            revel_scores.append(None)
            cadd_scores.append(None)
            continue
        result = fetch_revel_cadd(hgvs)
        revel_scores.append(result["revel"])
        cadd_scores.append(result["cadd"])
        if i % 10 == 0:
            print(f"  ...{i}/{len(df)} queried")

    df["revel_score"] = revel_scores
    df["cadd_phred"] = cadd_scores

    n_revel = df["revel_score"].notna().sum()
    n_cadd = df["cadd_phred"].notna().sum()
    print(f"\nREVEL score found: {n_revel}/{len(df)} variants "
          f"({len(df) - n_revel} missing -- expected, REVEL covers missense variants primarily)")
    print(f"CADD phred found: {n_cadd}/{len(df)} variants "
          f"({len(df) - n_cadd} missing)")

    df["function_ordinal"] = df["function_term"].map(FUNCTION_ORDINAL)

    print("\n--- Spearman correlations (function_ordinal vs score) ---")
    results = {}
    for score_col, score_name in [("revel_score", "REVEL"), ("cadd_phred", "CADD")]:
        valid = df[df[score_col].notna()]
        if len(valid) < 3:
            print(f"{score_name}: insufficient data ({len(valid)} non-null values) to compute correlation.")
            results[score_name] = {"n": len(valid), "rho": None, "p": None}
            continue
        rho, p = spearmanr(valid["function_ordinal"], valid[score_col])
        print(f"{score_name}: n={len(valid)}, Spearman rho={rho:.3f}, p={p:.4f}")
        results[score_name] = {"n": len(valid), "rho": rho, "p": p}

    # Boxplots: score distribution grouped by function_term
    for score_col, score_name in [("revel_score", "REVEL score"), ("cadd_phred", "CADD phred")]:
        valid = df[df[score_col].notna()]
        if len(valid) < 3:
            continue
        plt.figure(figsize=(7, 5))
        order = ["No function", "Decreased function", "Normal function"]
        data_by_class = [valid[valid["function_term"] == cls][score_col].dropna() for cls in order]
        plt.boxplot(data_by_class, tick_labels=order)
        plt.ylabel(score_name)
        plt.title(f"{score_name} by CPIC Functional Status")
        plt.tight_layout()
        out_path = OUT_DIR / f"step9_boxplot_{score_col}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    # Save full results table
    out_csv = OUT_DIR / "step9_external_benchmark_results.csv"
    df[["GeneSymbol", "Name", "function_term", "revel_score", "cadd_phred"]].to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    summary_csv = OUT_DIR / "step9_correlation_summary.csv"
    pd.DataFrame(results).T.to_csv(summary_csv)
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()