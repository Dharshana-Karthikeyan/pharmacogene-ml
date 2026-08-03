"""
Step 4: Merge allele definitions with function labels to build the
variant -> function crosswalk, then merge with ClinVar for the final
labeled dataset.
"""

import pandas as pd
from pathlib import Path

from config import PROCESSED_DIR


def build_variant_function_crosswalk(
    def_files: dict[str, Path],
    function_labels_df: pd.DataFrame,
) -> pd.DataFrame:
    from step3_clinpgx import parse_allele_definition_file

    all_crosswalks = []

    for gene, path in def_files.items():
        try:
            cw = parse_allele_definition_file(path, gene)
        except ValueError as e:
            print(f"SKIPPED {gene}: {e}")
            continue

        n_before = len(cw)
        cw = cw[cw["allele_name"] != "*1"]
        print(f"{gene}: dropped {n_before - len(cw)} rows belonging to reference allele *1")

        gene_labels = function_labels_df[function_labels_df["gene"] == gene][
            ["allele_name", "function_term", "usable_label"]
        ]
        merged = cw.merge(gene_labels, on="allele_name", how="left")
        all_crosswalks.append(merged)

        print(f"{gene}: {len(cw)} (allele, variant) pairs parsed (excl. *1), "
              f"{cw['rsid'].notna().sum()} with an rsID")

    full = pd.concat(all_crosswalks, ignore_index=True) if all_crosswalks else pd.DataFrame()

    if not full.empty:
        has_rsid = full.dropna(subset=["rsid"])
        ambiguous = (
            has_rsid.groupby(["gene", "rsid"])["function_term"]
            .nunique()
            .reset_index(name="n_distinct_functions")
        )
        ambiguous = ambiguous[ambiguous["n_distinct_functions"] > 1]
        print(f"\n{len(ambiguous)} (gene, rsID) pairs map to MORE THAN ONE distinct "
              f"function label across different alleles -- documented ambiguity, "
              f"see data/processed/ambiguous_variant_labels.csv")
        ambiguous.to_csv(PROCESSED_DIR / "ambiguous_variant_labels.csv", index=False)

    out_path = PROCESSED_DIR / "variant_function_crosswalk.csv"
    full.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return full


def build_labeled_dataset(
    crosswalk_df: pd.DataFrame,
    clinvar_df: pd.DataFrame,
) -> pd.DataFrame:
    cw = crosswalk_df[crosswalk_df["usable_label"] == True].copy()

    has_rsid = cw.dropna(subset=["rsid"])
    ambig_check = (
        has_rsid.groupby(["gene", "rsid"])["function_term"]
        .nunique()
        .reset_index(name="n_distinct")
    )
    ambiguous_keys = set(
        ambig_check[ambig_check["n_distinct"] > 1]
        .apply(lambda r: (r["gene"], r["rsid"]), axis=1)
    )
    before = len(cw)
    cw = cw[~cw.apply(lambda r: (r["gene"], r["rsid"]) in ambiguous_keys, axis=1)]
    print(f"Excluded {before - len(cw)} crosswalk rows belonging to ambiguous (gene, rsid) pairs")

    cw_unique = cw.dropna(subset=["rsid"]).drop_duplicates(subset=["gene", "rsid"])[
        ["gene", "rsid", "allele_name", "function_term"]
    ].rename(columns={"allele_name": "defining_allele"})

    clinvar = clinvar_df.copy()
    clinvar["rsid_norm"] = clinvar["RS# (dbSNP)"].apply(
        lambda x: f"rs{int(float(x))}" if pd.notna(x) and str(x) not in ("-1", "nan") else None
    )

    labeled = clinvar.merge(
        cw_unique, left_on=["GeneSymbol", "rsid_norm"], right_on=["gene", "rsid"], how="inner"
    )

    print(f"\nFinal labeled dataset: {len(labeled)} ClinVar variants with a usable, "
          f"unambiguous function-status label")
    print(labeled.groupby("GeneSymbol")["function_term"].value_counts())

    out_path = PROCESSED_DIR / "labeled_dataset_phase1.csv"
    labeled.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return labeled
