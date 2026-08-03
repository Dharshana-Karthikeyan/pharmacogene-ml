"""
Runs the full pharmacogene labeling pipeline end-to-end, step by step.
Equivalent to the original monolithic script's __main__ block.
"""

from config import LOCKED_PANEL
from step2_clinvar import load_clinvar_for_panel
from step3_clinpgx import (
    pull_clinpgx_allele_functions,
    download_allele_definition_files,
    inspect_allele_definition_file,
)
from step4_merge import build_variant_function_crosswalk, build_labeled_dataset
from step5_features import pull_vep_annotations
from step6_conservation import pull_conservation_scores


if __name__ == "__main__":
    clinvar_df = load_clinvar_for_panel()
    print(clinvar_df.head())

    print("\n" + "=" * 70)
    print("Pulling ClinPGx allele function labels (step 3)...")
    print("=" * 70)
    function_labels_df = pull_clinpgx_allele_functions()
    print(function_labels_df.head())

    print("\n" + "=" * 70)
    print("Downloading allele definition files (step 4 prep)...")
    print("=" * 70)
    def_files = download_allele_definition_files()

    if "CYP2D6" in def_files:
        inspect_allele_definition_file(def_files["CYP2D6"])

    print("\n" + "=" * 70)
    print("Building variant -> function crosswalk (step 4)...")
    print("=" * 70)
    locked_def_files = {g: p for g, p in def_files.items() if g in LOCKED_PANEL}
    crosswalk_df = build_variant_function_crosswalk(locked_def_files, function_labels_df)
    print(crosswalk_df.head(10))

    print("\n" + "=" * 70)
    print("Merging with ClinVar to build final labeled dataset...")
    print("=" * 70)
    labeled_df = build_labeled_dataset(crosswalk_df, clinvar_df)

    print("\n" + "=" * 70)
    print("Pulling VEP annotations (step 5: feature engineering)...")
    print("=" * 70)
    featured_df = pull_vep_annotations(labeled_df)
    print(featured_df[["GeneSymbol", "Name", "most_severe_consequence",
                        "vep_impact", "gnomad_af", "function_term"]].head(10))

    print("\n" + "=" * 70)
    print("Pulling conservation scores (step 6: phyloP + GERP)...")
    print("=" * 70)
    conservation_df = pull_conservation_scores(featured_df)
    print(conservation_df[["GeneSymbol", "Name", "phylop100way",
                            "gerp_score", "function_term"]].head(10))
