"""
Step 12: SHAP explainability for the Phase 2 semi-supervised model.

Trains final Random Forest on gold-standard labeled (73) + all pseudo-labeled
variants (824) from step 11, then generates SHAP plots comparable to step 8.

Comparison with step 8 (Phase 1 SHAP) shows whether semi-supervised training
shifted feature importance, particularly for the Decreased function class.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

LABELED_PATH = Path(__file__).resolve().parent / "data" / "processed" / "labeled_dataset_expanded.csv"
UNLABELED_PATH = Path(__file__).resolve().parent / "data" / "processed" / "unlabeled_pool_annotated.csv"
RESULTS_PATH   = Path(__file__).resolve().parent / "data" / "processed" / "step11_self_training_results.csv"
OUT_DIR        = Path(__file__).resolve().parent / "data" / "processed"

VEP_IMPACT_ORDER     = {"MODIFIER": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}
NUMERIC_FEATURES     = ["vep_impact_ordinal", "gnomad_af", "phylop100way", "gerp_score"]
CATEGORICAL_FEATURES = ["most_severe_consequence"]
CONF_THRESHOLD       = 0.85


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vep_impact_ordinal"] = df["vep_impact"].map(VEP_IMPACT_ORDER)
    return df[["most_severe_consequence", "vep_impact_ordinal",
               "gnomad_af", "phylop100way", "gerp_score"]].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def get_pseudo_labels(
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_unlabeled: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce step 11's self-training to get pseudo-labeled pool."""
    from sklearn.utils.class_weight import compute_sample_weight

    X_train = X_labeled.copy()
    y_train = y_labeled.copy()
    X_pool  = X_unlabeled.copy()

    print(f"Reproducing self-training (threshold={CONF_THRESHOLD})...")

    for iteration in range(1, 11):
        if len(X_pool) == 0:
            break
        preprocessor = build_preprocessor()
        y_enc = label_encoder.transform(y_train)
        X_proc = preprocessor.fit_transform(X_train)

        model = RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=42)
        model.fit(X_proc, y_enc)

        X_pool_proc = preprocessor.transform(X_pool)
        proba = model.predict_proba(X_pool_proc)
        max_proba = proba.max(axis=1)
        pred_class_idx = proba.argmax(axis=1)
        pred_labels = label_encoder.inverse_transform(pred_class_idx)

        confident_mask = max_proba >= CONF_THRESHOLD
        n_confident = confident_mask.sum()
        print(f"  Iteration {iteration}: {n_confident} pseudo-labels added")

        if n_confident == 0:
            break

        pseudo_X = X_pool[confident_mask].copy()
        pseudo_y = pd.Series(pred_labels[confident_mask],
                             index=pseudo_X.index)

        X_train = pd.concat([X_train, pseudo_X], ignore_index=True)
        y_train = pd.concat([y_train, pseudo_y], ignore_index=True)
        X_pool  = X_pool[~confident_mask].copy()

    n_pseudo = len(X_train) - len(X_labeled)
    print(f"Total pseudo-labeled: {n_pseudo}")
    X_pseudo = X_train.iloc[len(X_labeled):].copy()
    y_pseudo = y_train.iloc[len(X_labeled):].copy()
    return X_pseudo, y_pseudo


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    labeled = pd.read_csv(LABELED_PATH)
    labeled = labeled[labeled["function_term"] != "Increased function"].copy()
    unlabeled = pd.read_csv(UNLABELED_PATH)
    unlabeled = unlabeled[unlabeled["most_severe_consequence"].notna()].copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(labeled["function_term"])
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    X_labeled   = prepare_features(labeled)
    y_labeled   = labeled["function_term"].copy()
    X_unlabeled = prepare_features(unlabeled)

    # ── Reproduce pseudo-labels from step 11 ─────────────────────────────────
    X_pseudo, y_pseudo = get_pseudo_labels(
        X_labeled, y_labeled, X_unlabeled, label_encoder)

    # ── Build full training set (gold + pseudo) ───────────────────────────────
    X_full = pd.concat([X_labeled, X_pseudo], ignore_index=True)
    y_full = pd.concat([y_labeled, y_pseudo], ignore_index=True)
    print(f"\nFull training set: {len(X_full)} variants "
          f"({len(X_labeled)} gold + {len(X_pseudo)} pseudo-labeled)")
    print(f"Class distribution:\n{y_full.value_counts()}")

  # ── Train final model ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(X_full)
    feature_names = preprocessor.get_feature_names_out()

    y_enc = label_encoder.transform(y_full)
    model = RandomForestClassifier(
        class_weight="balanced", n_estimators=300, random_state=42)
    model.fit(X_proc, y_enc)
    print(f"\nFinal model trained on {len(X_full)} variants.")

    import joblib
    model_data = {
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_cols": ["most_severe_consequence", "vep_impact_ordinal",
                         "gnomad_af", "phylop100way", "gerp_score"],
        "vep_impact_order": VEP_IMPACT_ORDER,
        "training_n_gold": len(X_labeled),
        "training_n_pseudo": len(X_pseudo),
        "training_n_total": len(X_full),
        "classes": list(label_encoder.classes_),
    }
    model_path = OUT_DIR / "phase2_rf_model.joblib"
    joblib.dump(model_data, model_path)
    print(f"Saved trained model: {model_path}")
    
    # ── SHAP ──────────────────────────────────────────────────────────────────
    X_dense = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc
    X_df = pd.DataFrame(X_dense, columns=feature_names)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        shap_by_class = shap_values
    else:
        shap_by_class = [shap_values[:, :, i]
                         for i in range(len(class_names))]

    # Overall summary bar plot
    print("\nGenerating SHAP summary bar plot (Phase 2)...")
    plt.figure()
    shap.summary_plot(shap_by_class, X_df, class_names=class_names,
                      plot_type="bar", show=False)
    bar_path = OUT_DIR / "shap_phase2_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved: {bar_path}")

    # Per-class beeswarm plots
    for i, cls_name in enumerate(class_names):
        print(f"Generating beeswarm: {cls_name}...")
        plt.figure()
        shap.summary_plot(shap_by_class[i], X_df, show=False)
        safe_name = cls_name.replace(" ", "_")
        path = OUT_DIR / f"shap_phase2_beeswarm_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # Save raw importance table
    rows = []
    for i, cls_name in enumerate(class_names):
        mean_abs = np.abs(shap_by_class[i]).mean(axis=0)
        for feat, val in zip(feature_names, mean_abs):
            rows.append({"phase": "Phase2", "class": cls_name,
                         "feature": feat, "mean_abs_shap": val})
    importance_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "shap_phase2_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Compare top features Phase 1 vs Phase 2
    phase1_path = OUT_DIR / "shap_feature_importance.csv"
    if phase1_path.exists():
        p1 = pd.read_csv(phase1_path)
        p1_overall = (p1.groupby("feature")["mean_abs_shap"]
                        .mean().sort_values(ascending=False))
        p2_overall = (importance_df.groupby("feature")["mean_abs_shap"]
                        .mean().sort_values(ascending=False))
        print("\n--- Top 5 features: Phase 1 vs Phase 2 ---")
        print("Phase 1:")
        print(p1_overall.head(5).to_string())
        print("\nPhase 2:")
        print(p2_overall.head(5).to_string())
    else:
        overall = (importance_df.groupby("feature")["mean_abs_shap"]
                     .mean().sort_values(ascending=False))
        print("\nTop 5 features (Phase 2):")
        print(overall.head(5).to_string())


if __name__ == "__main__":
    main()
