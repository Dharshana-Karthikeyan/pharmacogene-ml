"""
Step 8: SHAP explainability for the selected model (Random Forest).

Trains a final Random Forest on the FULL labeled dataset (all usable rows) --
step 7's CV was for honest performance evaluation; this step's model is for
interpretation, so it uses all available signal.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just save to file
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "labeled_dataset_expanded.csv"
OUT_DIR = Path(__file__).resolve().parent / "data" / "processed"

VEP_IMPACT_ORDER = {"MODIFIER": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}
NUMERIC_FEATURES = ["vep_impact_ordinal", "gnomad_af", "phylop100way", "gerp_score"]
CATEGORICAL_FEATURES = ["most_severe_consequence"]


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df = df[df["function_term"] != "Increased function"].copy()
    df["vep_impact_ordinal"] = df["vep_impact"].map(VEP_IMPACT_ORDER)

    X = df[["most_severe_consequence", "vep_impact_ordinal", "gnomad_af",
            "phylop100way", "gerp_score"]].copy()
    y = df["function_term"].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    X, y = load_and_prepare_data()
    print(f"Training final Random Forest on {len(X)} rows (full dataset, no held-out split).")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    print(f"Processed feature count: {len(feature_names)} -> {list(feature_names)}")

    model = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42)
    model.fit(X_processed, y_encoded)

    # SHAP
    import numpy as np
    X_processed_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    X_processed_float = X_processed_dense.astype(np.float64)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed_float)

    # shap_values shape handling: newer SHAP versions return array of shape
    # (n_samples, n_features, n_classes); older versions return a list of
    # per-class arrays. Handle both.
    if isinstance(shap_values, list):
        shap_by_class = shap_values
    else:
        shap_by_class = [shap_values[:, :, i] for i in range(len(class_names))]

    X_df = pd.DataFrame(X_processed_float, columns=feature_names)

    # Overall importance summary (mean |SHAP| across all classes)
    print("\nGenerating SHAP summary bar plot (overall feature importance)...")
    plt.figure()
    shap.summary_plot(shap_by_class, X_df, class_names=class_names,
                       plot_type="bar", show=False)
    bar_path = OUT_DIR / "shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved: {bar_path}")

    # Per-class beeswarm plots
    for i, cls_name in enumerate(class_names):
        print(f"Generating SHAP beeswarm plot for class: {cls_name}...")
        plt.figure()
        shap.summary_plot(shap_by_class[i], X_df, show=False)
        safe_name = cls_name.replace(" ", "_")
        beeswarm_path = OUT_DIR / f"shap_beeswarm_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=150)
        plt.close()
        print(f"Saved: {beeswarm_path}")

    # Save raw mean |SHAP| values per feature per class as a table too
    import numpy as np
    rows = []
    for i, cls_name in enumerate(class_names):
        mean_abs = np.abs(shap_by_class[i]).mean(axis=0)
        for feat, val in zip(feature_names, mean_abs):
            rows.append({"class": cls_name, "feature": feat, "mean_abs_shap": val})
    importance_df = pd.DataFrame(rows)
    importance_csv = OUT_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"Saved: {importance_csv}")

    print("\nTop 5 features overall (mean |SHAP| across all classes):")
    overall = importance_df.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False)
    print(overall.head(5))


if __name__ == "__main__":
    main()
