"""
Step 7: Train/compare Logistic Regression, Random Forest, and XGBoost
using Stratified 5-Fold CV on the labeled pharmacogene dataset.

Decisions applied (see project log):
- 'Increased function' class dropped (n=2, insufficient for CV) -> 3-class problem.
- GeneSymbol excluded as a feature (avoid gene-identity shortcut learning).
- Class imbalance handled via class_weight / sample_weight (not oversampling).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "labeled_dataset_expanded.csv"
OUT_PATH = Path(__file__).resolve().parent / "data" / "processed" / "step7_model_comparison_results.csv"

VEP_IMPACT_ORDER = {"MODIFIER": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    n_total = len(df)

    n_increased = (df["function_term"] == "Increased function").sum()
    df = df[df["function_term"] != "Increased function"].copy()
    print(f"Excluded {n_increased} 'Increased function' row(s) (insufficient for CV). "
          f"{len(df)}/{n_total} rows retained for modeling.")

    df["vep_impact_ordinal"] = df["vep_impact"].map(VEP_IMPACT_ORDER)
    n_unmapped = df["vep_impact_ordinal"].isna().sum()
    if n_unmapped > 0:
        print(f"WARNING: {n_unmapped} row(s) had an unrecognized vep_impact value; "
              f"these will be median-imputed.")

    n_missing_af = df["gnomad_af"].isna().sum()
    print(f"gnomad_af missing for {n_missing_af}/{len(df)} rows -> will be median-imputed "
          f"within each CV fold (fit on training data only, no leakage).")

    feature_cols = ["most_severe_consequence", "vep_impact_ordinal", "gnomad_af",
                     "phylop100way", "gerp_score"]
    X = df[feature_cols].copy()
    y = df["function_term"].copy()

    print(f"\nFinal feature set: {feature_cols}")
    print(f"Class distribution:\n{y.value_counts()}\n")

    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["vep_impact_ordinal", "gnomad_af", "phylop100way", "gerp_score"]
    categorical_features = ["most_severe_consequence"]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, random_state=42, eval_metric="mlogloss",
        ),
    }


def run_cv_comparison(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_models()
    results = []

    for model_name, model in models.items():
        print(f"\n{'=' * 60}\nRunning {n_splits}-fold CV: {model_name}\n{'=' * 60}")

        y_true_all, y_pred_all = [], []
        fold_accs, fold_f1s = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            preprocessor = build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            if model_name == "XGBoost":
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
                model.fit(X_train_proc, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train_proc, y_train)

            preds = model.predict(X_test_proc)
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

            fold_acc = accuracy_score(y_test, preds)
            fold_f1 = f1_score(y_test, preds, average="macro")
            fold_accs.append(fold_acc)
            fold_f1s.append(fold_f1)
            print(f"  Fold {fold_idx}: accuracy = {fold_acc:.3f}, macro F1 = {fold_f1:.3f}")

        import numpy as np
        overall_acc = accuracy_score(y_true_all, y_pred_all)
        macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
        per_class_f1 = f1_score(y_true_all, y_pred_all, average=None)

        acc_mean = np.mean(fold_accs)
        acc_std  = np.std(fold_accs)
        f1_mean  = np.mean(fold_f1s)
        f1_std   = np.std(fold_f1s)

        print(f"\n{model_name} -- aggregated out-of-fold results:")
        print(classification_report(y_true_all, y_pred_all, target_names=class_names))
        print(f"Per-fold accuracy:  {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"Per-fold macro F1:  {f1_mean:.3f} ± {f1_std:.3f}")

        row = {"model": model_name,
               "accuracy": overall_acc,
               "accuracy_mean": acc_mean, "accuracy_std": acc_std,
               "macro_f1": macro_f1,
               "macro_f1_mean": f1_mean, "macro_f1_std": f1_std}
        for cls_name, f1 in zip(class_names, per_class_f1):
            row[f"f1_{cls_name.replace(' ', '_')}"] = f1
        results.append(row)

    return pd.DataFrame(results)


def main():
    X, y = load_and_prepare_data()
    results_df = run_cv_comparison(X, y, n_splits=5)

    print(f"\n{'=' * 60}\nFINAL MODEL COMPARISON\n{'=' * 60}")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
