"""
Step 11: Semi-supervised self-training on pharmacogene variants (Phase 2).

Strategy:
- Start with 75 CPIC-labeled variants (Phase 1 dataset) as gold-standard anchor.
- Iteratively pseudo-label high-confidence predictions from the 1224 unlabeled pool.
- Compare final semi-supervised model against Phase 1 supervised baseline
  using the same Stratified 5-Fold CV framework for a fair, apples-to-apples
  comparison.

Confidence threshold: 0.85 (variant must have >= 85% predicted probability
for its top class to be added as a pseudo-label). Adjustable via CONF_THRESHOLD.
Max iterations: 10 (stops early if no new pseudo-labels added in a round).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

LABELED_PATH = Path(__file__).resolve().parent / "data" / "processed" / "labeled_dataset_expanded.csv"
UNLABELED_PATH = Path(__file__).resolve().parent / "data" / "processed" / "unlabeled_pool_annotated.csv"
OUT_DIR        = Path(__file__).resolve().parent / "data" / "processed"

VEP_IMPACT_ORDER = {"MODIFIER": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}
FEATURE_COLS     = ["most_severe_consequence", "vep_impact_ordinal",
                    "gnomad_af", "phylop100way", "gerp_score"]
NUMERIC_FEATURES = ["vep_impact_ordinal", "gnomad_af", "phylop100way", "gerp_score"]
CATEGORICAL_FEATURES = ["most_severe_consequence"]

CONF_THRESHOLD = 0.85
MAX_ITERATIONS = 10
N_SPLITS       = 5
RANDOM_STATE   = 42


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vep_impact_ordinal"] = df["vep_impact"].map(VEP_IMPACT_ORDER)
    return df[FEATURE_COLS].copy()


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


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced", n_estimators=300,
        random_state=RANDOM_STATE,
    )

def run_cv(X: pd.DataFrame, y: pd.Series,
           label_encoder: LabelEncoder, tag: str) -> dict:
    """Supervised-only CV — used for Phase 1 baseline evaluation."""
    y_enc = label_encoder.transform(y)
    class_names = label_encoder.classes_
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []
    fold_accs, fold_f1s = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(X, y_enc), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        preprocessor = build_preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc  = preprocessor.transform(X_test)

        model = build_model()
        model.fit(X_train_proc, y_train)

        preds = model.predict(X_test_proc)
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        fold_accs.append(accuracy_score(y_test, preds))
        fold_f1s.append(f1_score(y_test, preds, average="macro"))

    acc      = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
    per_cls  = f1_score(y_true_all, y_pred_all, average=None)

    print(f"\n{'='*60}")
    print(f"{tag} -- Stratified {N_SPLITS}-Fold CV results")
    print(f"{'='*60}")
    print(classification_report(y_true_all, y_pred_all,
                                target_names=class_names))
    print(f"Per-fold accuracy:  {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"Per-fold macro F1:  {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
    metrics = {"tag": tag, "accuracy": acc, "macro_f1": macro_f1,
               "accuracy_mean": np.mean(fold_accs), "accuracy_std": np.std(fold_accs),
               "macro_f1_mean": np.mean(fold_f1s), "macro_f1_std": np.std(fold_f1s),
               "n_gold_labeled": len(X), "n_pseudo_labeled": 0}
    for cls_name, f1 in zip(class_names, per_cls):
        metrics[f"f1_{cls_name.replace(' ', '_')}"] = f1
    return metrics

def run_cv_honest(
    X_gold: pd.DataFrame,
    y_gold: pd.Series,
    X_pseudo: pd.DataFrame,
    y_pseudo: pd.Series,
    label_encoder: LabelEncoder,
    tag: str,
) -> dict:
    """
    Honest semi-supervised CV evaluation:
    - CV folds drawn from gold-standard labeled variants only
    - Pseudo-labeled variants added to TRAINING folds only, never test folds
    - Test folds contain only gold-standard labeled variants
    """
    y_gold_enc = label_encoder.transform(y_gold)
    y_pseudo_enc = label_encoder.transform(y_pseudo) if len(y_pseudo) > 0 else np.array([])
    class_names = label_encoder.classes_

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []
    fold_accs, fold_f1s = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(X_gold, y_gold_enc), start=1):

        X_train_gold = X_gold.iloc[train_idx]
        y_train_gold = y_gold_enc[train_idx]
        X_test = X_gold.iloc[test_idx]
        y_test = y_gold_enc[test_idx]

        if len(X_pseudo) > 0:
            X_train_all = pd.concat([X_train_gold, X_pseudo], ignore_index=True)
            y_train_all = np.concatenate([y_train_gold, y_pseudo_enc])
        else:
            X_train_all = X_train_gold
            y_train_all = y_train_gold

        preprocessor = build_preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train_all)
        X_test_proc = preprocessor.transform(X_test)

        model = build_model()
        model.fit(X_train_proc, y_train_all)

        preds = model.predict(X_test_proc)
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        fold_accs.append(accuracy_score(y_test, preds))
        fold_f1s.append(f1_score(y_test, preds, average="macro"))

    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
    per_cls = f1_score(y_true_all, y_pred_all, average=None)

    print(f"\n{'='*60}")
    print(f"{tag} -- Honest Semi-Supervised CV")
    print(f"(test folds: gold-standard only | train folds: gold + pseudo-labeled)")
    print(f"{'='*60}")
    print(classification_report(y_true_all, y_pred_all,
                                target_names=class_names))
    print(f"Per-fold accuracy:  {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"Per-fold macro F1:  {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")

    metrics = {"tag": tag, "accuracy": acc, "macro_f1": macro_f1,
               "accuracy_mean": np.mean(fold_accs), "accuracy_std": np.std(fold_accs),
               "macro_f1_mean": np.mean(fold_f1s), "macro_f1_std": np.std(fold_f1s),
               "n_gold_labeled": len(X_gold),
               "n_pseudo_labeled": len(X_pseudo)}
    for cls_name, f1 in zip(class_names, per_cls):
        metrics[f"f1_{cls_name.replace(' ', '_')}"] = f1
    return metrics


def self_training_loop(
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_unlabeled: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> tuple[pd.DataFrame, pd.Series, list[dict]]:
    """
    Iterative self-training with confidence thresholding.
    Returns expanded (X, y) and a log of each iteration.
    """
    X_train = X_labeled.copy()
    y_train = y_labeled.copy()
    X_pool  = X_unlabeled.copy()
    iteration_log = []

    print(f"\nSelf-training: threshold={CONF_THRESHOLD}, "
          f"max_iterations={MAX_ITERATIONS}")
    print(f"Starting labeled pool: {len(X_train)} variants")
    print(f"Unlabeled pool: {len(X_pool)} variants\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        if len(X_pool) == 0:
            print(f"Iteration {iteration}: unlabeled pool exhausted. Stopping.")
            break

        # Train on current labeled set
        preprocessor = build_preprocessor()
        y_enc = label_encoder.transform(y_train)
        X_proc = preprocessor.fit_transform(X_train)

        model = build_model()
        model.fit(X_proc, y_enc)

        # Predict on unlabeled pool
        X_pool_proc = preprocessor.transform(X_pool)
        proba = model.predict_proba(X_pool_proc)
        max_proba = proba.max(axis=1)
        pred_class_idx = proba.argmax(axis=1)
        pred_labels = label_encoder.inverse_transform(pred_class_idx)

        # Select high-confidence predictions
        confident_mask = max_proba >= CONF_THRESHOLD
        n_confident = confident_mask.sum()

        print(f"Iteration {iteration}: "
              f"{n_confident}/{len(X_pool)} unlabeled variants "
              f"exceed confidence threshold {CONF_THRESHOLD}")

        if n_confident == 0:
            print("  No confident predictions. Stopping early.")
            break

        # Add pseudo-labeled variants to training set
        pseudo_X = X_pool[confident_mask].copy()
        pseudo_y = pd.Series(pred_labels[confident_mask],
                             index=pseudo_X.index)

        # Log class distribution of pseudo-labels
        pseudo_dist = pseudo_y.value_counts().to_dict()
        print(f"  Pseudo-label distribution: {pseudo_dist}")

        X_train = pd.concat([X_train, pseudo_X], ignore_index=True)
        y_train = pd.concat([y_train, pseudo_y], ignore_index=True)
        X_pool  = X_pool[~confident_mask].copy()

        iteration_log.append({
            "iteration": iteration,
            "n_pseudo_added": int(n_confident),
            "n_total_labeled": len(X_train),
            "n_remaining_unlabeled": len(X_pool),
            "pseudo_label_dist": str(pseudo_dist),
        })

        print(f"  Total labeled after iteration: {len(X_train)}, "
              f"remaining unlabeled: {len(X_pool)}")

    return X_train, y_train, iteration_log


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    labeled = pd.read_csv(LABELED_PATH)
    labeled = labeled[labeled["function_term"] != "Increased function"].copy()
    print(f"Labeled (gold standard): {len(labeled)} variants")

    unlabeled = pd.read_csv(UNLABELED_PATH)
    unlabeled = unlabeled[unlabeled["most_severe_consequence"].notna()].copy()
    print(f"Unlabeled (annotated pool): {len(unlabeled)} variants")

    # ── Fit label encoder on gold-standard classes ────────────────────────────
    label_encoder = LabelEncoder()
    label_encoder.fit(labeled["function_term"])
    print(f"Classes: {list(label_encoder.classes_)}")

    # ── Prepare features ─────────────────────────────────────────────────────
    X_labeled   = prepare_features(labeled)
    y_labeled   = labeled["function_term"].copy()
    X_unlabeled = prepare_features(unlabeled)

    # ── Phase 1 baseline CV (supervised only, same framework) ────────────────
    print("\n" + "="*60)
    print("PHASE 1 BASELINE: Supervised Random Forest (75 labeled only)")
    print("="*60)
    baseline_metrics = run_cv(X_labeled, y_labeled, label_encoder,
                              tag="Phase1_Supervised_Baseline")

    # ── Self-training ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: Self-training (iterative pseudo-labeling)")
    print("="*60)
    X_expanded, y_expanded, iteration_log = self_training_loop(
        X_labeled, y_labeled, X_unlabeled, label_encoder
    )

  # ── Phase 2 honest CV ─────────────────────────────────────────────────────
    # Separate pseudo-labeled from gold-standard in the expanded set
    X_pseudo_only = X_expanded.iloc[len(X_labeled):].copy()
    y_pseudo_only = y_expanded.iloc[len(X_labeled):].copy()
    print(f"\nPseudo-labeled variants added to training: {len(X_pseudo_only)}")

    print("\n" + "="*60)
    print("PHASE 2 EVALUATION: Honest semi-supervised CV")
    print("(tested on gold-standard variants only)")
    print("="*60)
    expanded_metrics = run_cv_honest(
        X_labeled, y_labeled,
        X_pseudo_only, y_pseudo_only,
        label_encoder,
        tag="Phase2_SemiSupervised_Honest"
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame([baseline_metrics, expanded_metrics])
    results_path = OUT_DIR / "step11_self_training_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved comparison: {results_path}")

    log_df = pd.DataFrame(iteration_log)
    log_path = OUT_DIR / "step11_iteration_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Saved iteration log: {log_path}")

# ── Confusion matrix for Phase 2 model ───────────────────────────────────
    print("\nGenerating confusion matrix (Phase 2 honest CV)...")
    cm_true, cm_pred = [], []
    class_names = label_encoder.classes_

    skf_cm = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
    y_gold_enc = label_encoder.transform(y_labeled)
    y_pseudo_enc = label_encoder.transform(y_pseudo_only)

    for train_idx, test_idx in skf_cm.split(X_labeled, y_gold_enc):
        X_train_gold = X_labeled.iloc[train_idx]
        y_train_gold = y_gold_enc[train_idx]
        X_test = X_labeled.iloc[test_idx]
        y_test = y_gold_enc[test_idx]

        X_train_all = pd.concat([X_train_gold, X_pseudo_only], ignore_index=True)
        y_train_all = np.concatenate([y_train_gold, y_pseudo_enc])

        preprocessor = build_preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train_all)
        X_test_proc  = preprocessor.transform(X_test)

        model = build_model()
        model.fit(X_train_proc, y_train_all)

        preds = model.predict(X_test_proc)
        cm_true.extend(y_test)
        cm_pred.extend(preds)

    cm = confusion_matrix(cm_true, cm_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix — Phase 2 Semi-Supervised RF\n(Stratified 5-Fold CV, gold-standard test folds only)", fontsize=11)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12)

    plt.tight_layout()
    cm_path = OUT_DIR / "step11_confusion_matrix_phase2.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")
    
    # ── Print final comparison ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL COMPARISON: Phase 1 vs Phase 2")
    print(f"{'='*60}")
    print(results_df[["tag", "n_gold_labeled", "n_pseudo_labeled",
                       "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()