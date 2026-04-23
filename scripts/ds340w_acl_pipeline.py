# -*- coding: utf-8 -*-
"""
ds340w_acl_pipeline.py
======================
DS 340W — ACL Injury Prediction: Production Pipeline
Author : Moulik Jain

Scientific grounding:
    Jauhiainen et al. (2022). "Predicting ACL Injury Using Machine Learning on
    Data From an Extensive Screening Test Battery of 880 Female Elite Athletes."
    Am J Sports Med, 50(11): 2917-2924.  DOI: 10.1177/03635465221112095

Peer-reviewer concerns addressed (see DS340W_ReviewersCommentResponse.pdf):
  1. Overfitting / Sample Size  → 10-Fold Stratified CV; SMOTE applied *inside*
                                   each training fold only (no leakage).
  2. 100% Precision artefact    → Precision-Recall trade-off printed explicitly;
                                   threshold choice explained in console output.
  3. Generalizability           → Three heterogeneous Kaggle datasets fused into
                                   one master frame with dataset-source indicator.
  4. Interpretability           → SHAP summary + feature-importance bar chart
                                   highlighting the engineered "Cutting Signature."
  5. Real-world coach use       → 65% Yellow-Flag dashboard with action protocol.

Datasets (downloaded via kagglehub):
  • ziya07/athlete-injury-and-performance-dataset
  • anjalibhegam/multimodal-sports-injury-prediction-dataset
  • yuanchunhong/university-football-injury-prediction-dataset

Required packages (install once):
    pip install kagglehub imbalanced-learn xgboost lightgbm shap
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import glob
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn core
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)

# Imbalanced-learn (SMOTE lives inside each fold — prevents data leakage)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional heavy dependencies — handled gracefully so the script runs even if
# a package is missing (reviewer concern: reproducibility on different machines)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    print("[WARN] XGBoost unavailable — run: brew install libomp && pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    print("[WARN] LightGBM unavailable — run: pip install lightgbm")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not found — run: pip install shap")



# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════════

# Each dataset is identified by its unique target column name.
# This lets us auto-classify any uploaded CSV without caring about file names.
_TARGET_TO_ALIAS = {
    "injury_indicator":    "collegiate",   # ziya07 dataset
    "injury_risk":         "multimodal",   # anjalibhegam dataset
    "injury_next_season":  "football",     # yuanchunhong dataset
}


def load_uploaded_datasets(search_dir: str = ".") -> dict:
    """
    Load datasets from CSV files already present in `search_dir`
    (e.g. uploaded via Colab's file panel or files.upload()).

    Detection logic: each CSV is classified by which target column it
    contains (case-insensitive).  Multiple shards for the same dataset
    (train / val / test) are automatically concatenated.

    Returns a dict { alias -> pd.DataFrame }.
    """
    csvs = glob.glob(os.path.join(search_dir, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in '{search_dir}'. "
            "Upload all three dataset files first."
        )

    buckets: dict = {"collegiate": [], "multimodal": [], "football": []}

    for path in sorted(csvs):
        try:
            # Read only the header row to identify the dataset cheaply
            cols_lower = [c.lower() for c in pd.read_csv(path, nrows=0).columns]
        except Exception:
            continue

        matched = False
        for target_col, alias in _TARGET_TO_ALIAS.items():
            if target_col in cols_lower:
                buckets[alias].append(path)
                matched = True
                break
        if not matched:
            print(f"[SKIP] Could not classify: {os.path.basename(path)}")

    dataframes = {}
    for alias, paths in buckets.items():
        if not paths:
            print(f"[WARN] No files found for dataset '{alias}' — "
                  f"expected a column named: "
                  f"{[k for k,v in _TARGET_TO_ALIAS.items() if v==alias][0]}")
            continue
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        print(f"[LOAD] {alias:12s}: {df.shape[0]:>6,} rows × {df.shape[1]} cols  "
              f"({len(paths)} file(s): {[os.path.basename(p) for p in paths]})")
        dataframes[alias] = df

    if not dataframes:
        raise ValueError("No datasets could be loaded. Check that the uploaded "
                         "files contain the expected target columns.")
    return dataframes


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

# Canonical name → ordered list of possible raw column names per dataset.
# The first match found wins; all values are min-max scaled to [0, 1] so
# features from different measurement units become comparable.
COLUMN_MAP = {
    # ── Workload / training load ──────────────────────────────────────────
    "workload":           ["Training_Hours_Per_Week", "workload_intensity",
                           "training_hours_per_week"],
    "training_intensity": ["Training_Intensity", "workload_intensity",
                           "training_hours_per_week"],
    # ── Fatigue ───────────────────────────────────────────────────────────
    "fatigue":            ["Fatigue_Score", "fatigue_index", "stress_level"],
    # ── Recovery ──────────────────────────────────────────────────────────
    "recovery_days":      ["Recovery_Days_Per_Week", "rest_period", "sleep_hours"],
    # ── Previous injury history ───────────────────────────────────────────
    "prev_injury":        ["ACL_Risk_Score", "previous_injury_history",
                           "injury_history"],
    # ── Speed / deceleration proxy ────────────────────────────────────────
    "speed":              ["Performance_Score", "speed", "sprint_speed"],
    # ── Movement symmetry ─────────────────────────────────────────────────
    "symmetry":           ["Load_Balance_Score", "gait_symmetry", "agility"],
    # ── Impact / joint load ───────────────────────────────────────────────
    "impact":             ["ACL_Risk_Score", "ground_reaction_force",
                           "knee_strength"],
    # ── Match / competition schedule ──────────────────────────────────────
    "match_load":         ["Match_Count_Per_Week", "step_count",
                           "matches_played"],
    # ── Injury labels (handled separately below) ──────────────────────────
    "injury_label":       ["Injury_Indicator", "injury_risk",
                           "Injury_Next_Season"],
}


def _extract_and_scale(df: pd.DataFrame, candidates: list) -> pd.Series:
    """
    Find the first matching column (case-insensitive) from `candidates`,
    coerce to float, impute NaN with the column median, then min-max scale
    to [0, 1].  Returns a zero-filled Series if nothing matches.
    """
    for col in candidates:
        matched = [c for c in df.columns if c.lower() == col.lower()]
        if matched:
            s = pd.to_numeric(df[matched[0]], errors="coerce")
            s = s.fillna(s.median())
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else s * 0.0
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def align_dataset(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Map a raw source DataFrame onto the canonical 10-feature schema.
    The `source` column lets the models learn dataset-specific baselines,
    which partially compensates for different measurement protocols across
    datasets — analogous to Jauhiainen et al.'s sport/year stratification.
    """
    out = pd.DataFrame(index=df.index)
    out["source"] = source_name

    for feature, candidates in COLUMN_MAP.items():
        if feature == "injury_label":
            continue
        out[feature] = _extract_and_scale(df, candidates)

    # Injury label — round to binary 0/1 after scaling
    out["injury_label"] = (
        _extract_and_scale(df, COLUMN_MAP["injury_label"]).round().astype(int)
    )
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive three biomechanical composite features — the "Cutting Signature" —
    inspired by Jauhiainen et al. (2022) and the midterm SHAP findings.

    1. Fatigue_Accumulation
       Weighted sum of fatigue and workload minus recovery, approximating the
       "rolling 7-day load accumulation" concept from wearable-sensor research
       (Rossi et al., 2018; Dower et al., 2019).

    2. Lateral_Force_Proxy
       Speed × (1 − symmetry): athletes who decelerate fast AND move
       asymmetrically generate the lateral knee abduction moments that
       Jauhiainen et al. flag as a key — though poorly isolated — risk factor.

    3. Symmetry_Index
       |impact − symmetry|: large deltas indicate that one limb absorbs
       disproportionate ground-reaction force, a recognised ACL precursor.
    """
    df = df.copy()

    df["Fatigue_Accumulation"] = (
        0.50 * df["fatigue"]
        + 0.30 * df["workload"]
        - 0.20 * df["recovery_days"]
    ).clip(0.0, 1.0)

    df["Lateral_Force_Proxy"] = (
        df["speed"] * (1.0 - df["symmetry"])
    ).clip(0.0, 1.0)

    df["Symmetry_Index"] = (
        (df["impact"] - df["symmetry"]).abs()
    ).clip(0.0, 1.0)

    return df


def preprocess_data(dataframes: dict):
    """
    Full preprocessing pipeline.
    Returns X (pd.DataFrame, all numeric) and y (pd.Series, binary 0/1).
    """
    print("\n[PREPROCESS] Aligning features across datasets...")
    parts = []
    for name, df in dataframes.items():
        aligned = align_dataset(df, name)
        aligned = engineer_features(aligned)
        parts.append(aligned)
        injury_rate = aligned["injury_label"].mean() * 100
        print(f"             {name:12s}: {len(aligned):>5,} rows  "
              f"|  injury rate {injury_rate:.1f}%")

    master = pd.concat(parts, ignore_index=True)

    # One-hot encode data source so the model can learn dataset-level offsets
    master = pd.get_dummies(master, columns=["source"], prefix="src", dtype=float)

    y = master.pop("injury_label").astype(int)
    X = master.select_dtypes(include=[np.number])

    total_injury_rate = y.mean() * 100
    print(f"\n[PREPROCESS] Master frame: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"             Overall injury prevalence: {total_injury_rate:.1f}%  "
          f"(class ratio ≈ 1:{int(round((1-y.mean())/y.mean()))})")
    return X, y


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — ADVANCED ML ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

def build_model_zoo(random_state: int = 42) -> dict:
    """
    Construct four candidate pipelines.

    Each pipeline follows the pattern:
        StandardScaler → SMOTE → Classifier

    SMOTE is placed *inside* the pipeline so it is applied only to each
    training fold during cross-validation — never to the held-out test fold.
    This is the key methodological fix raised by peer reviewer #1 and
    described in the midterm response document.

    The Stacking Ensemble takes the *raw* (non-SMOTE) base estimators as its
    internal learners so that the meta-learner sees un-resampled probabilities,
    then the outer ImbPipeline applies SMOTE once before the whole ensemble.
    """

    # ── Raw estimators (no scaler/SMOTE yet) ─────────────────────────────
    raw = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced",   # second imbalance safeguard
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGB_AVAILABLE:
        raw["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,        # upweights the minority injury class
            eval_metric="auc",
            random_state=random_state,
            verbosity=0,
        )

    if LGB_AVAILABLE:
        raw["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )

    # Stacking needs at least two base learners
    if len(raw) >= 2:
        stack_base = [(name, clone(est)) for name, est in raw.items()]
        raw["Stacking Ensemble"] = StackingClassifier(
            estimators=stack_base,
            final_estimator=LogisticRegression(max_iter=1_000, C=0.5),
            cv=5,
            n_jobs=-1,
        )

    # ── Wrap every estimator in Scaler → SMOTE → Model ───────────────────
    pipelines = {}
    for name, estimator in raw.items():
        pipelines[name] = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=random_state, k_neighbors=5)),
            ("clf",    estimator),
        ])

    return pipelines


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4a — STRATIFIED CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def cross_validate_models(models: dict, X: pd.DataFrame, y: pd.Series,
                          n_splits: int = 10) -> pd.DataFrame:
    """
    10-Fold Stratified Cross-Validation — mirrors Jauhiainen et al.'s
    repeated CV approach.  Each fold preserves the injured/uninjured ratio
    (stratify=y).  AUC-ROC is reported as mean ± std across folds.

    Jauhiainen et al. used 100 repetitions of 5-fold CV; here we use
    10-fold (standard for datasets of this size) and report min/max to
    show the variance that a *single* CV run would hide — the key
    methodological lesson from Table 1 of their paper.
    """
    print(f"\n[CV] {n_splits}-Fold Stratified Cross-Validation "
          f"(SMOTE inside each fold — no leakage)...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    summary_rows = []

    for name, pipeline in models.items():
        scores = cross_val_score(
            pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        row = {
            "Model":    name,
            "Mean AUC": round(scores.mean(), 4),
            "Std AUC":  round(scores.std(),  4),
            "Min AUC":  round(scores.min(),  4),
            "Max AUC":  round(scores.max(),  4),
        }
        summary_rows.append(row)
        print(f"     {name:22s}  AUC = {scores.mean():.3f} ± {scores.std():.3f}"
              f"  (range {scores.min():.3f}–{scores.max():.3f})")

    return pd.DataFrame(summary_rows).sort_values("Mean AUC", ascending=False)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4b — HOLD-OUT EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_on_holdout(models: dict, X: pd.DataFrame, y: pd.Series,
                        test_size: float = 0.20):
    """
    Fit every pipeline on 80% of data, evaluate on a stratified 20% hold-out.
    Returns:
        results_df  — per-model metric table
        trained     — dict { name -> (fitted_pipeline, y_prob_test) }
        X_test, y_test
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    results, trained = [], {}
    for name, pipeline in models.items():
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]

        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, y_pred),               4),
            "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_te, y_pred),                  4),
            "F1":        round(f1_score(y_te, y_pred),                      4),
            "ROC-AUC":   round(roc_auc_score(y_te, y_prob),                 4),
            "Avg Prec":  round(average_precision_score(y_te, y_prob),       4),
        })
        trained[name] = (pipeline, y_prob)

    return pd.DataFrame(results), trained, X_te, y_te


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4c — VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_roc_and_pr_curves(trained: dict, y_test: pd.Series,
                           save_prefix: str = "output"):
    """
    Side-by-side ROC and Precision-Recall curves.
    The PR curve is especially important here because it shows the
    precision cost of achieving higher recall — directly addressing the
    peer reviewer's question about the 100% precision figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, (_, y_prob) in trained.items():
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, lw=2, label=f"{name}  AUC={auc:.3f}")

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, lw=2, label=f"{name}  AP={ap:.3f}")

    ax1.plot([0, 1], [0, 1], "k--", lw=1, color="grey")
    ax1.set_xlabel("False Positive Rate");  ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves — Multi-Dataset ACL Prediction")
    ax1.legend(loc="lower right", fontsize=8);  ax1.grid(alpha=0.3)

    baseline_pr = y_test.mean()
    ax2.axhline(baseline_pr, color="grey", linestyle="--", lw=1,
                label=f"Baseline (prevalence={baseline_pr:.2f})")
    ax2.set_xlabel("Recall");  ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves\n"
                  "(65% threshold trades some recall for higher precision)")
    ax2.legend(loc="upper right", fontsize=8);  ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{save_prefix}_roc_pr_curves.png"
    plt.savefig(path, dpi=150);  plt.show()
    print(f"[SAVED] {path}")


def plot_confusion_matrices(trained: dict, X_test: pd.DataFrame,
                            y_test: pd.Series, save_prefix: str = "output"):
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, (pipeline, _)) in zip(axes, trained.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["No Injury", "Injury"],
                    yticklabels=["No Injury", "Injury"])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted");  ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — 20% Hold-out Test Set", fontsize=11)
    plt.tight_layout()
    path = f"{save_prefix}_confusion_matrices.png"
    plt.savefig(path, dpi=150);  plt.show()
    print(f"[SAVED] {path}")


def plot_feature_importance(pipeline, feature_names: list,
                            top_n: int = 15, save_prefix: str = "output"):
    """
    Bar chart of Random Forest (or XGBoost/LightGBM) feature importances.
    The three engineered "Cutting Signature" features are coloured crimson
    to highlight that the biomechanical composites, not raw columns, drive
    the model — validating the feature engineering step to peer reviewers.
    """
    clf = pipeline.named_steps.get("clf")
    importances = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "named_estimators_"):
        # Stacking: pull from the first base estimator that exposes importances
        for est in clf.named_estimators_.values():
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                break

    if importances is None:
        print("[SKIP] Feature importance not available for this model type.")
        return

    # Guard against shape mismatch (SMOTE augments training rows, not columns)
    n = min(len(importances), len(feature_names))
    fi = pd.Series(importances[:n], index=feature_names[:n])
    fi = fi.sort_values(ascending=False).head(top_n)

    cutting_signature = {"Fatigue_Accumulation", "Lateral_Force_Proxy",
                         "Symmetry_Index"}
    colors = ["crimson" if f in cutting_signature else "steelblue"
              for f in fi.index]

    plt.figure(figsize=(11, 6))
    fi.plot(kind="bar", color=colors, edgecolor="black", width=0.7)
    plt.title('Feature Importance — "Cutting Signature" features in crimson\n'
              '(Fatigue_Accumulation, Lateral_Force_Proxy, Symmetry_Index)',
              fontsize=11)
    plt.ylabel("Relative Importance")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{save_prefix}_feature_importance.png"
    plt.savefig(path, dpi=150);  plt.show()
    print(f"[SAVED] {path}")


def plot_shap_summary(pipeline, X_test: pd.DataFrame,
                      save_prefix: str = "output"):
    """
    SHAP TreeExplainer summary plot.
    Uses the scaled-but-not-SMOTE X_test (SMOTE is training-only).
    If SHAP is unavailable the function exits gracefully.
    """
    if not SHAP_AVAILABLE:
        print("[SKIP] SHAP not installed — pip install shap")
        return

    clf     = pipeline.named_steps.get("clf")
    scaler  = pipeline.named_steps.get("scaler")
    X_scaled = scaler.transform(X_test)

    # For Stacking, use the first base estimator that supports TreeExplainer
    if hasattr(clf, "named_estimators_"):
        for est_name, est in clf.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                clf = est
                print(f"[SHAP] Using base estimator: {est_name}")
                break

    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_scaled)
        # Binary classifiers return a list [neg_class, pos_class]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv, X_scaled,
            feature_names=X_test.columns.tolist(),
            max_display=15,
            show=False,
        )
        plt.title("SHAP Summary — Contributions to ACL Injury Risk\n"
                  "(Cutting Signature features expected in top positions)",
                  fontsize=11)
        plt.tight_layout()
        path = f"{save_prefix}_shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight");  plt.show()
        print(f"[SAVED] {path}")
    except Exception as exc:
        print(f"[WARN] SHAP plot failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4d — FALSE NEGATIVE ERROR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def print_false_negative_report(pipeline, X_test: pd.DataFrame,
                                y_test: pd.Series):
    """
    Identify every athlete the model missed (False Negatives).

    Clinical framing (from peer reviewer Q4 / midterm response):
        False Positive  = unnecessary modified training session  (low cost)
        False Negative  = undetected ACL tear + 6-9 month rehab (high cost)

    Jauhiainen et al. (2022) found AUC ≈ 0.63 for their best model, meaning
    a non-trivial fraction of true injuries are inevitably missed with
    current screening variables.  This report makes that explicit.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    fn_mask = (y_test.values == 1) & (y_pred == 0)
    tp_mask = (y_test.values == 1) & (y_pred == 1)

    print(f"\n{'═'*62}")
    print("ERROR ANALYSIS — False Negatives (Missed ACL Risk Events)")
    print(f"{'═'*62}")
    print(f"  Total injuries in test set  : {int(y_test.sum())}")
    print(f"  Correctly flagged (TP)      : {tp_mask.sum()}")
    print(f"  Missed           (FN)       : {fn_mask.sum()}")
    print(f"  Recall                      : "
          f"{tp_mask.sum() / max(y_test.sum(), 1):.3f}")

    if fn_mask.sum() > 0:
        fn_df = X_test[fn_mask].copy()
        fn_df["Pred_Prob"] = y_prob[fn_mask]
        priority_cols = [
            "Fatigue_Accumulation", "Lateral_Force_Proxy", "Symmetry_Index",
            "workload", "fatigue", "speed", "impact", "prev_injury", "Pred_Prob"
        ]
        show_cols = [c for c in priority_cols if c in fn_df.columns]
        print(f"\n  Missed athlete profiles (top {min(8, fn_mask.sum())}):")
        print(fn_df[show_cols].head(8).to_string(index=False))
        print("\n  NOTE: Low Pred_Prob on missed cases suggests the model's\n"
              "  uncertainty zone (~0.40–0.65) is where interventions matter most.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — 65% EARLY WARNING DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

# Priority order for surfacing risk drivers in the Coach's Report
_RISK_PRIORITY = [
    "Fatigue_Accumulation", "Lateral_Force_Proxy", "Symmetry_Index",
    "impact", "workload", "fatigue", "speed", "prev_injury",
    "match_load", "recovery_days", "training_intensity",
]

_RISK_LABELS = {
    "Fatigue_Accumulation": "High Fatigue Accumulation",
    "Lateral_Force_Proxy":  "Extreme Lateral Cutting Load",
    "Symmetry_Index":       "Movement Asymmetry Detected",
    "impact":               "High Joint Impact / Knee Load",
    "workload":             "Excessive Weekly Workload",
    "fatigue":              "Elevated Fatigue Score",
    "speed":                "High Speed With Low Recovery",
    "prev_injury":          "Prior Injury History",
    "match_load":           "Dense Match Schedule",
    "recovery_days":        "Insufficient Recovery Time",
    "training_intensity":   "High Training Intensity",
}


def _top_risk_factors(row: pd.Series, top_n: int = 2) -> str:
    """Return the top-N feature labels for one athlete row."""
    available = [f for f in _RISK_PRIORITY if f in row.index]
    top = sorted(available, key=lambda f: float(row[f]), reverse=True)[:top_n]
    return "  +  ".join(_RISK_LABELS.get(f, f) for f in top)


def early_warning_dashboard(pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                             threshold: float = 0.65,
                             save_prefix: str = "output") -> pd.DataFrame:
    """
    Replicate and extend the 65% danger-threshold logic from DS340W code.py.

    Three risk tiers (colour-coded for the coaching staff):
        GREEN  [0.00–0.35)  →  Low risk, normal training
        AMBER  [0.35–0.65)  →  Moderate risk, monitor closely
        RED    [0.65–1.00]  →  High risk, Yellow-Flag protocol triggered

    The dashboard is intentionally COACH-FACING, not athlete-facing.
    Showing raw probabilities to athletes may introduce compensatory
    movement patterns that paradoxically increase injury risk
    (see Psychological Feedback Loop section in the midterm response doc).
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    report = X_test.reset_index(drop=True).copy()
    report["Athlete_ID"]      = [f"ATH-{i+1:04d}" for i in range(len(report))]
    report["Risk_Probability"] = y_prob
    report["Actual_Injury"]   = y_test.reset_index(drop=True).values

    report["Risk_Tier"] = pd.cut(
        report["Risk_Probability"],
        bins=[-0.001, 0.35, threshold, 1.001],
        labels=["GREEN  (Low)", "AMBER  (Moderate)", "RED    (High)"],
    )

    report["Top_Risk_Factors"] = report.apply(
        lambda row: _top_risk_factors(row, top_n=2), axis=1
    )

    flagged = report[report["Risk_Probability"] >= threshold].sort_values(
        "Risk_Probability", ascending=False
    )

    # ── Console output ────────────────────────────────────────────────────
    divider = "═" * 72
    print(f"\n{divider}")
    print("  AI EARLY WARNING DASHBOARD — COACH'S REPORT")
    print(f"  Risk Threshold : {threshold*100:.0f}%  (Yellow-Flag protocol)")
    print(f"  Athletes Screened: {len(report)}  |  Flagged RED: {len(flagged)}")
    print(divider)

    if flagged.empty:
        print("  [OK] All athletes are in the GREEN (Low Risk) zone.")
    else:
        display_cols = ["Athlete_ID", "Risk_Probability", "Risk_Tier",
                        "Top_Risk_Factors", "Actual_Injury"]
        print(flagged[display_cols].head(20).to_string(index=False))

        print(f"\n{'─'*72}")
        print("  COACH ACTION PROTOCOL  (for RED-flagged athletes)")
        print("  ① Reduce cutting-drill volume by 30-40% for this session.")
        print("  ② Prescribe neuromuscular warm-up focusing on knee alignment.")
        print("  ③ Schedule biomechanical reassessment within 48 hours.")
        print("  ④ Athlete stays on field — LOAD MODIFICATION, not benching.")
        print("  ⑤ Reassign athlete to low-impact conditioning during high-risk window.")

    # ── Tier distribution bar chart ───────────────────────────────────────
    tier_counts = report["Risk_Tier"].value_counts().sort_index()
    colors_bar  = ["#2ecc71", "#f39c12", "#e74c3c"]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(tier_counts.index, tier_counts.values,
                   color=colors_bar[:len(tier_counts)], edgecolor="black", width=0.5)
    plt.bar_label(bars, labels=[f"n={v}" for v in tier_counts.values], padding=3)
    plt.title("Athlete Risk Distribution — Early Warning Dashboard", fontsize=11)
    plt.ylabel("Number of Athletes")
    plt.ylim(0, tier_counts.max() * 1.15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{save_prefix}_risk_distribution.png"
    plt.savefig(path, dpi=150);  plt.show()
    print(f"\n[SAVED] {path}")

    # ── Save full report CSV ──────────────────────────────────────────────
    csv_path = f"{save_prefix}_coach_risk_report.csv"
    report.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    return report


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main(output_prefix: str = "acl_pipeline"):
    """
    End-to-end pipeline:
        Download → Preprocess → Engineer → Cross-Validate →
        Hold-out Eval → Visualise → Dashboard
    """
    print("═" * 72)
    print("  DS 340W — ACL Injury Prediction: Production Pipeline")
    print("  Jauhiainen et al. (2022)  |  Three-Dataset Fusion")
    print("═" * 72)

    # ── 1. Load uploaded CSVs ─────────────────────────────────────────────
    dataframes = load_uploaded_datasets()

    # ── 2. Preprocess & engineer ──────────────────────────────────────────
    X, y = preprocess_data(dataframes)
    feature_names = X.columns.tolist()

    # ── 3. Build model zoo ────────────────────────────────────────────────
    models = build_model_zoo()
    model_names = list(models.keys())
    print(f"\n[MODELS] Evaluating: {model_names}")

    # ── 4a. 10-Fold stratified cross-validation ───────────────────────────
    cv_df = cross_validate_models(models, X, y, n_splits=10)
    print(f"\n[CV SUMMARY]\n{cv_df.to_string(index=False)}")
    cv_df.to_csv(f"{output_prefix}_cv_summary.csv", index=False)
    print(f"[SAVED] {output_prefix}_cv_summary.csv")

    # ── 4b. Hold-out evaluation ───────────────────────────────────────────
    print("\n[HOLDOUT] Fitting on 80%, evaluating on 20% hold-out...")
    results_df, trained, X_test, y_test = evaluate_on_holdout(models, X, y)
    print(f"\n[HOLDOUT METRICS]\n{results_df.to_string(index=False)}")
    results_df.to_csv(f"{output_prefix}_holdout_metrics.csv", index=False)
    print(f"[SAVED] {output_prefix}_holdout_metrics.csv")

    # ── 4c. Pick best model (by CV AUC — not hold-out, to avoid p-hacking) ─
    best_name     = cv_df.iloc[0]["Model"]
    best_pipeline = trained[best_name][0]
    print(f"\n[BEST MODEL] {best_name}  "
          f"(CV AUC={cv_df.iloc[0]['Mean AUC']:.3f} ± {cv_df.iloc[0]['Std AUC']:.3f})")

    # ── 4d. Visualisations ────────────────────────────────────────────────
    plot_roc_and_pr_curves(trained, y_test,            save_prefix=output_prefix)
    plot_confusion_matrices(trained, X_test, y_test,   save_prefix=output_prefix)
    plot_feature_importance(best_pipeline, feature_names, save_prefix=output_prefix)
    plot_shap_summary(best_pipeline, X_test,           save_prefix=output_prefix)

    # ── 4e. False-negative error analysis ────────────────────────────────
    print_false_negative_report(best_pipeline, X_test, y_test)

    # ── 5. Early warning dashboard ────────────────────────────────────────
    early_warning_dashboard(
        best_pipeline, X_test, y_test,
        threshold=0.65,
        save_prefix=output_prefix,
    )

    print(f"\n{'═'*72}")
    print("  Pipeline complete.  All outputs saved with prefix: "
          f"'{output_prefix}_*'")
    print("═" * 72)


if __name__ == "__main__":
    main()
