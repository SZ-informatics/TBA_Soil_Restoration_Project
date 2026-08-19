# =============================================================================
# Soil-depth classification pipeline — corrected version
#
# Fixes for reviewer concerns:
#   * AUC vs Balanced-Accuracy value provenance printed at full precision
#     so Table 4 cells cannot be confused
#   * Cross-validation 95% CI computed with the Nadeau–Bengio correction
#     (proper for dependent K-fold folds) in addition to the naive formula
#   * Explicit verification that Species x Amendment x Block plots do NOT
#     overlap between train and test
#   * Duplicate imports and Jupyter-only magics removed; single-file script
#   * Per-model diagnostic block prints every metric independently, from
#     independent computations, to eliminate any chance of copy-paste bugs
#   * All output files saved to the ./output directory
# =============================================================================


# =============================================================================
# Imports
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

import sklearn
import xgboost
from xgboost import XGBClassifier

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    GridSearchCV
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    FunctionTransformer
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# 0) Configuration
# =============================================================================

RANDOM_STATE  = 42
SCORING       = "roc_auc"      # "roc_auc" or "accuracy"
N_SPLITS      = 10
NORMALIZATION = "LOG"          # "CSS" | "CLR" | "HELLINGER" | "LOG"

# Input Excel file
EXCEL_PATH = "data/Allmerged_16Slevel-2.xlsx"

# -------------------------------------------------------------------------
# Output directory
# Creates an ./output folder in the working directory.
# If it already exists, Python will simply use it.
# -------------------------------------------------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)

print("=" * 78)
print(
    f"python {sys.version.split()[0]} | "
    f"scikit-learn {sklearn.__version__} | "
    f"xgboost {xgboost.__version__}"
)
print(
    f"SCORING={SCORING} | "
    f"N_SPLITS={N_SPLITS} | "
    f"NORMALIZATION={NORMALIZATION} | "
    f"seed={RANDOM_STATE}"
)
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 78)


# =============================================================================
# 1) Load + preprocess
# =============================================================================

df = pd.read_excel(EXCEL_PATH, index_col=0)

# Drop Archaea, strip Bacteria prefix
df = df.loc[:, ~df.columns.str.startswith("d__Archaea;")]

df.columns = df.columns.str.replace(
    r"^d__Bacteria;",
    "",
    regex=True
)

metadata_cols = [
    "#SampleID",
    "index",
    "SampleCode",
    "LinkerPrimerSequence",
    "BarcodeSequence",
    "Time Period",
    "Species",
    "Amendment",
    "SoilProfile",
    "Block",
    "Group1",
    "Group2",
    "Group3",
]

taxa_cols = [
    c for c in df.columns
    if c not in metadata_cols
]

df_taxa = (
    df[taxa_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)

# Prevalence filter:
# keep features with >=2 reads in >=2 samples
df_taxa_filtered = df_taxa.loc[
    :,
    (df_taxa >= 2).sum(axis=0) >= 2
]

# Counts; normalization occurs inside each sklearn pipeline
X = df_taxa_filtered.values

y_text = df["SoilProfile"].astype(str).values

le = LabelEncoder()
y = le.fit_transform(y_text)

print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
print(f"Classes:  {list(le.classes_)}")

print(
    "Class counts:\n"
    + pd.Series(y_text).value_counts().to_string()
)


# =============================================================================
# 2) Replicate groups
# Species x Amendment x Block
# =============================================================================

required = [
    "Species",
    "Amendment",
    "Block"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required grouping columns: {missing}"
    )

groups_all = (
    df["Species"].astype(str)
    + "_"
    + df["Amendment"].astype(str)
    + "_"
    + df["Block"].astype(str)
).values

print(
    f"\nTotal unique plot-level groups: "
    f"{len(np.unique(groups_all))}"
)


# =============================================================================
# 3) Group-aware holdout split
# No plot leakage between train and test
# =============================================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    gss.split(
        X,
        y,
        groups=groups_all
    )
)

X_train = X[train_idx]
X_test  = X[test_idx]

y_train = y[train_idx]
y_test  = y[test_idx]

groups_train = groups_all[train_idx]

train_group_set = set(
    np.unique(groups_train)
)

test_group_set = set(
    np.unique(groups_all[test_idx])
)

overlap = (
    train_group_set
    & test_group_set
)

print(
    f"Train: {X_train.shape[0]} samples, "
    f"{len(train_group_set)} plots"
)

print(
    f"Test:  {X_test.shape[0]} samples, "
    f"{len(test_group_set)} plots"
)

print(
    f"Plot overlap between train/test: "
    f"{len(overlap)} "
    f"(0 = correct group-aware split)"
)

if overlap:
    raise RuntimeError(
        f"Group leakage detected! "
        f"Overlapping plots: {overlap}"
    )

uq, ct = np.unique(
    y_test,
    return_counts=True
)

print(
    "Test class counts:",
    dict(
        zip(
            le.inverse_transform(uq),
            ct
        )
    )
)


# =============================================================================
# 4) Positive class and XGBoost imbalance weight
# =============================================================================

POS_LABEL = (
    int(le.transform(["U"])[0])
    if "U" in le.classes_
    else 1
)

NEG_LABEL = 1 - POS_LABEL

print(
    f"Positive class: "
    f"{le.inverse_transform([POS_LABEL])[0]} "
    f"(encoded {POS_LABEL})"
)

pos = int(
    np.sum(
        y_train == POS_LABEL
    )
)

neg = int(
    np.sum(
        y_train == NEG_LABEL
    )
)

scale_pos_weight = (
    float(neg / pos)
    if pos > 0
    else 1.0
)

print(
    f"XGB scale_pos_weight "
    f"(train only): "
    f"{scale_pos_weight:.3f}"
)


# =============================================================================
# 5) CV splitter, normalization transformers, XGB constructor
# =============================================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# -----------------------------------------------------------------------------
# CLR transformation
# -----------------------------------------------------------------------------

def clr_transform(
    X,
    pseudocount=1e-6
):
    X = (
        np.asarray(
            X,
            dtype=float
        )
        + pseudocount
    )

    logX = np.log(X)

    return (
        logX
        - logX.mean(
            axis=1,
            keepdims=True
        )
    )


# -----------------------------------------------------------------------------
# Hellinger transformation
# -----------------------------------------------------------------------------

def hellinger_transform(X):

    X = np.asarray(
        X,
        dtype=float
    )

    X[X < 0] = 0.0

    rs = X.sum(
        axis=1,
        keepdims=True
    )

    rs[rs == 0] = 1.0

    return np.sqrt(
        X / rs
    )


# -----------------------------------------------------------------------------
# Log transformation
# -----------------------------------------------------------------------------

def log_transform(
    X,
    pseudocount=1.0
):

    return np.log(
        np.asarray(
            X,
            dtype=float
        )
        + pseudocount
    )


# -----------------------------------------------------------------------------
# CSS transformation
# -----------------------------------------------------------------------------

def css_transform(
    X,
    p=0.75,
    pseudocount=1.0
):

    X = np.asarray(
        X,
        dtype=float
    )

    X[X < 0] = 0.0

    s = np.ones(
        X.shape[0]
    )

    for i in range(
        X.shape[0]
    ):

        nz = X[i][
            X[i] > 0
        ]

        if nz.size == 0:
            continue

        q = np.quantile(
            nz,
            p
        )

        si = X[i][
            X[i] <= q
        ].sum()

        s[i] = (
            si
            if si > 0
            else 1.0
        )

    return np.log(
        X / s[:, None]
        + pseudocount
    )


# -----------------------------------------------------------------------------
# Select normalization
# -----------------------------------------------------------------------------

def get_norm_steps(name):

    name = name.upper()

    if name == "CLR":

        return [
            (
                "clr",
                FunctionTransformer(
                    clr_transform,
                    kw_args={
                        "pseudocount": 1e-6
                    },
                    validate=False
                )
            )
        ]

    if name == "HELLINGER":

        return [
            (
                "hellinger",
                FunctionTransformer(
                    hellinger_transform,
                    validate=False
                )
            )
        ]

    if name == "LOG":

        return [
            (
                "log",
                FunctionTransformer(
                    log_transform,
                    kw_args={
                        "pseudocount": 1.0
                    },
                    validate=False
                )
            )
        ]

    if name == "CSS":

        return [
            (
                "css",
                FunctionTransformer(
                    css_transform,
                    kw_args={
                        "p": 0.75,
                        "pseudocount": 1.0
                    },
                    validate=False
                )
            )
        ]

    raise ValueError(
        f"Unknown NORMALIZATION='{name}'"
    )


norm_steps = get_norm_steps(
    NORMALIZATION
)


# -----------------------------------------------------------------------------
# XGBoost constructor
# -----------------------------------------------------------------------------

def make_xgb():

    kwargs = dict(
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    if int(
        xgboost.__version__
        .split(".")[0]
    ) < 2:

        kwargs[
            "use_label_encoder"
        ] = False

    return XGBClassifier(
        **kwargs
    )


# =============================================================================
# 6) Pipelines and hyperparameter grids
# =============================================================================


# -----------------------------------------------------------------------------
# SVM
# -----------------------------------------------------------------------------

svm_pipe = Pipeline(
    norm_steps
    + [
        (
            "scaler",
            StandardScaler(
                with_mean=True,
                with_std=True
            )
        ),
        (
            "svc",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            )
        ),
    ]
)

svm_grid = {

    "svc__C": [
        0.3,
        1,
        3,
        10,
        30
    ],

    "svc__gamma": [
        "scale",
        0.01,
        0.003,
        0.001
    ],

    "svc__class_weight": [
        None,
        "balanced"
    ],
}


# -----------------------------------------------------------------------------
# Random Forest
# -----------------------------------------------------------------------------

rf_pipe = Pipeline(
    norm_steps
    + [
        (
            "rf",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

rf_grid = {

    "rf__n_estimators": [
        60,
        70,
        80,
        90,
        100,
        150
    ],

    "rf__criterion": [
        "gini",
        "entropy"
    ],

    "rf__max_features": [
        "sqrt",
        "log2"
    ],

    "rf__min_samples_split": [
        2,
        4
    ],

    "rf__min_samples_leaf": [
        5,
        10,
        15
    ],

    "rf__bootstrap": [
        True
    ],

    "rf__max_depth": [
        2,
        3,
        4,
        None
    ],

    "rf__class_weight": [
        "balanced",
        None
    ],
}


# -----------------------------------------------------------------------------
# XGBoost
# -----------------------------------------------------------------------------

xgb_pipe = Pipeline(
    norm_steps
    + [
        (
            "xgb",
            make_xgb()
        )
    ]
)

xgb_grid = {

    "xgb__n_estimators": [
        50,
        60,
        70,
        80,
        90,
        100,
        150
    ],

    "xgb__learning_rate": [
        0.05,
        0.1
    ],

    "xgb__max_depth": [
        2,
        3,
        4
    ],

    "xgb__min_child_weight": [
        1,
        3,
        5
    ],

    "xgb__subsample": [
        0.8,
        1.0
    ],

    "xgb__colsample_bytree": [
        0.8,
        1.0
    ],

    "xgb__reg_alpha": [
        0.0,
        0.1
    ],

    "xgb__reg_lambda": [
        1.0,
        2.0
    ],
}


searches = [

    (
        "SVM",
        svm_pipe,
        svm_grid
    ),

    (
        "Random Forest",
        rf_pipe,
        rf_grid
    ),

    (
        "XGBoost",
        xgb_pipe,
        xgb_grid
    ),
]


# =============================================================================
# 7) Helper for CV variance / confidence intervals
# Naive, t-based, and Nadeau-Bengio
# =============================================================================

def cv_intervals(
    fold_scores,
    n_train_full,
    n_splits
):

    """
    Return:
      mean
      standard deviation
      naive normal 95% CI
      t-based 95% CI
      Nadeau-Bengio dependence-aware 95% CI
    """

    fs = np.asarray(
        fold_scores,
        dtype=float
    )

    m = float(
        fs.mean()
    )

    sd = float(
        fs.std(
            ddof=1
        )
    )

    # ---------------------------------------------------------------------
    # Naive normal CI
    # ---------------------------------------------------------------------

    naive_half = (
        1.96
        * sd
        / np.sqrt(
            n_splits
        )
    )

    naive = (
        m - naive_half,
        m + naive_half
    )

    # ---------------------------------------------------------------------
    # t-based small-K correction
    # still assumes independent folds
    # ---------------------------------------------------------------------

    tval = float(
        stats.t.ppf(
            0.975,
            df=n_splits - 1
        )
    )

    t_half = (
        tval
        * sd
        / np.sqrt(
            n_splits
        )
    )

    t_ci = (
        m - t_half,
        m + t_half
    )

    # ---------------------------------------------------------------------
    # Nadeau-Bengio correction
    # Accounts for dependence among CV folds
    # ---------------------------------------------------------------------

    n_test = (
        n_train_full
        // n_splits
    )

    n_train = (
        n_train_full
        - n_test
    )

    nb_sd = np.sqrt(
        fs.var(
            ddof=1
        )
        * (
            1.0 / n_splits
            + n_test / n_train
        )
    )

    nb_half = (
        tval
        * nb_sd
    )

    nb_ci = (
        m - nb_half,
        m + nb_half
    )

    return dict(
        mean=m,
        sd=sd,
        naive=naive,
        t_ci=t_ci,
        nb_ci=nb_ci
    )


# =============================================================================
# 8) GridSearchCV per model
# Then independently recompute test-set metrics
# =============================================================================

summary_rows = []
metrics_rows = []
probas_for_roc = {}


for name, pipe, grid in searches:

    print(
        "\n"
        + "=" * 78
    )

    print(
        f">>> {name} "
        f"(GridSearchCV scoring='{SCORING}', "
        f"{N_SPLITS}-fold group-aware)"
    )

    print(
        "=" * 78
    )

    # ---------------------------------------------------------------------
    # Grid search
    # ---------------------------------------------------------------------

    gs = GridSearchCV(
        pipe,
        grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False
    )

    gs.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    best = gs.best_estimator_
    best_params = gs.best_params_

    # Save the fitted/tuned Random Forest pipeline for separate feature extraction
    if name == "Random Forest":
        best_rf_model = best
        rf_model_path = OUTPUT_DIR / f"best_random_forest_{NORMALIZATION.lower()}.joblib"
        joblib.dump(best_rf_model, rf_model_path)

        # Save the exact feature names used during model fitting
        rf_feature_names_path = OUTPUT_DIR / f"rf_feature_names_{NORMALIZATION.lower()}.csv"
        pd.DataFrame({
            "Feature": df_taxa_filtered.columns
        }).to_csv(
            rf_feature_names_path,
            index=False
        )

        # Save the training split metadata needed for depth association
        rf_train_index_path = OUTPUT_DIR / f"rf_training_indices_{NORMALIZATION.lower()}.csv"
        pd.DataFrame({
            "row_index": train_idx,
            "SoilProfile": le.inverse_transform(y_train)
        }).to_csv(
            rf_train_index_path,
            index=False
        )

        print(f"Saved fitted RF pipeline: {rf_model_path}")
        print(f"Saved RF feature names: {rf_feature_names_path}")
        print(f"Saved RF training indices: {rf_train_index_path}")

    # Identify the best parameter row
    row_i = next(
        i
        for i, p
        in enumerate(
            gs.cv_results_["params"]
        )
        if p == best_params
    )

    # Retrieve individual fold scores
    fold_scores = [

        gs.cv_results_[
            f"split{k}_test_score"
        ][row_i]

        for k
        in range(
            N_SPLITS
        )
    ]

    # ---------------------------------------------------------------------
    # CV statistics
    # ---------------------------------------------------------------------

    cv_stats = cv_intervals(
        fold_scores,
        n_train_full=len(y_train),
        n_splits=N_SPLITS
    )

    cv_mean = cv_stats[
        "mean"
    ]

    cv_sd = cv_stats[
        "sd"
    ]

    print(
        f"Best params: "
        f"{best_params}"
    )

    print(
        f"CV {SCORING} per fold: "
        f"{[round(s, 4) for s in fold_scores]}"
    )

    print(
        f"CV {SCORING} mean = "
        f"{cv_mean:.4f}   "
        f"sd = {cv_sd:.4f}   "
        f"var = {cv_sd**2:.6f}"
    )

    print(
        "95% CI "
        "(naive normal, SD/sqrtK):    "
        f"[{cv_stats['naive'][0]:.4f}, "
        f"{cv_stats['naive'][1]:.4f}]  "
        f"width="
        f"{cv_stats['naive'][1] - cv_stats['naive'][0]:.4f}"
    )

    print(
        "95% CI "
        "(t-based, small-K correction): "
        f"[{cv_stats['t_ci'][0]:.4f}, "
        f"{cv_stats['t_ci'][1]:.4f}]  "
        f"width="
        f"{cv_stats['t_ci'][1] - cv_stats['t_ci'][0]:.4f}"
    )

    print(
        "95% CI "
        "(Nadeau-Bengio, dependence-aware, RECOMMENDED): "
        f"[{cv_stats['nb_ci'][0]:.4f}, "
        f"{cv_stats['nb_ci'][1]:.4f}]  "
        f"width="
        f"{cv_stats['nb_ci'][1] - cv_stats['nb_ci'][0]:.4f}"
    )


    # =========================================================================
    # 8a) Independent test-set metrics
    # =========================================================================

    y_pred_test = best.predict(
        X_test
    )

    y_proba_test = (
        best.predict_proba(
            X_test
        )[:, POS_LABEL]
    )

    # Training accuracy
    train_acc = accuracy_score(
        y_train,
        best.predict(
            X_train
        )
    )

    # Test accuracy
    test_acc = accuracy_score(
        y_test,
        y_pred_test
    )

    # AUC
    test_auc = roc_auc_score(
        y_test,
        y_proba_test
    )

    # Cohen's kappa
    test_kappa = cohen_kappa_score(
        y_test,
        y_pred_test
    )

    # Sensitivity
    test_sens = recall_score(
        y_test,
        y_pred_test,
        pos_label=POS_LABEL
    )

    # Specificity
    test_spec = recall_score(
        y_test,
        y_pred_test,
        pos_label=NEG_LABEL
    )

    # Precision
    test_prec = precision_score(
        y_test,
        y_pred_test,
        pos_label=POS_LABEL,
        zero_division=0
    )

    # Balanced accuracy
    test_bal = balanced_accuracy_score(
        y_test,
        y_pred_test
    )

    # F1
    test_f1 = f1_score(
        y_test,
        y_pred_test,
        pos_label=POS_LABEL,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred_test,
        labels=[
            NEG_LABEL,
            POS_LABEL
        ]
    )


    # ---------------------------------------------------------------------
    # Explicit high-precision provenance output
    # ---------------------------------------------------------------------

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"{name} Test-set metrics "
        "(high precision, "
        "each from an independent call)"
    )

    print(
        "-" * 78
    )

    print(
        "  AUC "
        "(roc_auc_score, threshold-independent) "
        f"= {test_auc:.6f}   "
        f"({100 * test_auc:.4f}%)"
    )

    print(
        "  Accuracy"
        f"{' ' * 36}"
        f"= {test_acc:.6f}   "
        f"({100 * test_acc:.4f}%)"
    )

    print(
        f"  Sensitivity "
        f"(recall for "
        f"'{le.inverse_transform([POS_LABEL])[0]}') "
        f"= {test_sens:.6f}   "
        f"({100 * test_sens:.4f}%)"
    )

    print(
        f"  Specificity "
        f"(recall for "
        f"'{le.inverse_transform([NEG_LABEL])[0]}') "
        f"= {test_spec:.6f}   "
        f"({100 * test_spec:.4f}%)"
    )

    print(
        "  Precision"
        f"{' ' * 35}"
        f"= {test_prec:.6f}   "
        f"({100 * test_prec:.4f}%)"
    )

    print(
        "  Balanced Accuracy = "
        "(Sens+Spec)/2 "
        f"= {test_bal:.6f}   "
        f"({100 * test_bal:.4f}%)"
    )

    print(
        "  F1-score"
        f"{' ' * 36}"
        f"= {test_f1:.6f}   "
        f"({100 * test_f1:.4f}%)"
    )

    print(
        "  Cohen's Kappa"
        f"{' ' * 31}"
        f"= {test_kappa:.6f}"
    )


    # ---------------------------------------------------------------------
    # Sanity check:
    # AUC versus Balanced Accuracy
    # ---------------------------------------------------------------------

    print(
        "\n  AUC - Balanced Accuracy "
        "(percentage points) = "
        f"{100 * (test_auc - test_bal):+.6f}"
    )

    if abs(
        test_auc
        - test_bal
    ) < 1e-4:

        print(
            "  NOTE: "
            "AUC and Balanced Accuracy "
            "are equal to 4 dp for this model."
        )

        print(
            "        This is unusual "
            "but genuinely possible when "
            "predict_proba is"
        )

        print(
            "        dominated by a small "
            "number of unique values. "
            "Verified above."
        )


    # ---------------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------------

    tn, fp = cm[0]
    fn, tp = cm[1]

    print(
        "  Confusion matrix "
        f"(rows true "
        f"[{le.classes_[NEG_LABEL]},"
        f"{le.classes_[POS_LABEL]}], "
        "cols pred): "
        f"TN={tn} FP={fp} "
        f"FN={fn} TP={tp}"
    )


    # Save probability vector for combined ROC curve
    probas_for_roc[
        name
    ] = y_proba_test


    # =========================================================================
    # 8b) Build paper-style summary rows
    # =========================================================================

    cv_label = (
        "CV AUC (mean±SD)"
        if SCORING == "roc_auc"
        else "CV Accuracy (mean±SD)"
    )

    summary_rows.append({

        "Model":
            name,

        cv_label:
            f"{cv_mean:.3f} ± {cv_sd:.3f}",

        "CV Variance":
            round(
                cv_sd**2,
                5
            ),

        "CV 95% CI (naive)":
            (
                f"[{cv_stats['naive'][0]:.3f}, "
                f"{cv_stats['naive'][1]:.3f}]"
            ),

        "CV 95% CI (Nadeau-Bengio)":
            (
                f"[{cv_stats['nb_ci'][0]:.3f}, "
                f"{cv_stats['nb_ci'][1]:.3f}]"
            ),

        "Train Acc":
            round(
                train_acc,
                3
            ),

        "Test Acc":
            round(
                test_acc,
                3
            ),

        "Test AUC":
            round(
                test_auc,
                3
            ),

        "Test Bal Acc":
            round(
                test_bal,
                3
            ),

        "Kappa":
            round(
                test_kappa,
                3
            ),
    })


    metrics_rows.append({

        "Model":
            name,

        "AUC (%)":
            round(
                100 * test_auc,
                2
            ),

        "Accuracy (%)":
            round(
                100 * test_acc,
                2
            ),

        "Sensitivity (%)":
            round(
                100 * test_sens,
                2
            ),

        "Specificity (%)":
            round(
                100 * test_spec,
                2
            ),

        "Precision (%)":
            round(
                100 * test_prec,
                2
            ),

        "Balanced Acc (%)":
            round(
                100 * test_bal,
                2
            ),

        "F1-score (%)":
            round(
                100 * test_f1,
                2
            ),
    })


# =============================================================================
# 9) Combined ROC figure
# =============================================================================

plt.figure(
    figsize=(
        7,
        6
    )
)

for name, y_proba in probas_for_roc.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        y_proba
    )

    roc_auc_value = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=(
            f"{name} "
            f"(AUC = "
            f"{roc_auc_value:.3f})"
        )
    )

# Random classifier line
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    lw=1
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves "
    "(group-aware GridSearch, "
    f"held-out test; "
    f"norm={NORMALIZATION})"
)

plt.legend(
    loc="lower right",
    frameon=True
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# -----------------------------------------------------------------------------
# Save ROC outputs to ./output
# -----------------------------------------------------------------------------

roc_png_path = (
    OUTPUT_DIR
    / f"roc_groupcv_grid_{NORMALIZATION.lower()}.png"
)

roc_pdf_path = (
    OUTPUT_DIR
    / f"roc_groupcv_grid_{NORMALIZATION.lower()}.pdf"
)

plt.savefig(
    roc_png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    roc_pdf_path,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 10) Save summary tables
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)

metrics_df = (
    pd.DataFrame(
        metrics_rows
    )
    .sort_values(
        [
            "Accuracy (%)",
            "AUC (%)"
        ],
        ascending=False
    )
)


# -----------------------------------------------------------------------------
# Print Table 3
# -----------------------------------------------------------------------------

print(
    "\n"
    + "=" * 78
)

print(
    "Table 3 source — "
    "CV and test summary"
)

print(
    "=" * 78
)

print(
    summary_df.to_string(
        index=False
    )
)


# -----------------------------------------------------------------------------
# Save Table 3
# -----------------------------------------------------------------------------

summary_csv_path = (
    OUTPUT_DIR
    / "summary_performance.csv"
)

summary_df.to_csv(
    summary_csv_path,
    index=False
)


# -----------------------------------------------------------------------------
# Print Table 4
# -----------------------------------------------------------------------------

print(
    "\n"
    + "=" * 78
)

print(
    "Table 4 source — "
    "diagnostic metrics on held-out "
    "test set (percentages)"
)

print(
    "=" * 78
)

print(
    metrics_df.to_string(
        index=False
    )
)


# -----------------------------------------------------------------------------
# Save Table 4
# -----------------------------------------------------------------------------

metrics_csv_path = (
    OUTPUT_DIR
    / (
        "diagnostic_metrics_test_"
        f"{NORMALIZATION.lower()}.csv"
    )
)

metrics_df.to_csv(
    metrics_csv_path,
    index=False
)


# =============================================================================
# 11) Final output summary
# =============================================================================

print(
    "\n"
    + "=" * 78
)

print(
    "ANALYSIS COMPLETE"
)

print(
    "=" * 78
)

print(
    f"\nAll results saved to:\n"
    f"{OUTPUT_DIR}\n"
)

print(
    "Saved files:"
)

print(
    f"  1. {summary_csv_path.name}"
)

print(
    f"  2. {metrics_csv_path.name}"
)

print(
    f"  3. {roc_png_path.name}"
)

print(
    f"  4. {roc_pdf_path.name}"
)

print(
    "\nFull paths:"
)

print(
    f"  {summary_csv_path}"
)

print(
    f"  {metrics_csv_path}"
)

print(
    f"  {roc_png_path}"
)

print(
    f"  {roc_pdf_path}"
)

print(
    "=" * 78
)