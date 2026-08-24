#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Group-aware machine-learning analysis for soil-depth classification.

Primary analysis
----------------
LOG10_RA:
    Counts -> per-sample relative abundance -> log10(RA + 1e-6)

Sensitivity analysis
--------------------
CLR:
    Counts -> add small pseudocount -> centered log-ratio transformation

The CLR analysis is included as a sensitivity analysis to determine whether
the soil-depth classification signal and model-performance patterns depend
strongly on transformation strategy.

Experimental grouping
---------------------
Samples are grouped by:
    Species x Amendment x Block

This combination represents unique field plots. Samples originating from the
same field plot, including paired soil depths and repeated sampling events,
are kept together during both train/test splitting and cross-validation to
prevent plot-level information leakage.

Models
------
- Random Forest
- XGBoost
- Support Vector Machine (RBF kernel)

Cross-validation
----------------
Hyperparameters are tuned using 10-fold StratifiedGroupKFold cross-validation.

Cross-validation AUC uncertainty is summarized using the Nadeau-Bengio
dependence-aware variance correction. The correction uses the mean
validation/training ratio from the actual realized StratifiedGroupKFold
splits.

This implementation yields the same rounded 95% confidence intervals reported
in the manuscript as the original K-fold approximation.

Outputs
-------
- Best hyperparameters
- Full GridSearchCV results
- Cross-validation AUC mean ± SD
- Nadeau-Bengio-corrected 95% CI
- Training accuracy
- Held-out test accuracy
- Held-out AUC
- Cohen's Kappa
- Sensitivity
- Specificity
- Precision
- Balanced accuracy
- F1-score
- ROC curve
- Fitted Random Forest model
- Random Forest feature names
- Train/test indices
- CV fold structure
"""


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
    GridSearchCV,
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    FunctionTransformer,
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
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

RANDOM_STATE = 42

SCORING = "roc_auc"

N_SPLITS = 10

PSEUDOCOUNT = 1e-6


# -----------------------------------------------------------------------------
# Transformation selection
# -----------------------------------------------------------------------------
#
# "LOG10_RA"
#     PRIMARY analysis reported in the manuscript:
#
#         counts
#           -> per-sample relative abundance
#           -> log10(relative abundance + 1e-6)
#
#
# "CLR"
#     SENSITIVITY analysis:
#
#         counts
#           -> add pseudocount
#           -> centered log-ratio transformation
#
# To reproduce the primary manuscript analysis:
#
#     NORMALIZATION = "LOG10_RA"
#
# To reproduce the CLR sensitivity analysis:
#
#     NORMALIZATION = "CLR"
# -----------------------------------------------------------------------------

NORMALIZATION = "LOG10_RA"


# -----------------------------------------------------------------------------
# Input file
# -----------------------------------------------------------------------------

EXCEL_PATH = "Allmerged_16Slevel-2.xlsx"


# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------
#
# Results are written to:
#
#     ./output/log10_ra/
#
# or:
#
#     ./output/clr/
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    Path("output")
    / NORMALIZATION.lower()
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


np.random.seed(
    RANDOM_STATE
)


print(
    "=" * 80
)

print(
    f"Python {sys.version.split()[0]} | "
    f"scikit-learn {sklearn.__version__} | "
    f"xgboost {xgboost.__version__}"
)

print(
    f"SCORING={SCORING} | "
    f"N_SPLITS={N_SPLITS} | "
    f"NORMALIZATION={NORMALIZATION} | "
    f"seed={RANDOM_STATE}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    "=" * 80
)


# =============================================================================
# 1) Load and preprocess data
# =============================================================================

df = pd.read_excel(
    EXCEL_PATH,
    index_col=0,
)


# -----------------------------------------------------------------------------
# Remove archaeal features
# -----------------------------------------------------------------------------

df = df.loc[
    :,
    ~df.columns.str.startswith(
        "d__Archaea;"
    ),
]


# -----------------------------------------------------------------------------
# Remove bacterial-domain prefix from taxonomy labels
# -----------------------------------------------------------------------------

df.columns = df.columns.str.replace(
    r"^d__Bacteria;",
    "",
    regex=True,
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
    c
    for c in df.columns
    if c not in metadata_cols
]


df_taxa = (
    df[taxa_cols]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .fillna(0)
)


# =============================================================================
# 2) Feature prevalence filtering
# =============================================================================
#
# Retain bacterial features with:
#
#     >= 2 reads in >= 2 samples
# =============================================================================

df_taxa_filtered = df_taxa.loc[
    :,
    (df_taxa >= 2).sum(axis=0) >= 2,
]


X = (
    df_taxa_filtered
    .values
)


y_text = (
    df["SoilProfile"]
    .astype(str)
    .values
)


le = LabelEncoder()

y = le.fit_transform(
    y_text
)


print(
    f"\nSamples:  {X.shape[0]}"
)

print(
    f"Features: {X.shape[1]}"
)

print(
    f"Classes:  {list(le.classes_)}"
)


print(
    "\nClass counts:"
)

print(
    pd.Series(
        y_text
    )
    .value_counts()
    .to_string()
)


# =============================================================================
# 3) Define field-plot groups
# =============================================================================
#
# Unique field plot:
#
#     Species x Amendment x Block
#
# Samples from the same plot are never allowed to be split across:
#
#     - training and held-out test data
#     - training and validation folds during CV
# =============================================================================

required_group_cols = [
    "Species",
    "Amendment",
    "Block",
]


missing = [
    c
    for c in required_group_cols
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
    "\nTotal unique plot-level groups:",
    len(
        np.unique(
            groups_all
        )
    ),
)


# =============================================================================
# 4) Group-aware held-out train/test split
# =============================================================================
#
# 70:30 split performed at the plot level.
# =============================================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE,
)


train_idx, test_idx = next(
    gss.split(
        X,
        y,
        groups=groups_all,
    )
)


X_train = X[
    train_idx
]

X_test = X[
    test_idx
]


y_train = y[
    train_idx
]

y_test = y[
    test_idx
]


groups_train = groups_all[
    train_idx
]

groups_test = groups_all[
    test_idx
]


train_group_set = set(
    np.unique(
        groups_train
    )
)


test_group_set = set(
    np.unique(
        groups_test
    )
)


overlap = (
    train_group_set
    & test_group_set
)


print(
    "\nHeld-out split"
)

print(
    "-" * 80
)

print(
    f"Training samples: {len(y_train)}"
)

print(
    f"Test samples:     {len(y_test)}"
)

print(
    f"Training plots:   {len(train_group_set)}"
)

print(
    f"Test plots:       {len(test_group_set)}"
)

print(
    f"Plot overlap:     {len(overlap)}"
)


if overlap:

    raise RuntimeError(
        "Group leakage detected between "
        "training and held-out test sets."
    )


# =============================================================================
# 5) Save held-out split information
# =============================================================================

split_df = pd.DataFrame(
    {
        "row_index":
            np.concatenate(
                [
                    train_idx,
                    test_idx,
                ]
            ),

        "split":
            (
                ["train"] * len(train_idx)
                + ["test"] * len(test_idx)
            ),
    }
)


split_df.to_csv(
    OUTPUT_DIR
    / "train_test_split_indices.csv",
    index=False,
)


# =============================================================================
# 6) Define positive and negative classes
# =============================================================================
#
# Upper soil (U) is treated as the positive class for binary diagnostics.
# =============================================================================

POS_LABEL = (
    int(
        le.transform(
            ["U"]
        )[0]
    )
    if "U" in le.classes_
    else 1
)


NEG_LABEL = (
    1
    - POS_LABEL
)


print(
    "\nPositive class:",
    le.inverse_transform(
        [POS_LABEL]
    )[0],
)


# =============================================================================
# 7) XGBoost class weighting
# =============================================================================

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
    float(
        neg / pos
    )
    if pos > 0
    else 1.0
)


print(
    "XGBoost scale_pos_weight:",
    round(
        scale_pos_weight,
        4,
    ),
)


# =============================================================================
# 8) Transformation functions
# =============================================================================


# -----------------------------------------------------------------------------
# A) Primary transformation:
#    log10-transformed relative abundance
# -----------------------------------------------------------------------------

def log10_relative_abundance_transform(
    X,
    pseudocount=1e-6,
):
    """
    Primary transformation used in the manuscript.

    Steps
    -----
    1. Start with non-negative abundance counts.
    2. Convert each sample to relative abundance:

           RA_ij = count_ij / total_counts_i

    3. Add a small pseudocount.
    4. Apply:

           log10(RA_ij + 1e-6)

    The pseudocount is required because log10(0) is undefined.
    """

    X = np.asarray(
        X,
        dtype=float,
    ).copy()


    X[
        X < 0
    ] = 0.0


    row_sums = X.sum(
        axis=1,
        keepdims=True,
    )


    row_sums[
        row_sums == 0
    ] = 1.0


    X_ra = (
        X
        / row_sums
    )


    return np.log10(
        X_ra
        + pseudocount
    )


# -----------------------------------------------------------------------------
# B) CLR sensitivity transformation
# -----------------------------------------------------------------------------

def clr_transform(
    X,
    pseudocount=1e-6,
):
    """
    Centered log-ratio (CLR) transformation.

    This is evaluated as a sensitivity analysis to assess whether the
    depth-classification pattern depends strongly on the primary
    log10-relative-abundance transformation.

    For each sample:

        CLR(x_j)
          =
        log(x_j + pseudocount)
          -
        mean[
            log(x_1 + pseudocount),
            ...,
            log(x_D + pseudocount)
        ]

    The transformation expresses each feature relative to the geometric
    mean abundance of all features within the same sample.

    A small pseudocount is required because microbial count matrices
    contain zeros and logarithms of zero are undefined.

    CLR can be computed directly from counts because it is invariant to
    multiplication of all feature abundances in a sample by the same
    positive constant.
    """

    X = np.asarray(
        X,
        dtype=float,
    ).copy()


    X[
        X < 0
    ] = 0.0


    X_pc = (
        X
        + pseudocount
    )


    logX = np.log(
        X_pc
    )


    return (
        logX
        - logX.mean(
            axis=1,
            keepdims=True,
        )
    )


# =============================================================================
# 9) Select transformation
# =============================================================================

def get_norm_steps(
    name
):

    name = name.upper()


    if name == "LOG10_RA":

        return [
            (
                "log10_ra",
                FunctionTransformer(
                    log10_relative_abundance_transform,
                    kw_args={
                        "pseudocount":
                            PSEUDOCOUNT,
                    },
                    validate=False,
                ),
            )
        ]


    if name == "CLR":

        return [
            (
                "clr",
                FunctionTransformer(
                    clr_transform,
                    kw_args={
                        "pseudocount":
                            PSEUDOCOUNT,
                    },
                    validate=False,
                ),
            )
        ]


    raise ValueError(
        "Unknown NORMALIZATION="
        f"'{name}'. "
        "Use 'LOG10_RA' or 'CLR'."
    )


norm_steps = get_norm_steps(
    NORMALIZATION
)


# =============================================================================
# 10) Stratified group-aware cross-validation
# =============================================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# -----------------------------------------------------------------------------
# Materialize the exact folds once.
#
# Advantages:
#
# 1. Every classifier receives the identical CV partition.
# 2. Group overlap can be verified explicitly.
# 3. Actual realized fold sizes can be used in the Nadeau-Bengio correction.
# -----------------------------------------------------------------------------

cv_splits = list(
    cv.split(
        X_train,
        y_train,
        groups=groups_train,
    )
)


fold_rows = []


print(
    "\n"
    + "=" * 80
)

print(
    "STRATIFIED GROUP K-FOLD STRUCTURE"
)

print(
    "=" * 80
)


for fold_number, (
    fold_train_idx,
    fold_valid_idx,
) in enumerate(
    cv_splits,
    start=1,
):


    fold_train_groups = set(
        groups_train[
            fold_train_idx
        ]
    )


    fold_valid_groups = set(
        groups_train[
            fold_valid_idx
        ]
    )


    fold_overlap = (
        fold_train_groups
        & fold_valid_groups
    )


    if fold_overlap:

        raise RuntimeError(
            f"Group leakage detected "
            f"in CV fold {fold_number}: "
            f"{fold_overlap}"
        )


    n_train_fold = len(
        fold_train_idx
    )


    n_valid_fold = len(
        fold_valid_idx
    )


    validation_train_ratio = (
        n_valid_fold
        / n_train_fold
    )


    fold_rows.append(
        {
            "Fold":
                fold_number,

            "Train_samples":
                n_train_fold,

            "Validation_samples":
                n_valid_fold,

            "Train_groups":
                len(
                    fold_train_groups
                ),

            "Validation_groups":
                len(
                    fold_valid_groups
                ),

            "Group_overlap":
                len(
                    fold_overlap
                ),

            "Validation_to_train_ratio":
                validation_train_ratio,
        }
    )


    print(
        f"Fold {fold_number:02d}: "
        f"train={n_train_fold:3d}, "
        f"validation={n_valid_fold:3d}, "
        f"ratio={validation_train_ratio:.6f}, "
        f"group overlap={len(fold_overlap)}"
    )


fold_df = pd.DataFrame(
    fold_rows
)


fold_df.to_csv(
    OUTPUT_DIR
    / "stratified_group_cv_folds.csv",
    index=False,
)


# =============================================================================
# 11) Nadeau-Bengio confidence intervals
# =============================================================================

def cv_intervals(
    fold_scores,
    cv_splits,
):
    """
    Calculate cross-validation summary statistics.

    Cross-validation fold estimates are correlated because the training
    datasets overlap. Therefore, a naive SD/sqrt(K) interval can
    underestimate uncertainty.

    Nadeau-Bengio corrected variance:

        corrected_variance
            =
        fold_score_variance
            *
        (
            1/K
            +
            n_validation / n_training
        )

    Because StratifiedGroupKFold can create folds of slightly unequal size,
    the n_validation/n_training term is calculated from each realized fold,
    and the mean ratio across folds is used.

    A Student's t critical value with K - 1 degrees of freedom is used for
    the 95% confidence interval.

    Reference
    ---------
    Nadeau C, Bengio Y. 2003.
    Inference for the Generalization Error.
    Machine Learning 52:239-281.
    """

    fs = np.asarray(
        fold_scores,
        dtype=float,
    )


    K = len(
        fs
    )


    mean_score = float(
        fs.mean()
    )


    sd = float(
        fs.std(
            ddof=1
        )
    )


    variance = float(
        fs.var(
            ddof=1
        )
    )


    t_value = float(
        stats.t.ppf(
            0.975,
            df=K - 1,
        )
    )


    # -------------------------------------------------------------------------
    # Naive normal interval
    # -------------------------------------------------------------------------

    naive_half = (
        1.96
        * sd
        / np.sqrt(
            K
        )
    )


    naive_ci = (
        mean_score
        - naive_half,

        mean_score
        + naive_half,
    )


    # -------------------------------------------------------------------------
    # Actual validation/training ratios from realized group-aware folds
    # -------------------------------------------------------------------------

    fold_ratios = []


    for (
        fold_train_idx,
        fold_valid_idx,
    ) in cv_splits:


        fold_ratios.append(
            len(
                fold_valid_idx
            )
            /
            len(
                fold_train_idx
            )
        )


    mean_validation_train_ratio = float(
        np.mean(
            fold_ratios
        )
    )


    # -------------------------------------------------------------------------
    # Nadeau-Bengio dependence-aware standard error
    # -------------------------------------------------------------------------

    corrected_variance = (
        variance
        * (
            1.0 / K
            + mean_validation_train_ratio
        )
    )


    corrected_se = np.sqrt(
        corrected_variance
    )


    nb_half = (
        t_value
        * corrected_se
    )


    nb_ci_raw = (
        mean_score
        - nb_half,

        mean_score
        + nb_half,
    )


    # AUC is theoretically bounded by [0, 1].
    # Preserve raw values internally, but also provide bounded values
    # for manuscript-style reporting.

    nb_ci_bounded = (
        max(
            0.0,
            nb_ci_raw[0],
        ),

        min(
            1.0,
            nb_ci_raw[1],
        ),
    )


    return {

        "mean":
            mean_score,

        "sd":
            sd,

        "variance":
            variance,

        "naive_ci":
            naive_ci,

        "nb_ci_raw":
            nb_ci_raw,

        "nb_ci":
            nb_ci_bounded,

        "mean_validation_train_ratio":
            mean_validation_train_ratio,
    }


# =============================================================================
# 12) XGBoost constructor
# =============================================================================

def make_xgb():

    kwargs = dict(

        objective="binary:logistic",

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",

        eval_metric="logloss",

        scale_pos_weight=
            scale_pos_weight,
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
# 13) Model pipelines and hyperparameter grids
# =============================================================================


# -----------------------------------------------------------------------------
# Random Forest
# -----------------------------------------------------------------------------

rf_pipe = Pipeline(

    norm_steps

    + [

        (
            "rf",

            RandomForestClassifier(

                random_state=
                    RANDOM_STATE,

                n_jobs=-1,
            ),
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
        150,
    ],

    "rf__criterion": [
        "gini",
        "entropy",
    ],

    "rf__max_features": [
        "sqrt",
        "log2",
    ],

    "rf__max_depth": [
        2,
        3,
        4,
        None,
    ],

    "rf__min_samples_split": [
        2,
        4,
    ],

    "rf__min_samples_leaf": [
        5,
        10,
        15,
    ],

    "rf__bootstrap": [
        True,
    ],

    "rf__class_weight": [
        "balanced",
        None,
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
            make_xgb(),
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
        150,
    ],

    "xgb__learning_rate": [
        0.05,
        0.10,
    ],

    "xgb__max_depth": [
        2,
        3,
        4,
    ],

    "xgb__min_child_weight": [
        1,
        3,
        5,
    ],

    "xgb__subsample": [
        0.8,
        1.0,
    ],

    "xgb__colsample_bytree": [
        0.8,
        1.0,
    ],

    "xgb__reg_alpha": [
        0.0,
        0.1,
    ],

    "xgb__reg_lambda": [
        1.0,
        2.0,
    ],
}


# -----------------------------------------------------------------------------
# Support Vector Machine
# -----------------------------------------------------------------------------

svm_pipe = Pipeline(

    norm_steps

    + [

        (
            "scaler",

            StandardScaler(
                with_mean=True,
                with_std=True,
            ),
        ),

        (
            "svc",

            SVC(
                kernel="rbf",
                probability=True,
                random_state=
                    RANDOM_STATE,
            ),
        ),

    ]
)


svm_grid = {

    "svc__C": [
        0.3,
        1,
        3,
        10,
        30,
    ],

    "svc__gamma": [
        "scale",
        0.01,
        0.003,
        0.001,
    ],

    "svc__class_weight": [
        None,
        "balanced",
    ],
}


searches = [

    (
        "Random Forest",
        rf_pipe,
        rf_grid,
    ),

    (
        "XGBoost",
        xgb_pipe,
        xgb_grid,
    ),

    (
        "SVM",
        svm_pipe,
        svm_grid,
    ),
]


# =============================================================================
# 14) Grid search and held-out evaluation
# =============================================================================

summary_rows = []

metrics_rows = []

probas_for_roc = {}

tuning_output_paths = []


for (
    model_name,
    pipeline,
    parameter_grid,
) in searches:


    print(
        "\n"
        + "=" * 80
    )


    print(
        f"{model_name.upper()} | "
        f"NORMALIZATION={NORMALIZATION}"
    )


    print(
        "=" * 80
    )


    # -------------------------------------------------------------------------
    # Hyperparameter optimization
    # -------------------------------------------------------------------------

    grid_search = GridSearchCV(

        estimator=
            pipeline,

        param_grid=
            parameter_grid,

        scoring=
            SCORING,

        cv=
            cv_splits,

        n_jobs=-1,

        refit=True,

        return_train_score=False,
    )


    grid_search.fit(
        X_train,
        y_train,
    )


    best_model = (
        grid_search
        .best_estimator_
    )


    best_params = (
        grid_search
        .best_params_
    )


    print(
        "\nBest parameters:"
    )


    print(
        best_params
    )


    # -------------------------------------------------------------------------
    # Save best parameters and full grid-search results
    # -------------------------------------------------------------------------

    model_file_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_",
        )
    )


    best_params_path = (

        OUTPUT_DIR

        / (
            f"{model_file_name}_"
            f"best_parameters_"
            f"{NORMALIZATION.lower()}.csv"
        )
    )


    pd.DataFrame(

        [

            {

                "Model":
                    model_name,

                "Normalization":
                    NORMALIZATION,

                "Scoring":
                    SCORING,

                "CV_Folds":
                    N_SPLITS,

                "Best_CV_Score":
                    grid_search.best_score_,

                **best_params,
            }

        ]

    ).to_csv(

        best_params_path,

        index=False,
    )


    grid_results_path = (

        OUTPUT_DIR

        / (
            f"{model_file_name}_"
            f"gridsearch_results_"
            f"{NORMALIZATION.lower()}.csv"
        )
    )


    pd.DataFrame(
        grid_search.cv_results_
    ).to_csv(
        grid_results_path,
        index=False,
    )


    tuning_output_paths.extend(
        [
            best_params_path,
            grid_results_path,
        ]
    )


    # -------------------------------------------------------------------------
    # Save fitted Random Forest model and feature metadata
    # -------------------------------------------------------------------------

    if model_name == "Random Forest":


        rf_model_path = (

            OUTPUT_DIR

            / (
                "best_random_forest_"
                f"{NORMALIZATION.lower()}.joblib"
            )
        )


        joblib.dump(
            best_model,
            rf_model_path,
        )


        rf_feature_names_path = (

            OUTPUT_DIR

            / (
                "rf_feature_names_"
                f"{NORMALIZATION.lower()}.csv"
            )
        )


        pd.DataFrame(

            {
                "Feature":
                    df_taxa_filtered.columns
            }

        ).to_csv(

            rf_feature_names_path,

            index=False,
        )


        rf_training_indices_path = (

            OUTPUT_DIR

            / (
                "rf_training_indices_"
                f"{NORMALIZATION.lower()}.csv"
            )
        )


        pd.DataFrame(

            {

                "row_index":
                    train_idx,

                "SoilProfile":
                    le.inverse_transform(
                        y_train
                    ),
            }

        ).to_csv(

            rf_training_indices_path,

            index=False,
        )


    # -------------------------------------------------------------------------
    # Locate best hyperparameter row
    # -------------------------------------------------------------------------

    best_row = next(

        i

        for i, params

        in enumerate(
            grid_search.cv_results_[
                "params"
            ]
        )

        if params
        == best_params
    )


    # -------------------------------------------------------------------------
    # Extract individual fold AUC scores
    # -------------------------------------------------------------------------

    fold_scores = [

        grid_search.cv_results_[
            f"split{k}_test_score"
        ][best_row]

        for k
        in range(
            N_SPLITS
        )
    ]


    # -------------------------------------------------------------------------
    # CV summary statistics
    # -------------------------------------------------------------------------

    cv_stats = cv_intervals(

        fold_scores=
            fold_scores,

        cv_splits=
            cv_splits,
    )


    cv_mean = (
        cv_stats[
            "mean"
        ]
    )


    cv_sd = (
        cv_stats[
            "sd"
        ]
    )


    print(
        "\nCV AUC per fold:"
    )


    print(
        [
            round(
                s,
                6,
            )
            for s
            in fold_scores
        ]
    )


    print(
        f"\nCV AUC mean = "
        f"{cv_mean:.6f}"
    )


    print(
        f"CV AUC SD   = "
        f"{cv_sd:.6f}"
    )


    print(
        "Mean validation/train ratio = "
        f"{cv_stats['mean_validation_train_ratio']:.8f}"
    )


    print(
        "Nadeau-Bengio 95% CI "
        "(raw) = "
        f"["
        f"{cv_stats['nb_ci_raw'][0]:.6f}, "
        f"{cv_stats['nb_ci_raw'][1]:.6f}"
        f"]"
    )


    print(
        "Nadeau-Bengio 95% CI "
        "(bounded for reporting) = "
        f"["
        f"{cv_stats['nb_ci'][0]:.6f}, "
        f"{cv_stats['nb_ci'][1]:.6f}"
        f"]"
    )


    # =========================================================================
    # Held-out test evaluation
    # =========================================================================

    y_pred_test = (
        best_model.predict(
            X_test
        )
    )


    y_proba_test = (

        best_model
        .predict_proba(
            X_test
        )[:, POS_LABEL]
    )


    train_accuracy = (
        accuracy_score(

            y_train,

            best_model.predict(
                X_train
            ),
        )
    )


    test_accuracy = (
        accuracy_score(
            y_test,
            y_pred_test,
        )
    )


    test_auc = (
        roc_auc_score(
            y_test,
            y_proba_test,
        )
    )


    kappa = (
        cohen_kappa_score(
            y_test,
            y_pred_test,
        )
    )


    sensitivity = (
        recall_score(

            y_test,

            y_pred_test,

            pos_label=
                POS_LABEL,
        )
    )


    specificity = (
        recall_score(

            y_test,

            y_pred_test,

            pos_label=
                NEG_LABEL,
        )
    )


    precision = (
        precision_score(

            y_test,

            y_pred_test,

            pos_label=
                POS_LABEL,

            zero_division=0,
        )
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred_test,
        )
    )


    f1 = (
        f1_score(

            y_test,

            y_pred_test,

            pos_label=
                POS_LABEL,

            zero_division=0,
        )
    )


    cm = (
        confusion_matrix(

            y_test,

            y_pred_test,

            labels=[
                NEG_LABEL,
                POS_LABEL,
            ],
        )
    )


    tn, fp = cm[0]

    fn, tp = cm[1]


    print(
        "\nHeld-out test metrics"
    )


    print(
        "-" * 80
    )


    print(
        f"Accuracy          = "
        f"{100 * test_accuracy:.2f}%"
    )


    print(
        f"AUC               = "
        f"{100 * test_auc:.2f}%"
    )


    print(
        f"Cohen's Kappa     = "
        f"{kappa:.3f}"
    )


    print(
        f"Sensitivity (U)   = "
        f"{100 * sensitivity:.2f}%"
    )


    print(
        f"Specificity (L)   = "
        f"{100 * specificity:.2f}%"
    )


    print(
        f"Precision         = "
        f"{100 * precision:.2f}%"
    )


    print(
        f"Balanced accuracy = "
        f"{100 * balanced_accuracy:.2f}%"
    )


    print(
        f"F1-score          = "
        f"{100 * f1:.2f}%"
    )


    print(
        f"Confusion matrix: "
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )


    # -------------------------------------------------------------------------
    # Save probabilities for ROC
    # -------------------------------------------------------------------------

    probas_for_roc[
        model_name
    ] = y_proba_test


    # -------------------------------------------------------------------------
    # Summary row
    # -------------------------------------------------------------------------

    summary_rows.append(

        {

            "Model":
                model_name,

            "Normalization":
                NORMALIZATION,

            "CV AUC (%)":
                round(
                    100 * cv_mean,
                    2,
                ),

            "CV AUC SD (%)":
                round(
                    100 * cv_sd,
                    2,
                ),

            "CV AUC NB 95% CI":
                (
                    f"["
                    f"{100 * cv_stats['nb_ci'][0]:.1f}, "
                    f"{100 * cv_stats['nb_ci'][1]:.1f}"
                    f"]"
                ),

            "Train Accuracy (%)":
                round(
                    100 * train_accuracy,
                    2,
                ),

            "Test Accuracy (%)":
                round(
                    100 * test_accuracy,
                    2,
                ),

            "Test AUC (%)":
                round(
                    100 * test_auc,
                    2,
                ),

            "Kappa":
                round(
                    kappa,
                    3,
                ),

            "Recall Lower (%)":
                round(
                    100 * specificity,
                    2,
                ),

            "Recall Upper (%)":
                round(
                    100 * sensitivity,
                    2,
                ),
        }
    )


    # -------------------------------------------------------------------------
    # Diagnostic row
    # -------------------------------------------------------------------------

    metrics_rows.append(

        {

            "Model":
                model_name,

            "Normalization":
                NORMALIZATION,

            "AUC (%)":
                round(
                    100 * test_auc,
                    2,
                ),

            "Accuracy (%)":
                round(
                    100 * test_accuracy,
                    2,
                ),

            "Sensitivity (%)":
                round(
                    100 * sensitivity,
                    2,
                ),

            "Specificity (%)":
                round(
                    100 * specificity,
                    2,
                ),

            "Precision (%)":
                round(
                    100 * precision,
                    2,
                ),

            "Balanced Accuracy (%)":
                round(
                    100 * balanced_accuracy,
                    2,
                ),

            "F1-score (%)":
                round(
                    100 * f1,
                    2,
                ),

            "Kappa":
                round(
                    kappa,
                    3,
                ),
        }
    )


# =============================================================================
# 15) ROC curves
# =============================================================================

plt.figure(
    figsize=(
        7,
        6,
    )
)


for (
    model_name,
    y_probability,
) in probas_for_roc.items():


    fpr, tpr, _ = roc_curve(

        y_test,

        y_probability,

        pos_label=
            POS_LABEL,
    )


    roc_auc_value = auc(
        fpr,
        tpr,
    )


    plt.plot(

        fpr,
        tpr,

        lw=2,

        label=(
            f"{model_name} "
            f"(AUC = {roc_auc_value:.3f})"
        ),
    )


plt.plot(
    [0, 1],
    [0, 1],
    "--",
    lw=1,
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curves "
    f"({NORMALIZATION}; "
    "group-aware held-out test)"
)


plt.legend(
    loc="lower right",
    frameon=True,
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


roc_png_path = (

    OUTPUT_DIR

    / (
        "roc_groupaware_"
        f"{NORMALIZATION.lower()}.png"
    )
)


roc_pdf_path = (

    OUTPUT_DIR

    / (
        "roc_groupaware_"
        f"{NORMALIZATION.lower()}.pdf"
    )
)


plt.savefig(
    roc_png_path,
    dpi=300,
    bbox_inches="tight",
)


plt.savefig(
    roc_pdf_path,
    bbox_inches="tight",
)


plt.close()


# =============================================================================
# 16) Save performance tables
# =============================================================================

summary_df = (
    pd.DataFrame(
        summary_rows
    )
)


metrics_df = (

    pd.DataFrame(
        metrics_rows
    )

    .sort_values(

        [
            "Accuracy (%)",
            "AUC (%)",
        ],

        ascending=False,
    )
)


summary_csv_path = (

    OUTPUT_DIR

    / (
        "summary_performance_"
        f"{NORMALIZATION.lower()}.csv"
    )
)


metrics_csv_path = (

    OUTPUT_DIR

    / (
        "diagnostic_metrics_test_"
        f"{NORMALIZATION.lower()}.csv"
    )
)


summary_df.to_csv(
    summary_csv_path,
    index=False,
)


metrics_df.to_csv(
    metrics_csv_path,
    index=False,
)


# =============================================================================
# 17) Print final tables
# =============================================================================

print(
    "\n"
    + "=" * 80
)


print(
    "MODEL PERFORMANCE SUMMARY"
)


print(
    "=" * 80
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 80
)


print(
    "HELD-OUT DIAGNOSTIC METRICS"
)


print(
    "=" * 80
)


print(
    metrics_df.to_string(
        index=False
    )
)


# =============================================================================
# 18) Final output summary
# =============================================================================

print(
    "\n"
    + "=" * 80
)


print(
    "ANALYSIS COMPLETE"
)


print(
    "=" * 80
)


print(
    f"Transformation: "
    f"{NORMALIZATION}"
)


if NORMALIZATION == "LOG10_RA":

    print(
        "Analysis role: "
        "PRIMARY manuscript analysis"
    )


elif NORMALIZATION == "CLR":

    print(
        "Analysis role: "
        "CLR transformation sensitivity analysis"
    )


print(
    f"\nResults directory:\n"
    f"{OUTPUT_DIR}"
)


print(
    "\nMain outputs:"
)


print(
    f"  {summary_csv_path.name}"
)


print(
    f"  {metrics_csv_path.name}"
)


print(
    f"  {roc_png_path.name}"
)


print(
    f"  {roc_pdf_path.name}"
)


print(
    "  stratified_group_cv_folds.csv"
)


print(
    "  train_test_split_indices.csv"
)


for path in tuning_output_paths:

    print(
        f"  {path.name}"
    )


print(
    "=" * 80
)
