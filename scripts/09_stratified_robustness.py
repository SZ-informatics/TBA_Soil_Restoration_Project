#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_stratified_robustness.py

Stratified robustness analysis for soil-depth classification.

Primary preprocessing
---------------------
Counts
    -> per-sample relative abundance
    -> log10(relative abundance + 1e-6)
    -> Random Forest

Purpose
-------
The optimized Random Forest configuration identified in the pooled
soil-depth classification analysis is retrained separately within:

    1. Switchgrass
    2. Perennial Sorghum
    3. Annual Sorghum
    4. Control
    5. Biochar
    6. TBA

This analysis evaluates whether the soil-depth classification signal
and leading predictive taxa remain broadly consistent across plant
systems and amendment treatments.

Validation
----------
Within each stratum:

    - Group-aware StratifiedGroupKFold
    - Maximum of 5 folds because strata are smaller than the pooled dataset
    - Groups defined as Species x Amendment x Block

The same optimized Random Forest hyperparameter configuration from the
pooled LOG10_RA analysis is used within every stratum.

Repository structure
--------------------
Input:
    data/Allmerged_16Slevel-2.xlsx

Tabular outputs:
    output/stratified/
        stratified_results_log10_ra.csv
        stratified_top_features_log10_ra.csv
        stratified_taxa_occurrence_log10_ra.csv

Figure outputs:
    figures/
        Fig_stratified_importance_log10_ra.png
        Fig_stratified_importance_log10_ra.pdf
"""


# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

import sklearn

from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)

from sklearn.preprocessing import (
    LabelEncoder,
    FunctionTransformer,
)

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# 0) CONFIGURATION
# =============================================================================

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Input data
# -----------------------------------------------------------------------------

EXCEL_PATH = (
    Path("data")
    / "Allmerged_16Slevel-2.xlsx"
)


# -----------------------------------------------------------------------------
# Tabular/statistical outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    Path("output")
    / "stratified"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------------------------------------------------
# Final manuscript figures
# -----------------------------------------------------------------------------

FIGURE_DIR = Path("figures")

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------------------------------------------------
# Number of taxa displayed for each direction within each stratum
# -----------------------------------------------------------------------------

K_LOWER = 6
K_UPPER = 6


# -----------------------------------------------------------------------------
# Smaller strata -> maximum 5-fold group-aware cross-validation
# -----------------------------------------------------------------------------

MAX_SPLITS = 5


# -----------------------------------------------------------------------------
# Plot colors
# -----------------------------------------------------------------------------

LOWER_COLOR = "#2c6fbb"
UPPER_COLOR = "#e08214"


# -----------------------------------------------------------------------------
# LOG10 relative-abundance pseudocount
# -----------------------------------------------------------------------------

PSEUDOCOUNT = 1e-6


np.random.seed(
    RANDOM_STATE
)


warnings.filterwarnings(
    "ignore"
)


print("=" * 78)

print(
    "STRATIFIED ROBUSTNESS ANALYSIS — LOG10_RA RANDOM FOREST"
)

print("=" * 78)


print(
    f"scikit-learn: {sklearn.__version__}"
)


print(
    f"Input file: {EXCEL_PATH}"
)


print(
    f"Tabular output directory: {OUTPUT_DIR}"
)


print(
    f"Figure directory: {FIGURE_DIR}"
)


print(
    "Primary transformation: "
    "log10(relative abundance + 1e-6)"
)


print(
    "RF configuration: "
    "n_estimators=70, entropy, max_depth=2, "
    "min_samples_leaf=10, class_weight=None"
)


# =============================================================================
# 1) LOAD DATA
# =============================================================================

if not EXCEL_PATH.exists():

    raise FileNotFoundError(
        "\nInput file not found:\n"
        f"{EXCEL_PATH}\n\n"
        "Run this script from the repository root directory."
    )


df = pd.read_excel(
    EXCEL_PATH,
    index_col=0
)


print(
    f"\nOriginal table shape: {df.shape}"
)


# =============================================================================
# 2) REMOVE ARCHAEA
# =============================================================================

df = df.loc[
    :,
    ~df.columns.str.startswith(
        "d__Archaea;"
    )
]


# =============================================================================
# 3) REMOVE BACTERIAL PREFIX
# =============================================================================

df.columns = (
    df.columns
    .str.replace(
        r"^d__Bacteria;",
        "",
        regex=True
    )
)


# =============================================================================
# 4) DEFINE METADATA AND TAXONOMIC COLUMNS
# =============================================================================

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


counts = (
    df[
        taxa_cols
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(0)
)


print(
    f"Taxonomic features before prevalence filtering: "
    f"{counts.shape[1]}"
)


# =============================================================================
# 5) SAME PREVALENCE FILTER AS PRIMARY ML
#
# Keep taxa with >=2 reads in >=2 samples
# =============================================================================

counts = counts.loc[
    :,
    (
        counts >= 2
    ).sum(
        axis=0
    ) >= 2
]


print(
    f"Taxonomic features after prevalence filtering: "
    f"{counts.shape[1]}"
)


# -----------------------------------------------------------------------------
# Feature labels
# -----------------------------------------------------------------------------

feature_names = np.asarray(
    counts.columns
)


display_feature_names = np.asarray(
    [
        f
        .replace(
            "p__",
            ""
        )
        .replace(
            "c__",
            ""
        )

        for f in feature_names
    ]
)


X = counts.to_numpy(
    dtype=float
)


# =============================================================================
# 6) TARGET LABEL
# =============================================================================

profile = (
    df[
        "SoilProfile"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace(
        {
            "UPPER":
                "U",

            "LOWER":
                "L",
        }
    )
    .to_numpy()
)


le = LabelEncoder()


y = le.fit_transform(
    profile
)


print(
    f"\nClasses: {list(le.classes_)}"
)


print(
    "Class counts:"
)


print(
    pd.Series(
        profile
    )
    .value_counts()
    .to_string()
)


# =============================================================================
# 7) DEFINE PLOT-LEVEL GROUPS
#
# Species x Amendment x Block
# =============================================================================

required_group_columns = [

    "Species",

    "Amendment",

    "Block",
]


missing_group_columns = [

    c

    for c in required_group_columns

    if c not in df.columns
]


if missing_group_columns:

    raise ValueError(
        "Missing required grouping columns: "
        f"{missing_group_columns}"
    )


groups = (
    df[
        "Species"
    ].astype(str)
    + "_"
    + df[
        "Amendment"
    ].astype(str)
    + "_"
    + df[
        "Block"
    ].astype(str)
).to_numpy()


species = (
    df[
        "Species"
    ]
    .astype(str)
    .to_numpy()
)


amendment = (
    df[
        "Amendment"
    ]
    .astype(str)
    .to_numpy()
)


print(
    f"\nUnique plot-level groups: "
    f"{len(np.unique(groups))}"
)


# =============================================================================
# 8) LOG10 RELATIVE-ABUNDANCE TRANSFORMATION
# =============================================================================

def log10_relative_abundance_transform(
    A,
    pseudocount=1e-6
):
    """
    Convert sample counts to relative abundance and then apply:

        log10(relative abundance + pseudocount)

    This is the same primary preprocessing used in the pooled
    LOG10_RA soil-depth classification analysis.
    """

    A = np.asarray(
        A,
        dtype=float
    ).copy()


    # Prevent negative values
    A[
        A < 0
    ] = 0.0


    # Calculate per-sample totals
    row_sums = A.sum(
        axis=1,
        keepdims=True
    )


    # Avoid division by zero
    row_sums[
        row_sums == 0
    ] = 1.0


    # Convert to relative abundance
    RA = (
        A
        / row_sums
    )


    # Apply LOG10 transformation
    return np.log10(
        RA
        + pseudocount
    )


# =============================================================================
# 9) TRANSFORM FULL MATRIX FOR DEPTH-ASSOCIATION DIRECTION ONLY
#
# The model receives raw counts and performs the identical transformation
# inside its sklearn Pipeline.
# =============================================================================

X_log10_ra = (
    log10_relative_abundance_transform(
        X,
        pseudocount=PSEUDOCOUNT
    )
)


print(
    "\nTransformed matrix range:"
)


print(
    f"Minimum = "
    f"{np.nanmin(X_log10_ra):.6f}"
)


print(
    f"Maximum = "
    f"{np.nanmax(X_log10_ra):.6f}"
)


# =============================================================================
# 10) OPTIMIZED POOLED RANDOM FOREST CONFIGURATION
#
# These are the optimal settings selected in the pooled LOG10_RA
# GridSearchCV analysis.
# =============================================================================

def best_rf():

    return Pipeline(
        [

            (
                "log10_ra",

                FunctionTransformer(
                    log10_relative_abundance_transform,
                    kw_args={
                        "pseudocount":
                            PSEUDOCOUNT
                    },
                    validate=False
                )
            ),

            (
                "rf",

                RandomForestClassifier(

                    n_estimators=70,

                    criterion="entropy",

                    max_features="sqrt",

                    min_samples_split=2,

                    min_samples_leaf=10,

                    bootstrap=True,

                    max_depth=2,

                    class_weight=None,

                    random_state=RANDOM_STATE,

                    n_jobs=-1,
                )
            ),
        ]
    )


# =============================================================================
# 11) DEFINE STRATA
#
# Plant-system strata:
#   Switchgrass
#   Perennial Sorghum
#   Annual Sorghum
#
# Amendment strata:
#   Control
#   Biochar
#   TBA
# =============================================================================

strata = [

    (
        "Switchgrass",
        species == "Native Grass"
    ),

    (
        "Perennial Sorghum",
        species == "PS"
    ),

    (
        "Annual Sorghum",
        species == "SOBI"
    ),

    (
        "Control",
        amendment == "Control"
    ),

    (
        "Biochar",
        amendment == "Biochar"
    ),

    (
        "TBA",
        amendment == "TBA"
    ),
]


# =============================================================================
# 12) STORAGE
# =============================================================================

summary_rows = []

feature_rows = []

panels = []


# =============================================================================
# 13) RUN STRATIFIED ROBUSTNESS ANALYSES
# =============================================================================

for label, mask in strata:


    print(
        "\n"
        + "=" * 78
    )


    print(
        f"STRATUM: {label}"
    )


    print(
        "=" * 78
    )


    # -------------------------------------------------------------------------
    # Subset data
    # -------------------------------------------------------------------------

    Xs = X[
        mask
    ]


    ys = y[
        mask
    ]


    gs = groups[
        mask
    ]


    ps = profile[
        mask
    ]


    transformed_subset = (
        X_log10_ra[
            mask
        ]
    )


    # -------------------------------------------------------------------------
    # Determine feasible number of group-aware folds
    # -------------------------------------------------------------------------

    n_unique_groups = len(
        np.unique(
            gs
        )
    )


    class_counts = np.bincount(
        ys
    )


    min_class_count = int(
        class_counts.min()
    )


    n_splits = min(
        MAX_SPLITS,
        n_unique_groups,
        min_class_count
    )


    if n_splits < 2:

        raise RuntimeError(
            f"Not enough independent groups/classes "
            f"for CV in stratum '{label}'."
        )


    print(
        f"Samples: {len(ys)}"
    )


    print(
        f"Unique groups: {n_unique_groups}"
    )


    print(
        f"CV folds: {n_splits}"
    )


    print(
        "Depth counts:"
    )


    print(
        pd.Series(
            ps
        )
        .value_counts()
        .to_string()
    )


    # -------------------------------------------------------------------------
    # Group-aware cross-validation
    # -------------------------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    model = best_rf()


    cv_results = cross_validate(
        model,
        Xs,
        ys,
        groups=gs,
        cv=cv,
        scoring={
            "accuracy":
                "accuracy",

            "roc_auc":
                "roc_auc",
        },
        n_jobs=-1,
        return_train_score=False
    )


    accuracy_scores = (
        cv_results[
            "test_accuracy"
        ]
    )


    auc_scores = (
        cv_results[
            "test_roc_auc"
        ]
    )


    acc_mean = float(
        accuracy_scores.mean()
    )


    acc_sd = float(
        accuracy_scores.std(
            ddof=1
        )
    )


    auc_mean = float(
        auc_scores.mean()
    )


    auc_sd = float(
        auc_scores.std(
            ddof=1
        )
    )


    print(
        f"CV accuracy = "
        f"{acc_mean:.3f} ± {acc_sd:.3f}"
    )


    print(
        f"CV AUC      = "
        f"{auc_mean:.3f} ± {auc_sd:.3f}"
    )


    # -------------------------------------------------------------------------
    # Fit final model to all samples in this stratum
    # for feature-importance extraction
    # -------------------------------------------------------------------------

    fitted_model = best_rf()


    fitted_model.fit(
        Xs,
        ys
    )


    importance = (
        fitted_model
        .named_steps[
            "rf"
        ]
        .feature_importances_
    )


    # -------------------------------------------------------------------------
    # Determine depth direction using transformed relative abundance
    #
    # Positive:
    #     lower mean > upper mean
    #
    # Negative:
    #     upper mean > lower mean
    # -------------------------------------------------------------------------

    lower_mask = (
        ps == "L"
    )


    upper_mask = (
        ps == "U"
    )


    lower_mean = (
        transformed_subset[
            lower_mask
        ]
        .mean(
            axis=0
        )
    )


    upper_mean = (
        transformed_subset[
            upper_mask
        ]
        .mean(
            axis=0
        )
    )


    difference = (
        lower_mean
        - upper_mean
    )


    # -------------------------------------------------------------------------
    # Rank taxa by Random Forest importance
    # -------------------------------------------------------------------------

    ranked_indices = np.argsort(
        importance
    )[
        ::-1
    ]


    lower_indices = [

        i

        for i in ranked_indices

        if difference[
            i
        ] > 0

    ][
        :K_LOWER
    ]


    upper_indices = [

        i

        for i in ranked_indices

        if difference[
            i
        ] <= 0

    ][
        :K_UPPER
    ]


    selected_indices = (
        lower_indices
        + upper_indices
    )


    # -------------------------------------------------------------------------
    # Store panel information for plotting
    # -------------------------------------------------------------------------

    panel_taxa = [

        display_feature_names[
            i
        ]

        for i in selected_indices
    ]


    panel_importance = [

        importance[
            i
        ]

        for i in selected_indices
    ]


    panel_colors = (
        [
            LOWER_COLOR
        ]
        * len(
            lower_indices
        )

        +

        [
            UPPER_COLOR
        ]
        * len(
            upper_indices
        )
    )


    panels.append(
        {

            "label":
                label,

            "n":
                int(
                    mask.sum()
                ),

            "acc":
                acc_mean,

            "acc_sd":
                acc_sd,

            "auc":
                auc_mean,

            "auc_sd":
                auc_sd,

            "taxa":
                panel_taxa,

            "importance":
                panel_importance,

            "colors":
                panel_colors,
        }
    )


    # -------------------------------------------------------------------------
    # Summary output row
    # -------------------------------------------------------------------------

    summary_rows.append(
        {

            "Stratum":
                label,

            "N":
                int(
                    mask.sum()
                ),

            "Unique_Groups":
                n_unique_groups,

            "CV_Folds":
                n_splits,

            "CV_Accuracy_Mean":
                round(
                    acc_mean,
                    4
                ),

            "CV_Accuracy_SD":
                round(
                    acc_sd,
                    4
                ),

            "CV_AUC_Mean":
                round(
                    auc_mean,
                    4
                ),

            "CV_AUC_SD":
                round(
                    auc_sd,
                    4
                ),

            "Top_Lower":
                "; ".join(
                    display_feature_names[
                        i
                    ]

                    for i in lower_indices
                ),

            "Top_Upper":
                "; ".join(
                    display_feature_names[
                        i
                    ]

                    for i in upper_indices
                ),
        }
    )


    # -------------------------------------------------------------------------
    # Detailed lower-soil feature output
    # -------------------------------------------------------------------------

    for rank, i in enumerate(
        lower_indices,
        start=1
    ):

        feature_rows.append(
            {

                "Stratum":
                    label,

                "Depth_Association":
                    "Lower soil",

                "Depth_Rank":
                    rank,

                "Feature":
                    display_feature_names[
                        i
                    ],

                "Original_Feature_Name":
                    feature_names[
                        i
                    ],

                "RF_Importance":
                    float(
                        importance[
                            i
                        ]
                    ),

                "Lower_Mean_Log10_RA":
                    float(
                        lower_mean[
                            i
                        ]
                    ),

                "Upper_Mean_Log10_RA":
                    float(
                        upper_mean[
                            i
                        ]
                    ),

                "Lower_minus_Upper_Log10_RA":
                    float(
                        difference[
                            i
                        ]
                    ),
            }
        )


    # -------------------------------------------------------------------------
    # Detailed upper-soil feature output
    # -------------------------------------------------------------------------

    for rank, i in enumerate(
        upper_indices,
        start=1
    ):

        feature_rows.append(
            {

                "Stratum":
                    label,

                "Depth_Association":
                    "Upper soil",

                "Depth_Rank":
                    rank,

                "Feature":
                    display_feature_names[
                        i
                    ],

                "Original_Feature_Name":
                    feature_names[
                        i
                    ],

                "RF_Importance":
                    float(
                        importance[
                            i
                        ]
                    ),

                "Lower_Mean_Log10_RA":
                    float(
                        lower_mean[
                            i
                        ]
                    ),

                "Upper_Mean_Log10_RA":
                    float(
                        upper_mean[
                            i
                        ]
                    ),

                "Lower_minus_Upper_Log10_RA":
                    float(
                        difference[
                            i
                        ]
                    ),
            }
        )


# =============================================================================
# 14) BUILD OUTPUT TABLES
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)


features_df = pd.DataFrame(
    feature_rows
)


# =============================================================================
# 15) SAVE TABULAR RESULTS
# =============================================================================

summary_output_path = (
    OUTPUT_DIR
    / "stratified_results_log10_ra.csv"
)


feature_output_path = (
    OUTPUT_DIR
    / "stratified_top_features_log10_ra.csv"
)


summary_df.to_csv(
    summary_output_path,
    index=False
)


features_df.to_csv(
    feature_output_path,
    index=False
)


print(
    "\n"
    + "=" * 78
)


print(
    "STRATIFIED SUMMARY"
)


print(
    "=" * 78
)


print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 16) IDENTIFY RECURRING TOP TAXA ACROSS STRATA
# =============================================================================

occurrence_table = (
    features_df[
        [
            "Stratum",
            "Feature",
            "Depth_Association"
        ]
    ]
    .drop_duplicates()
    .groupby(
        [
            "Feature",
            "Depth_Association"
        ]
    )
    .size()
    .reset_index(
        name="N_Strata"
    )
    .sort_values(
        [
            "N_Strata",
            "Feature"
        ],
        ascending=[
            False,
            True
        ]
    )
)


occurrence_output_path = (
    OUTPUT_DIR
    / "stratified_taxa_occurrence_log10_ra.csv"
)


occurrence_table.to_csv(
    occurrence_output_path,
    index=False
)


print(
    "\n"
    + "=" * 78
)


print(
    "TAXA OCCURRENCE ACROSS STRATA"
)


print(
    "=" * 78
)


print(
    occurrence_table.to_string(
        index=False
    )
)


print(
    "\nTaxa occurring in at least 4 of 6 strata:"
)


recurrent = occurrence_table[
    occurrence_table[
        "N_Strata"
    ] >= 4
]


if recurrent.empty:

    print(
        "  None"
    )


else:

    print(
        recurrent.to_string(
            index=False
        )
    )


# =============================================================================
# 17) CREATE 2 × 3 STRATIFIED FEATURE-IMPORTANCE FIGURE
# =============================================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        16,
        11
    )
)


for panel, ax in zip(
    panels,
    axes.ravel()
):


    taxa = (
        panel[
            "taxa"
        ][
            ::-1
        ]
    )


    values = (
        panel[
            "importance"
        ][
            ::-1
        ]
    )


    colors = (
        panel[
            "colors"
        ][
            ::-1
        ]
    )


    ax.barh(
        range(
            len(
                taxa
            )
        ),
        values,
        color=colors,
        edgecolor="0.25",
        linewidth=0.4
    )


    ax.set_yticks(
        range(
            len(
                taxa
            )
        )
    )


    ax.set_yticklabels(
        taxa,
        fontsize=9
    )


    ax.set_xlabel(
        "Random Forest importance",
        fontsize=9
    )


    ax.set_title(
        (
            f"{panel['label']} "
            f"(n={panel['n']}; "
            f"Acc {panel['acc']:.2f}, "
            f"AUC {panel['auc']:.2f})"
        ),
        fontsize=11,
        fontweight="bold"
    )


    ax.tick_params(
        axis="x",
        labelsize=8
    )


    ax.grid(
        axis="x",
        alpha=0.3
    )


# =============================================================================
# 18) FIGURE TITLE AND LEGEND
# =============================================================================

fig.suptitle(
    (
        "Depth-predictive taxa across plant systems and amendments\n"
        f"(top {K_LOWER} lower- and top {K_UPPER} "
        "upper-associated phyla per stratum; "
        "bar length = Random Forest importance)"
    ),
    fontsize=14,
    fontweight="bold"
)


legend = [

    Patch(
        facecolor=LOWER_COLOR,
        edgecolor="0.25",
        label="Lower soil-associated (15–30 cm)"
    ),

    Patch(
        facecolor=UPPER_COLOR,
        edgecolor="0.25",
        label="Upper soil-associated (0–15 cm)"
    ),
]


fig.legend(
    handles=legend,
    loc="lower center",
    ncol=2,
    fontsize=12,
    frameon=True,
    bbox_to_anchor=(
        0.5,
        -0.005
    )
)


plt.tight_layout(
    rect=[
        0,
        0.035,
        1,
        0.94
    ]
)


# =============================================================================
# 19) SAVE FIGURE TO figures/
# =============================================================================

figure_png_path = (
    FIGURE_DIR
    / "Fig_stratified_importance_log10_ra.png"
)


figure_pdf_path = (
    FIGURE_DIR
    / "Fig_stratified_importance_log10_ra.pdf"
)


plt.savefig(
    figure_png_path,
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    figure_pdf_path,
    bbox_inches="tight"
)


plt.close()


# =============================================================================
# 20) FINAL PERFORMANCE RANGE
# =============================================================================

accuracy_min = (
    summary_df[
        "CV_Accuracy_Mean"
    ]
    .min()
)


accuracy_max = (
    summary_df[
        "CV_Accuracy_Mean"
    ]
    .max()
)


auc_min = (
    summary_df[
        "CV_AUC_Mean"
    ]
    .min()
)


auc_max = (
    summary_df[
        "CV_AUC_Mean"
    ]
    .max()
)


print(
    "\n"
    + "=" * 78
)


print(
    "FINAL STRATIFIED RANGE"
)


print(
    "=" * 78
)


print(
    f"CV accuracy range: "
    f"{100 * accuracy_min:.1f}% – "
    f"{100 * accuracy_max:.1f}%"
)


print(
    f"CV AUC range: "
    f"{100 * auc_min:.1f}% – "
    f"{100 * auc_max:.1f}%"
)


# =============================================================================
# 21) FINAL OUTPUT SUMMARY
# =============================================================================

print(
    "\n"
    + "=" * 78
)


print(
    "STRATIFIED ROBUSTNESS ANALYSIS COMPLETE"
)


print(
    "=" * 78
)


print(
    "\nTabular outputs saved to:"
)


print(
    f"  {OUTPUT_DIR}"
)


print(
    "\nFigure outputs saved to:"
)


print(
    f"  {FIGURE_DIR}"
)


print(
    "\nSaved result files:"
)


print(
    f"  1. {summary_output_path.name}"
)


print(
    f"  2. {feature_output_path.name}"
)


print(
    f"  3. {occurrence_output_path.name}"
)


print(
    "\nSaved figure files:"
)


print(
    f"  4. {figure_png_path.name}"
)


print(
    f"  5. {figure_pdf_path.name}"
)


print(
    "\nPrimary preprocessing:"
)


print(
    "  counts"
)


print(
    "  -> relative abundance"
)


print(
    "  -> log10(relative abundance + 1e-6)"
)


print(
    "  -> optimized pooled Random Forest configuration"
)


print(
    "=" * 78
)
