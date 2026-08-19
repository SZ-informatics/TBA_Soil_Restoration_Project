# =============================================================================
# RANDOM FOREST FEATURE EXTRACTION BY SOIL DEPTH
# Run this script AFTER 01_soil_depth_modeling_full.py
# =============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# 0) Configuration
# =============================================================================

NORMALIZATION = "LOG"   # must match the modeling script

EXCEL_PATH = "data/Allmerged_16Slevel-2.xlsx"

OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOP_N_PER_DEPTH = 10


# =============================================================================
# 1) Load the fitted RF pipeline and supporting files
# =============================================================================

rf_model_path = (
    OUTPUT_DIR
    / f"best_random_forest_{NORMALIZATION.lower()}.joblib"
)

rf_feature_names_path = (
    OUTPUT_DIR
    / f"rf_feature_names_{NORMALIZATION.lower()}.csv"
)

rf_train_index_path = (
    OUTPUT_DIR
    / f"rf_training_indices_{NORMALIZATION.lower()}.csv"
)

if not rf_model_path.exists():
    raise FileNotFoundError(
        f"RF model file not found:\n{rf_model_path}\n\n"
        "Run the main modeling script first."
    )

if not rf_feature_names_path.exists():
    raise FileNotFoundError(
        f"RF feature-name file not found:\n{rf_feature_names_path}"
    )

if not rf_train_index_path.exists():
    raise FileNotFoundError(
        f"RF training-index file not found:\n{rf_train_index_path}"
    )

best_rf_model = joblib.load(
    rf_model_path
)

feature_names = (
    pd.read_csv(
        rf_feature_names_path
    )["Feature"]
    .astype(str)
    .to_numpy()
)

training_info = pd.read_csv(
    rf_train_index_path
)


# =============================================================================
# 2) Reload the original data exactly as in the main script
# =============================================================================

df = pd.read_excel(
    EXCEL_PATH,
    index_col=0
)

# Drop Archaea
df = df.loc[
    :,
    ~df.columns.str.startswith(
        "d__Archaea;"
    )
]

# Strip Bacteria prefix
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
    c
    for c in df.columns
    if c not in metadata_cols
]

df_taxa = (
    df[taxa_cols]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(0)
)

# Same prevalence filter used in model fitting
df_taxa_filtered = df_taxa.loc[
    :,
    (df_taxa >= 2).sum(
        axis=0
    ) >= 2
]

current_feature_names = np.asarray(
    df_taxa_filtered.columns
)

# Strict check that the feature matrix matches the fitted model
if not np.array_equal(
    current_feature_names,
    feature_names
):
    raise RuntimeError(
        "Feature names/order do not match the fitted RF model.\n"
        "Make sure EXCEL_PATH and preprocessing are identical "
        "to the original modeling script."
    )


# =============================================================================
# 3) Recover the exact RF training samples
# =============================================================================

row_index = (
    training_info["row_index"]
    .astype(int)
    .to_numpy()
)

train_depth = (
    training_info["SoilProfile"]
    .astype(str)
    .to_numpy()
)

X_all = df_taxa_filtered.values

X_train = X_all[
    row_index
]


# =============================================================================
# 4) Apply the SAME fitted preprocessing stored in the RF pipeline
# =============================================================================

X_train_transformed = np.asarray(
    X_train,
    dtype=float
).copy()

for (
    step_name,
    transformer
) in best_rf_model.steps[:-1]:

    X_train_transformed = (
        transformer.transform(
            X_train_transformed
        )
    )


# =============================================================================
# 5) Extract RF feature importance
# =============================================================================

rf_model = best_rf_model.named_steps[
    "rf"
]

rf_importance = np.asarray(
    rf_model.feature_importances_,
    dtype=float
)

if len(
    rf_importance
) != len(
    feature_names
):
    raise RuntimeError(
        "Number of RF importance values does not match "
        "number of taxonomic features."
    )


# =============================================================================
# 6) Calculate depth association
# =============================================================================

upper_mask = (
    train_depth == "U"
)

lower_mask = (
    train_depth == "L"
)

if not np.any(
    upper_mask
):
    raise RuntimeError(
        "No upper-soil ('U') samples found "
        "in RF training data."
    )

if not np.any(
    lower_mask
):
    raise RuntimeError(
        "No lower-soil ('L') samples found "
        "in RF training data."
    )

upper_mean = np.asarray(
    X_train_transformed[
        upper_mask
    ].mean(
        axis=0
    ),
    dtype=float
)

lower_mean = np.asarray(
    X_train_transformed[
        lower_mask
    ].mean(
        axis=0
    ),
    dtype=float
)

difference = (
    upper_mean
    - lower_mean
)

depth_association = np.where(
    difference > 0,
    "Upper soil",
    np.where(
        difference < 0,
        "Lower soil",
        "Equal"
    )
)


# =============================================================================
# 7) Build full RF feature table
# =============================================================================

rf_features = pd.DataFrame(
    {
        "Feature":
            feature_names,

        "RF_Importance":
            rf_importance,

        "Upper_Mean_Transformed":
            upper_mean,

        "Lower_Mean_Transformed":
            lower_mean,

        "Upper_minus_Lower":
            difference,

        "Depth_Association":
            depth_association,
    }
)

rf_features = (
    rf_features
    .sort_values(
        "RF_Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

rf_features[
    "Overall_RF_Rank"
] = np.arange(
    1,
    len(
        rf_features
    ) + 1
)


# =============================================================================
# 8) Extract top lower- and upper-soil-associated RF features
# =============================================================================

top_lower = (
    rf_features[
        rf_features[
            "Depth_Association"
        ] == "Lower soil"
    ]
    .head(
        TOP_N_PER_DEPTH
    )
    .copy()
    .reset_index(
        drop=True
    )
)

top_upper = (
    rf_features[
        rf_features[
            "Depth_Association"
        ] == "Upper soil"
    ]
    .head(
        TOP_N_PER_DEPTH
    )
    .copy()
    .reset_index(
        drop=True
    )
)

top_lower[
    "Depth_Rank"
] = np.arange(
    1,
    len(
        top_lower
    ) + 1
)

top_upper[
    "Depth_Rank"
] = np.arange(
    1,
    len(
        top_upper
    ) + 1
)

top_features_by_depth = pd.concat(
    [
        top_lower,
        top_upper
    ],
    ignore_index=True
)


# =============================================================================
# 9) Save CSV outputs
# =============================================================================

all_features_path = (
    OUTPUT_DIR
    / (
        "RF_all_features_by_depth_"
        f"{NORMALIZATION.lower()}.csv"
    )
)

top_lower_path = (
    OUTPUT_DIR
    / (
        "RF_top_lower_soil_features_"
        f"{NORMALIZATION.lower()}.csv"
    )
)

top_upper_path = (
    OUTPUT_DIR
    / (
        "RF_top_upper_soil_features_"
        f"{NORMALIZATION.lower()}.csv"
    )
)

combined_path = (
    OUTPUT_DIR
    / (
        "RF_top_features_by_depth_"
        f"{NORMALIZATION.lower()}.csv"
    )
)

rf_features.to_csv(
    all_features_path,
    index=False
)

top_lower.to_csv(
    top_lower_path,
    index=False
)

top_upper.to_csv(
    top_upper_path,
    index=False
)

top_features_by_depth.to_csv(
    combined_path,
    index=False
)


# =============================================================================
# 10) Print top features
# =============================================================================

print(
    "\n"
    + "=" * 78
)

print(
    "TOP LOWER-SOIL-ASSOCIATED RANDOM FOREST FEATURES"
)

print(
    "=" * 78
)

print(
    top_lower[
        [
            "Depth_Rank",
            "Feature",
            "RF_Importance",
            "Lower_Mean_Transformed",
            "Upper_Mean_Transformed",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\n"
    + "=" * 78
)

print(
    "TOP UPPER-SOIL-ASSOCIATED RANDOM FOREST FEATURES"
)

print(
    "=" * 78
)

print(
    top_upper[
        [
            "Depth_Rank",
            "Feature",
            "RF_Importance",
            "Upper_Mean_Transformed",
            "Lower_Mean_Transformed",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 11) Plot top predictive features by soil depth
# =============================================================================

# Arrange groups so LOWER-soil features appear together at the TOP
# and UPPER-soil features appear together at the BOTTOM.
#
# For a horizontal bar plot, the first rows are drawn at the bottom.
# Therefore, upper-soil rows are placed first and lower-soil rows last.
plot_upper = (
    top_upper
    .sort_values(
        "RF_Importance",
        ascending=True
    )
    .copy()
)

plot_lower = (
    top_lower
    .sort_values(
        "RF_Importance",
        ascending=True
    )
    .copy()
)

plot_df = pd.concat(
    [
        plot_upper,
        plot_lower
    ],
    ignore_index=True
)

plot_colors = plot_df[
    "Depth_Association"
].map(
    {
        "Lower soil":
            "tab:blue",

        "Upper soil":
            "orange",
    }
)

plt.figure(
    figsize=(
        10,
        8
    )
)

plt.barh(
    plot_df[
        "Feature"
    ],
    plot_df[
        "RF_Importance"
    ],
    color=plot_colors
)

plt.xlabel(
    "Feature Importance",
    fontweight="bold"
)

plt.ylabel(
    ""
)

plt.title(
    "Top Predictive 16S Phyla for Soil Depth",
    fontweight="bold"
)

legend_elements = [
    Patch(
        facecolor="tab:blue",
        label="Lower soil"
    ),
    Patch(
        facecolor="orange",
        label="Upper soil"
    )
]

plt.legend(
    handles=legend_elements,
    title="Depth Association",
    loc="lower right"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plot_png_path = (
    OUTPUT_DIR
    / (
        "RF_top_features_by_depth_"
        f"{NORMALIZATION.lower()}.png"
    )
)

plot_pdf_path = (
    OUTPUT_DIR
    / (
        "RF_top_features_by_depth_"
        f"{NORMALIZATION.lower()}.pdf"
    )
)

plt.savefig(
    plot_png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    plot_pdf_path,
    bbox_inches="tight"
)

plt.show()


# =============================================================================
# 12) Final output summary
# =============================================================================

print(
    "\n"
    + "=" * 78
)

print(
    "RF FEATURE EXTRACTION COMPLETE"
)

print(
    "=" * 78
)

print(
    f"\nAll RF feature results saved to:\n"
    f"{OUTPUT_DIR}\n"
)

print(
    "Saved files:"
)

print(
    f"  1. {all_features_path.name}"
)

print(
    f"  2. {top_lower_path.name}"
)

print(
    f"  3. {top_upper_path.name}"
)

print(
    f"  4. {combined_path.name}"
)

print(
    f"  5. {plot_png_path.name}"
)

print(
    f"  6. {plot_pdf_path.name}"
)