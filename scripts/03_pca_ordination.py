# =============================================================================
# PCA ANALYSIS FOR 16S PHYLA
#
# Transformation:
#   counts
#   -> relative abundance
#   -> log10(relative abundance + 1e-6)
#   -> PCA
#
# Outputs saved to:
#   ./output/pca/
#
# Upper soil = orange
# Lower soil = blue
# =============================================================================


# =============================================================================
# 0) IMPORTS
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from matplotlib.lines import Line2D


# =============================================================================
# 1) PATHS / SETTINGS
# =============================================================================

EXCEL_PATH = "Allmerged_16Slevel-2.xlsx"

# All PCA outputs are written to a repository-relative output directory.
# The directory is created automatically if it does not already exist.
OUTPUT_DIR = (
    Path("output")
    / "pca"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sample_col = "index"
depth_col = "SoilProfile"

TOP_N_ABUNDANT = 25
TOP_N_LOADINGS = 20

EPS = 1e-6

RANDOM_STATE = 42


print("=" * 78)
print("PCA ANALYSIS — LOG10 RELATIVE ABUNDANCE")
print("=" * 78)

print(
    f"Input file: {EXCEL_PATH}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    "Transformation: "
    "log10(relative abundance + 1e-6)"
)


# =============================================================================
# 2) LOAD DATA
# =============================================================================

df = pd.read_excel(
    EXCEL_PATH
)


metadata_cols = [
    "index",
    "BarcodeSequence",
    "SampleCode",
    "LinkerPrimerSequence",
    "Time Period",
    "Species",
    "Amendment",
    "SoilProfile",
    "Block",
    "Group1",
    "Group2",
]


taxa_cols = [
    c
    for c in df.columns
    if c not in metadata_cols
]


print(
    "\nOriginal data shape:",
    df.shape
)

print(
    "Number of taxonomic columns:",
    len(taxa_cols)
)


# =============================================================================
# 3) CONVERT TO LONG FORMAT
# =============================================================================

long = df.melt(
    id_vars=[
        sample_col,
        depth_col
    ],
    value_vars=taxa_cols,
    var_name="Taxon",
    value_name="Abundance"
)


# =============================================================================
# 4) EXTRACT PHYLUM
# =============================================================================

long[
    "Phylum"
] = long[
    "Taxon"
].str.extract(
    r"d__.*?;p__([^;]+)"
)


long = long.dropna(
    subset=[
        "Phylum"
    ]
)


long[
    "Abundance"
] = pd.to_numeric(
    long[
        "Abundance"
    ],
    errors="coerce"
).fillna(0)


# =============================================================================
# 5) SUM TO PHYLUM LEVEL PER SAMPLE
# =============================================================================

phy = (
    long
    .groupby(
        [
            sample_col,
            depth_col,
            "Phylum"
        ],
        as_index=False
    )[
        "Abundance"
    ]
    .sum()
)


# =============================================================================
# 6) CONVERT TO RELATIVE ABUNDANCE
# =============================================================================

phy[
    "RelAbund"
] = (
    phy
    .groupby(
        sample_col
    )[
        "Abundance"
    ]
    .transform(
        lambda x:
            x / (
                x.sum()
                if x.sum() != 0
                else 1.0
            )
    )
)


# =============================================================================
# 7) CREATE SAMPLE × PHYLUM MATRIX
# =============================================================================

X = phy.pivot_table(
    index=sample_col,
    columns="Phylum",
    values="RelAbund",
    fill_value=0
)


print(
    "\nInitial PCA matrix shape:",
    X.shape
)


# =============================================================================
# 8) VERIFY RELATIVE-ABUNDANCE SUMS
# =============================================================================

ra_sums = X.sum(
    axis=1
)


print(
    "\nRelative-abundance sums:"
)

print(
    f"Minimum = {ra_sums.min():.6f}"
)

print(
    f"Maximum = {ra_sums.max():.6f}"
)

print(
    f"Mean    = {ra_sums.mean():.6f}"
)


# =============================================================================
# 9) KEEP TOP 25 MOST ABUNDANT PHYLA
# =============================================================================

top_phyla_abund = (
    X
    .sum(
        axis=0
    )
    .sort_values(
        ascending=False
    )
    .head(
        TOP_N_ABUNDANT
    )
    .index
)


X = X.loc[
    :,
    top_phyla_abund
]


print(
    "\nPCA matrix after keeping top phyla:",
    X.shape
)


# =============================================================================
# 10) LOG10 TRANSFORMATION
#
# This matches the new LOG10_RA ML representation:
#
#     log10(relative abundance + 1e-6)
# =============================================================================

X_log = np.log10(
    X + EPS
)


print(
    "\nTransformed value range:"
)

print(
    f"Minimum = {X_log.to_numpy().min():.6f}"
)

print(
    f"Maximum = {X_log.to_numpy().max():.6f}"
)


# =============================================================================
# 11) FULL PCA
# =============================================================================

pca_full = PCA(
    random_state=RANDOM_STATE
)


pca_full.fit(
    X_log
)


# =============================================================================
# 12) FORCE PCA ORIENTATION TO MATCH ORIGINAL FIGURE
#
# PCA signs are arbitrary.
#
# Zixibacteria is used as the reference taxon so that:
#
#   PC1 loading > 0
#   PC2 loading > 0
#
# This affects orientation only.
# It does NOT affect:
#   - explained variance
#   - sample distances
#   - biological relationships
# =============================================================================

reference_taxon = "Zixibacteria"


if reference_taxon not in X.columns:

    raise ValueError(
        f"{reference_taxon} was not found "
        "in the PCA feature matrix."
    )


zixi_idx = X.columns.get_loc(
    reference_taxon
)


# Force positive Zixibacteria PC1 loading
if (
    pca_full.components_[
        0,
        zixi_idx
    ] < 0
):

    pca_full.components_[
        0,
        :
    ] *= -1


# Force positive Zixibacteria PC2 loading
if (
    pca_full.components_[
        1,
        zixi_idx
    ] < 0
):

    pca_full.components_[
        1,
        :
    ] *= -1


# =============================================================================
# 13) EXPLAINED VARIANCE
# =============================================================================

expl = (
    pca_full
    .explained_variance_ratio_
    * 100
)


cumexpl = np.cumsum(
    expl
)


print(
    "\n"
    + "=" * 70
)

print(
    "EXPLAINED VARIANCE"
)

print(
    "=" * 70
)


for i, value in enumerate(
    expl,
    start=1
):

    print(
        f"PC{i}: "
        f"{value:.2f}%"
    )


print(
    "\nCumulative explained variance:"
)


for i, value in enumerate(
    cumexpl,
    start=1
):

    print(
        f"PC1-PC{i}: "
        f"{value:.2f}%"
    )


print(
    f"\nPC1 + PC2 cumulative variance: "
    f"{cumexpl[1]:.2f}%"
)


print(
    f"PC1-PC4 cumulative variance: "
    f"{cumexpl[3]:.2f}%"
)


# =============================================================================
# 14) SAVE EXPLAINED-VARIANCE TABLE
# =============================================================================

variance_df = pd.DataFrame(
    {
        "Principal_Component":
            [
                f"PC{i}"
                for i in range(
                    1,
                    len(expl) + 1
                )
            ],

        "Variance_Explained_Percent":
            expl,

        "Cumulative_Variance_Percent":
            cumexpl,
    }
)


variance_output_path = (
    OUTPUT_DIR
    / "PCA_explained_variance_log10_ra.csv"
)


variance_df.to_csv(
    variance_output_path,
    index=False
)


# =============================================================================
# 15) SCREE PLOT
# =============================================================================

plt.figure(
    figsize=(
        7,
        4
    )
)


plt.plot(
    np.arange(
        1,
        len(expl) + 1
    ),
    expl,
    marker="o"
)


plt.xlabel(
    "Principal component",
    fontsize=13
)


plt.ylabel(
    "Variance explained (%)",
    fontsize=13
)


plt.title(
    "PCA scree plot",
    fontsize=15
)


plt.xticks(
    np.arange(
        1,
        len(expl) + 1
    ),
    fontsize=11
)


plt.yticks(
    fontsize=11
)


plt.grid(
    True,
    linewidth=0.3,
    alpha=0.5
)


plt.tight_layout()


scree_png_path = (
    OUTPUT_DIR
    / "PCA_scree_log10_ra.png"
)


scree_pdf_path = (
    OUTPUT_DIR
    / "PCA_scree_log10_ra.pdf"
)


plt.savefig(
    scree_png_path,
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    scree_pdf_path,
    bbox_inches="tight"
)


plt.show()


# =============================================================================
# 16) CUMULATIVE VARIANCE PLOT
# =============================================================================

plt.figure(
    figsize=(
        7,
        4
    )
)


plt.plot(
    np.arange(
        1,
        len(cumexpl) + 1
    ),
    cumexpl,
    marker="o"
)


plt.xlabel(
    "Principal component",
    fontsize=13
)


plt.ylabel(
    "Cumulative variance explained (%)",
    fontsize=13
)


plt.title(
    "PCA cumulative variance explained",
    fontsize=15
)


plt.xticks(
    np.arange(
        1,
        len(cumexpl) + 1
    ),
    fontsize=11
)


plt.yticks(
    fontsize=11
)


plt.grid(
    True,
    linewidth=0.3,
    alpha=0.5
)


plt.tight_layout()


cumulative_png_path = (
    OUTPUT_DIR
    / "PCA_cumulative_variance_log10_ra.png"
)


cumulative_pdf_path = (
    OUTPUT_DIR
    / "PCA_cumulative_variance_log10_ra.pdf"
)


plt.savefig(
    cumulative_png_path,
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    cumulative_pdf_path,
    bbox_inches="tight"
)


plt.show()


# =============================================================================
# 17) GENERATE PCA SCORES
# =============================================================================

scores_4d = (
    pca_full
    .transform(
        X_log
    )[
        :,
        :4
    ]
)


pc1_var = expl[0]
pc2_var = expl[1]
pc3_var = expl[2]
pc4_var = expl[3]

cum4_var = cumexpl[3]


print(
    f"\nRetained first four PCs. "
    f"Cumulative variance = "
    f"{cum4_var:.2f}%"
)


# =============================================================================
# 18) ALIGN METADATA
# =============================================================================

meta = (
    df[
        [
            sample_col,
            depth_col
        ]
    ]
    .drop_duplicates()
    .set_index(
        sample_col
    )
)


meta = meta.loc[
    X.index
]


plot_df = meta.copy()


plot_df[
    "PC1"
] = scores_4d[
    :,
    0
]


plot_df[
    "PC2"
] = scores_4d[
    :,
    1
]


plot_df[
    "PC3"
] = scores_4d[
    :,
    2
]


plot_df[
    "PC4"
] = scores_4d[
    :,
    3
]


# =============================================================================
# 19) SAVE PCA SCORES
# =============================================================================

scores_output = (
    plot_df
    .reset_index()
)


scores_output_path = (
    OUTPUT_DIR
    / "PCA_scores_log10_ra.csv"
)


scores_output.to_csv(
    scores_output_path,
    index=False
)


# =============================================================================
# 20) PCA LOADINGS
# =============================================================================

loadings_pc1 = pd.Series(
    pca_full.components_[0],
    index=X.columns
)


loadings_pc2 = pd.Series(
    pca_full.components_[1],
    index=X.columns
)


loadings_pc3 = pd.Series(
    pca_full.components_[2],
    index=X.columns
)


loadings_pc4 = pd.Series(
    pca_full.components_[3],
    index=X.columns
)


# =============================================================================
# 21) SELECT TOP 20 PC1 CONTRIBUTORS
# =============================================================================

top_phyla = (
    loadings_pc1
    .abs()
    .sort_values(
        ascending=False
    )
    .head(
        TOP_N_LOADINGS
    )
    .index
)


top_df = pd.DataFrame(
    {
        "PC1_loading":
            loadings_pc1.loc[
                top_phyla
            ],

        "PC2_loading":
            loadings_pc2.loc[
                top_phyla
            ],

        "PC3_loading":
            loadings_pc3.loc[
                top_phyla
            ],

        "PC4_loading":
            loadings_pc4.loc[
                top_phyla
            ],
    }
)


top_df[
    "Magnitude_PC1_PC2"
] = np.sqrt(
    top_df[
        "PC1_loading"
    ] ** 2
    +
    top_df[
        "PC2_loading"
    ] ** 2
)


top_df = (
    top_df
    .sort_values(
        "Magnitude_PC1_PC2",
        ascending=False
    )
)


# =============================================================================
# 22) DETERMINE UPPER/LOWER PC1 DIRECTION
# =============================================================================

depth_means = (
    plot_df
    .groupby(
        depth_col
    )[
        "PC1"
    ]
    .mean()
)


if not {
    "U",
    "L"
}.issubset(
    depth_means.index
):

    raise ValueError(
        f"Expected U and L in {depth_col}. "
        f"Found: {list(depth_means.index)}"
    )


upper_on_positive = (
    depth_means[
        "U"
    ]
    >
    depth_means[
        "L"
    ]
)


top_df[
    "DepthAssoc"
] = np.where(
    top_df[
        "PC1_loading"
    ] > 0,

    (
        "U"
        if upper_on_positive
        else "L"
    ),

    (
        "L"
        if upper_on_positive
        else "U"
    )
)


print(
    "\nMean PC1 score by depth:"
)


print(
    depth_means
)


# =============================================================================
# 23) SAVE LOADINGS TABLE
# =============================================================================

loadings_output_path = (
    OUTPUT_DIR
    / "PCA_top_20_phylum_loadings_log10_ra.csv"
)


top_df.to_csv(
    loadings_output_path
)


# =============================================================================
# 24) COLOR MAP
# =============================================================================

color_map = {
    "U": "#E69F00",
    "L": "#0072B2"
}


# =============================================================================
# 25) PLOT SETTINGS
# =============================================================================

TITLE_SIZE = 16

LABEL_SIZE = 14

TICK_SIZE = 12

LEGEND_SIZE = 11

LEGEND_TITLE_SIZE = 12

TAXA_LABEL_SIZE = 11

SCORE_MARKER_SIZE = 50

LOADING_MARKER_SIZE = 110


# =============================================================================
# 26) CREATE PCA SCORES + LOADINGS FIGURE
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        17,
        8
    ),
    gridspec_kw={
        "width_ratios": [
            1.0,
            1.35
        ]
    }
)


# =============================================================================
# 27) LEFT PANEL — PCA SCORES
# =============================================================================

sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue=depth_col,
    hue_order=[
        "U",
        "L"
    ],
    palette=color_map,
    s=SCORE_MARKER_SIZE,
    edgecolor="k",
    linewidth=0.25,
    alpha=0.85,
    ax=axes[0]
)


axes[0].axhline(
    0,
    color="grey",
    lw=0.6
)


axes[0].axvline(
    0,
    color="grey",
    lw=0.6
)


axes[0].set_title(
    "PCA Scores (PC1 vs PC2)",
    fontsize=TITLE_SIZE
)


axes[0].set_xlabel(
    f"PC1 ({pc1_var:.1f}% variance)",
    fontsize=LABEL_SIZE
)


axes[0].set_ylabel(
    f"PC2 ({pc2_var:.1f}% variance)",
    fontsize=LABEL_SIZE
)


axes[0].tick_params(
    axis="both",
    labelsize=TICK_SIZE
)


handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)


label_map = {
    "U": "Upper",
    "L": "Lower"
}


new_labels = [
    label_map.get(
        label,
        label
    )
    for label in labels
]


axes[0].legend(
    handles=handles,
    labels=new_labels,
    title="Soil Depth",
    loc="upper left",
    frameon=True,
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_TITLE_SIZE
)


# =============================================================================
# 28) RIGHT PANEL — LOADINGS
# =============================================================================

axes[1].scatter(
    top_df[
        "PC1_loading"
    ],
    top_df[
        "PC2_loading"
    ],
    c=top_df[
        "DepthAssoc"
    ].map(
        color_map
    ),
    s=LOADING_MARKER_SIZE,
    edgecolor="k",
    linewidth=0.25,
    alpha=0.92,
    zorder=3
)


axes[1].set_xlim(
    -0.65,
    0.90
)


axes[1].set_ylim(
    -0.45,
    0.78
)


axes[1].axhline(
    0,
    color="grey",
    lw=0.6,
    zorder=1
)


axes[1].axvline(
    0,
    color="grey",
    lw=0.6,
    zorder=1
)


# =============================================================================
# 29) IDENTIFY CENTRAL / CROWDED TAXA
# =============================================================================

central_x_cut = 0.25

central_y_cut = 0.14


central_mask = (
    top_df[
        "PC1_loading"
    ].abs()
    <= central_x_cut
) & (
    top_df[
        "PC2_loading"
    ].abs()
    <= central_y_cut
)


central_df = (
    top_df.loc[
        central_mask
    ]
    .copy()
)


outer_df = (
    top_df.loc[
        ~central_mask
    ]
    .copy()
)


# =============================================================================
# 30) CENTRAL LABEL GROUPS
# =============================================================================

central_left = (
    central_df[
        central_df[
            "PC1_loading"
        ] <= 0
    ]
    .sort_values(
        "PC2_loading",
        ascending=False
    )
)


central_right = (
    central_df[
        central_df[
            "PC1_loading"
        ] > 0
    ]
    .sort_values(
        "PC2_loading",
        ascending=False
    )
)


LEFT_LABEL_X = -0.34

RIGHT_LABEL_X = 0.36


def evenly_spaced_positions(
    n,
    top,
    bottom
):

    if n == 0:
        return np.array([])

    if n == 1:
        return np.array(
            [
                (
                    top
                    + bottom
                )
                / 2
            ]
        )

    return np.linspace(
        top,
        bottom,
        n
    )


left_y_positions = evenly_spaced_positions(
    len(
        central_left
    ),
    0.16,
    -0.18
)


right_y_positions = evenly_spaced_positions(
    len(
        central_right
    ),
    0.16,
    -0.21
)


# =============================================================================
# 31) CENTRAL LEFT LABELS
# =============================================================================

for (
    (phylum, row),
    label_y
) in zip(
    central_left.iterrows(),
    left_y_positions
):

    point_x = float(
        row[
            "PC1_loading"
        ]
    )

    point_y = float(
        row[
            "PC2_loading"
        ]
    )


    axes[1].annotate(
        str(
            phylum
        ),
        xy=(
            point_x,
            point_y
        ),
        xytext=(
            LEFT_LABEL_X,
            float(
                label_y
            )
        ),
        textcoords="data",
        ha="right",
        va="center",
        fontsize=TAXA_LABEL_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.10",
            fc="white",
            ec="none",
            alpha=0.90
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="0.65",
            lw=0.6,
            alpha=0.75,
            shrinkA=2,
            shrinkB=3
        ),
        zorder=4
    )


# =============================================================================
# 32) CENTRAL RIGHT LABELS
# =============================================================================

for (
    (phylum, row),
    label_y
) in zip(
    central_right.iterrows(),
    right_y_positions
):

    point_x = float(
        row[
            "PC1_loading"
        ]
    )

    point_y = float(
        row[
            "PC2_loading"
        ]
    )


    axes[1].annotate(
        str(
            phylum
        ),
        xy=(
            point_x,
            point_y
        ),
        xytext=(
            RIGHT_LABEL_X,
            float(
                label_y
            )
        ),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=TAXA_LABEL_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.10",
            fc="white",
            ec="none",
            alpha=0.90
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="0.65",
            lw=0.6,
            alpha=0.75,
            shrinkA=2,
            shrinkB=3
        ),
        zorder=4
    )


# =============================================================================
# 33) OUTER TAXA LABELS
# =============================================================================

for phylum, row in outer_df.iterrows():

    x = float(
        row[
            "PC1_loading"
        ]
    )

    y = float(
        row[
            "PC2_loading"
        ]
    )


    if x >= 0:

        dx = 8

        ha = "left"

    else:

        dx = -8

        ha = "right"


    if y >= 0:

        dy = 6

        va = "bottom"

    else:

        dy = -6

        va = "top"


    axes[1].annotate(
        str(
            phylum
        ),
        xy=(
            x,
            y
        ),
        xytext=(
            dx,
            dy
        ),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=TAXA_LABEL_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.10",
            fc="white",
            ec="none",
            alpha=0.90
        ),
        zorder=4
    )


# =============================================================================
# 34) LOADINGS AXES
# =============================================================================

axes[1].set_title(
    "PCA Loadings (PC1 vs PC2)",
    fontsize=TITLE_SIZE
)


axes[1].set_xlabel(
    f"PC1 loading ({pc1_var:.1f}% variance)",
    fontsize=LABEL_SIZE
)


axes[1].set_ylabel(
    f"PC2 loading ({pc2_var:.1f}% variance)",
    fontsize=LABEL_SIZE
)


axes[1].tick_params(
    axis="both",
    labelsize=TICK_SIZE
)


# =============================================================================
# 35) LOADINGS LEGEND
# =============================================================================

taxa_legend = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        label="Upper-associated",
        markerfacecolor=color_map[
            "U"
        ],
        markeredgecolor="k",
        markersize=8
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        label="Lower-associated",
        markerfacecolor=color_map[
            "L"
        ],
        markeredgecolor="k",
        markersize=8
    )
]


axes[1].legend(
    handles=taxa_legend,
    title="Soil Depth",
    loc="upper right",
    frameon=True,
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_TITLE_SIZE
)


# =============================================================================
# 36) FINAL LAYOUT
# =============================================================================

plt.tight_layout()


plt.subplots_adjust(
    wspace=0.20
)


# =============================================================================
# 37) SAVE MAIN PCA FIGURE
# =============================================================================

pca_png_path = (
    OUTPUT_DIR
    / "PCA_PC1_PC2_expanded_loadings_log10_ra.png"
)


pca_pdf_path = (
    OUTPUT_DIR
    / "PCA_PC1_PC2_expanded_loadings_log10_ra.pdf"
)


plt.savefig(
    pca_png_path,
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    pca_pdf_path,
    bbox_inches="tight"
)


plt.show()


# =============================================================================
# 38) DIAGNOSTICS
# =============================================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PCA DIAGNOSTICS — LOG10 RELATIVE ABUNDANCE"
)

print(
    "=" * 70
)


print(
    "\nMean PC1 by soil depth:"
)


print(
    depth_means
)


print(
    "\nTop PC1 loadings:"
)


print(
    loadings_pc1
    .sort_values(
        key=np.abs,
        ascending=False
    )
    .head(
        30
    )
)


print(
    "\nTop PC2 loadings:"
)


print(
    loadings_pc2
    .sort_values(
        key=np.abs,
        ascending=False
    )
    .head(
        30
    )
)


# =============================================================================
# 39) FINAL OUTPUT SUMMARY
# =============================================================================

print(
    "\n"
    + "=" * 78
)

print(
    "PCA ANALYSIS COMPLETE"
)

print(
    "=" * 78
)


print(
    "\nTransformation:"
)

print(
    "  log10(relative abundance + 1e-6)"
)


print(
    "\nSaved files:"
)


print(
    f"  1. {pca_png_path}"
)

print(
    f"  2. {pca_pdf_path}"
)

print(
    f"  3. {loadings_output_path}"
)

print(
    f"  4. {scores_output_path}"
)

print(
    f"  5. {variance_output_path}"
)

print(
    f"  6. {scree_png_path}"
)

print(
    f"  7. {scree_pdf_path}"
)

print(
    f"  8. {cumulative_png_path}"
)

print(
    f"  9. {cumulative_pdf_path}"
)


print(
    "=" * 78
)
