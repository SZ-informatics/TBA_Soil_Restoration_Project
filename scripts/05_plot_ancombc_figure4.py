"""
Regenerate Figure 4 (ANCOM-BC differential abundance, lower vs. upper soil)
directly from the ANCOM-BC output slices, so the figure is guaranteed to match
the run it is plotted from.

Inputs (the ANCOM-BC "slice" CSVs; each has phyla in rows and model terms in
columns, with the depth contrast in the column 'SoilProfileL'):
    - lfc_slice.csv    log-fold changes
    - se_slice.csv     standard errors  (used for the error bars)
    - q_val_slice.csv  BH-adjusted q-values (used for the significance filter)

Convention: SoilProfileL is the "Lower vs Upper" contrast with Upper as the
reference, so a POSITIVE LFC = enriched in the LOWER profile (15-30 cm) and a
NEGATIVE LFC = enriched in the UPPER profile (0-15 cm).

Edit the CONFIG block for your paths / thresholds and run:  python plot_ancombc_figure4.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------- CONFIG -------------------------------------
LFC_CSV   = "lfc_slice.csv"
SE_CSV    = "se_slice.csv"
QVAL_CSV  = "q_val_slice.csv"

CONTRAST_COL       = "SoilProfileL"   # the depth column in the slice files
Q_THRESHOLD        = 0.05             # significance cutoff
TOP_N              = 20               # number of phyla to display (by |LFC|)
EXCLUDE_UNCLASSIFIED = True           # drop phyla with no phylum-level name
EXCLUDE_TAXA       = ["SAR324_clade(Marine_group_B)"]  # phyla to hide by name
OUTFILE            = "Figure4_ANCOMBC.png"

COLOR_LOWER = "#1f77b4"   # blue  -> enriched in lower soil
COLOR_UPPER = "#ff7f0e"   # orange-> enriched in upper soil
# --------------------------------------------------------------------------


def phylum_name(raw_id):
    """Extract a clean phylum label from a SILVA-style id like
    'd__Bacteria;p__Zixibacteria'. Returns None if there is no p__ name."""
    if "p__" in raw_id:
        name = raw_id.split("p__")[-1].strip()
        return name if name else None
    return None


def load_slice(path, value_name):
    df = pd.read_csv(path)
    df["phylum"] = df["id"].apply(phylum_name)
    out = df[["id", "phylum", CONTRAST_COL]].rename(columns={CONTRAST_COL: value_name})
    return out


# ---- assemble a tidy table: phylum | lfc | se | q -------------------------
lfc = load_slice(LFC_CSV,  "lfc")
se  = load_slice(SE_CSV,   "se")[["id", "se"]]
q   = load_slice(QVAL_CSV, "q")[["id", "q"]]

data = lfc.merge(se, on="id").merge(q, on="id")

# significance filter
data = data[data["q"] < Q_THRESHOLD].copy()

# optionally drop unnamed phyla (e.g. 'd__Bacteria;__')
if EXCLUDE_UNCLASSIFIED:
    data = data[data["phylum"].notna()].copy()

# drop any phyla listed in EXCLUDE_TAXA (matched by phylum name)
if EXCLUDE_TAXA:
    data = data[~data["phylum"].isin(EXCLUDE_TAXA)].copy()

# keep the TOP_N by absolute effect size, then order for plotting
data["abs_lfc"] = data["lfc"].abs()
data = data.sort_values("abs_lfc", ascending=False).head(TOP_N)
data = data.sort_values("lfc", ascending=True)          # most-upper at bottom
data = data.reset_index(drop=True)

colors = np.where(data["lfc"] > 0, COLOR_LOWER, COLOR_UPPER)
labels = ["p__" + p for p in data["phylum"]]

# ------------------------------ plot --------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))

ax.barh(
    y=np.arange(len(data)),
    width=data["lfc"],
    xerr=data["se"],
    color=colors,
    edgecolor="none",
    error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2.5),
)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(np.arange(len(data)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Log Fold Change (Lower vs Upper soil)", fontsize=11)
ax.set_ylabel("Phylum", fontsize=11)
ax.set_title("Differentially Abundant Taxa Across Soil Depth\n(ANCOM-BC; $q < 0.05$)",
             fontsize=12)

legend_handles = [
    Patch(facecolor=COLOR_LOWER, label="Enriched in lower soil"),
    Patch(facecolor=COLOR_UPPER, label="Enriched in upper soil"),
]
ax.legend(handles=legend_handles, title="Depth association",
          loc="lower right", frameon=True, fontsize=9, title_fontsize=9)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTFILE} with {len(data)} phyla "
      f"({int((data['lfc']>0).sum())} lower, {int((data['lfc']<0).sum())} upper).")
