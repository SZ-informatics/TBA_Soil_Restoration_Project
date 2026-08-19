# =========================================================
# Stratified robustness check for depth classification
# ---------------------------------------------------------
# Trains the SAME best Random Forest depth classifier separately within
# each plant system and within each amendment, using group-aware CV, to
# confirm the depth signal (and its top taxa) is not treatment-specific.
#
# Outputs:
#   * stratified_results.csv          -> per-stratum n, CV accuracy, CV AUC, top taxa
#   * Fig_stratified_importance.png/.pdf -> 2x3 panel:
#       - top row = plant systems, bottom row = amendments
#       - bar length = Random Forest importance
#       - colour    = soil layer of enrichment (blue lower, amber upper)
#       - each panel shows the top K_LOWER lower-enriched + K_UPPER upper-enriched phyla
#
# Requires: pandas numpy scikit-learn matplotlib openpyxl pillow
# =========================================================
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")   # remove for interactive windows
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------- config ----------------
RANDOM_STATE = 42
EXCEL_PATH   = "data/Allmerged_16Slevel-2.xlsx"   # phylum-level counts + metadata
K_LOWER      = 6        # lower-soil-enriched phyla shown per panel
K_UPPER      = 6        # upper-soil-enriched phyla shown per panel
MAX_SPLITS   = 5        # CV folds within each (smaller) stratum
LOWER_COLOR  = "#2c6fbb"   # blue  = enriched in lower soil (15-30 cm)
UPPER_COLOR  = "#e08214"   # amber = enriched in upper soil (0-15 cm)
np.random.seed(RANDOM_STATE)
print(f"scikit-learn {sklearn.__version__}")

# ---------------- load + preprocess (same as the main pipeline) ----------------
df = pd.read_excel(EXCEL_PATH, index_col=0)
df = df.loc[:, ~df.columns.str.startswith('d__Archaea;')]
df.columns = df.columns.str.replace(r'^d__Bacteria;', '', regex=True)
meta = ['#SampleID','index','SampleCode','LinkerPrimerSequence','Time Period',
        'Species','Amendment','SoilProfile','Block','Group1','Group2','Group3']
taxa_cols = [c for c in df.columns if c not in meta]
counts = df[taxa_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
counts = counts.loc[:, (counts >= 2).sum(axis=0) >= 2]           # >=2 reads in >=2 samples
feats  = [f.replace('p__','').replace('c__','') for f in counts.columns]
X      = counts.values
profile = df['SoilProfile'].astype(str).values                  # 'U' / 'L'
y = LabelEncoder().fit_transform(profile)
groups = (df['Species'].astype(str)+'_'+df['Amendment'].astype(str)+'_'+df['Block'].astype(str)).values
species, amend = df['Species'].astype(str).values, df['Amendment'].astype(str).values
logX = np.log(X + 1.0)                                            # for direction of enrichment

def log_transform(A, pc=1.0): return np.log(np.asarray(A, float) + pc)

def best_rf():
    # best Random Forest configuration from the pooled GridSearchCV
    return Pipeline([
        ("log", FunctionTransformer(log_transform, validate=False)),
        ("rf", RandomForestClassifier(
            n_estimators=70, criterion="entropy", max_features="sqrt",
            min_samples_split=2, min_samples_leaf=5, bootstrap=True,
            max_depth=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1))])

# strata: 3 plant systems (top row) + 3 amendments (bottom row)
strata = [("Switchgrass",       species == "Native Grass"),
          ("Perennial Sorghum", species == "PS"),
          ("Annual Sorghum",    species == "SOBI"),
          ("Control",           amend == "Control"),
          ("Biochar",           amend == "Biochar"),
          ("TBA",               amend == "TBA")]

rows, panels = [], []
for label, mask in strata:
    Xs, ys, gs, ps, lx = X[mask], y[mask], groups[mask], profile[mask], logX[mask]
    n_splits = min(MAX_SPLITS, len(np.unique(gs)), np.bincount(ys).min())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cvr = cross_validate(best_rf(), Xs, ys, groups=gs, cv=cv,
                         scoring=["accuracy", "roc_auc"], n_jobs=-1)
    acc, auc = cvr["test_accuracy"].mean(), cvr["test_roc_auc"].mean()

    m = best_rf(); m.fit(Xs, ys)
    imp = m.named_steps["rf"].feature_importances_
    diff = lx[ps == "L"].mean(0) - lx[ps == "U"].mean(0)          # >0 => lower-enriched
    lower = [i for i in np.argsort(imp)[::-1] if diff[i] > 0][:K_LOWER]
    upper = [i for i in np.argsort(imp)[::-1] if diff[i] <= 0][:K_UPPER]
    idx = lower + upper
    panels.append({"label": label, "n": int(mask.sum()), "acc": acc, "auc": auc,
                   "taxa": [feats[i] for i in idx], "imp": [imp[i] for i in idx],
                   "col": [LOWER_COLOR]*len(lower) + [UPPER_COLOR]*len(upper)})
    rows.append({"stratum": label, "n": int(mask.sum()), "cv_accuracy": round(acc,3),
                 "cv_auc": round(auc,3),
                 "top_lower": "; ".join(feats[i] for i in lower),
                 "top_upper": "; ".join(feats[i] for i in upper)})
    print(f"{label:18s} n={mask.sum():3d}  CV acc={acc:.3f}  CV AUC={auc:.3f}")

pd.DataFrame(rows).to_csv("stratified_results.csv", index=False)

# ---------------- 2x3 panel figure ----------------
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
for r, ax in zip(panels, axes.ravel()):
    taxa, vals, cols = r["taxa"][::-1], r["imp"][::-1], r["col"][::-1]
    ax.barh(range(len(taxa)), vals, color=cols, edgecolor="0.25", linewidth=0.4)
    ax.set_yticks(range(len(taxa))); ax.set_yticklabels(taxa, fontsize=9)
    ax.set_xlabel("Random Forest importance", fontsize=9)
    ax.set_title(f"{r['label']}  (n={r['n']}; Acc {r['acc']:.2f}, AUC {r['auc']:.2f})",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', labelsize=8); ax.grid(axis='x', alpha=0.3)
fig.suptitle("Depth-predictive taxa are consistent across plant systems and amendments\n"
             f"(top {K_LOWER} lower- and top {K_UPPER} upper-enriched phyla per stratum; "
             "bar length = Random Forest importance)", fontsize=14, fontweight="bold")
legend = [Patch(facecolor=LOWER_COLOR, edgecolor="0.25", label="Enriched in lower soil (15–30 cm)"),
          Patch(facecolor=UPPER_COLOR, edgecolor="0.25", label="Enriched in upper soil (0–15 cm)")]
fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=12,
           frameon=True, bbox_to_anchor=(0.5, -0.005))
plt.tight_layout(rect=[0, 0.035, 1, 0.94])
plt.savefig("Fig_stratified_importance.png", dpi=300, bbox_inches="tight")
plt.savefig("Fig_stratified_importance.pdf", bbox_inches="tight")
print("\nSaved: stratified_results.csv, Fig_stratified_importance.png/.pdf")
