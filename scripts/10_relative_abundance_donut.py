# =========================================================
# Figure 3 - Relative abundance of microbial phyla by soil depth
# Nested donut: inner ring = Lower soil (15-30 cm), outer ring = Upper soil (0-15 cm).
# Top 20 phyla shown individually; the remainder pooled as "Others".
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt

# === Load and process data ===
df = pd.read_excel("data/Allmerged_16Slevel-2.xlsx")

metadata_cols = ['index', 'BarcodeSequence', 'SampleCode', 'LinkerPrimerSequence',
                 'Time Period', 'Species', 'Amendment', 'SoilProfile', 'Block', 'Group1', 'Group2']
taxa_cols = [col for col in df.columns if col not in metadata_cols]

# Melt and extract Phylum (p__)
df_melted = df.melt(id_vars='SoilProfile', value_vars=taxa_cols,
                    var_name='Taxon', value_name='Abundance')
df_melted[['Phylum']] = df_melted['Taxon'].str.extract(r'd__.*?;p__(?P<Phylum>[^;]+)')
df_melted = df_melted.dropna(subset=['Phylum'])
df_melted = df_melted[df_melted['Abundance'] > 0]

# Group, summarize, and compute relative abundance at Phylum level
df_phy = df_melted.groupby(['SoilProfile', 'Phylum'])['Abundance'].sum().reset_index()
df_phy['Total'] = df_phy.groupby('Phylum')['Abundance'].transform('sum')

# Keep top 20 phyla and lump the rest as "Others"
top_phyla = df_phy.groupby('Phylum')['Total'].sum().sort_values(ascending=False).head(20).index.tolist()
df_phy['Phylum'] = df_phy['Phylum'].apply(lambda x: x if x in top_phyla else 'Others')
df_phy = df_phy.groupby(['SoilProfile', 'Phylum'])['Abundance'].sum().reset_index()
df_phy['Relative'] = df_phy.groupby('SoilProfile')['Abundance'].transform(lambda x: x / x.sum())

# Sort phyla, move Others to end
phyla_sorted = sorted([p for p in df_phy['Phylum'].unique() if p != 'Others'])
plot_phyla = phyla_sorted + ['Others']

# Inner ring = LOWER soil, outer ring = UPPER soil
inner = df_phy[df_phy['SoilProfile'] == 'L'].set_index('Phylum').reindex(plot_phyla).fillna(0)
outer = df_phy[df_phy['SoilProfile'] == 'U'].set_index('Phylum').reindex(plot_phyla).fillna(0)

# Shared palette so a phylum has the same colour in both rings
n_classes = len(plot_phyla)
all_colors = list(plt.colormaps['tab20'].colors) + list(plt.colormaps['tab20b'].colors)
earthy_colors = all_colors[:n_classes]

# === Colours ===
BG  = "#F3F2F9"   # background
INK = "#222428"   # text
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300})

# === Plot ===
fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Inner = LOWER soil
wedges_inner, _, autotexts_inner = ax.pie(
    inner['Relative'], radius=0.7, labels=None, colors=earthy_colors,
    wedgeprops=dict(width=0.3, edgecolor=BG, linewidth=1.2),
    autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '')

# Outer = UPPER soil
wedges_outer, _, autotexts_outer = ax.pie(
    outer['Relative'], radius=1.0, labels=None, colors=earthy_colors,
    wedgeprops=dict(width=0.3, edgecolor=BG, linewidth=1.2),
    autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '')

for t in [*autotexts_inner, *autotexts_outer]:
    t.set_fontsize(9); t.set_color(INK); t.set_weight('bold')
for t in autotexts_outer:
    x, y = t.get_position()
    t.set_position((x*1.35, y*1.35))

ax.add_artist(plt.Circle((0, 0), 0.4, color=BG))
ax.text(0., 0.30, 'Lower\nSoil', ha='center', va='center', fontsize=12, weight='bold', color=INK)
ax.text(0, -1.15, 'Upper\nSoil', ha='center', va='center', fontsize=15, weight='bold', color=INK)

leg = ax.legend(wedges_outer, plot_phyla, title="Phylum (Top + Others)",
                bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
leg.get_frame().set_facecolor(BG); leg.get_frame().set_edgecolor('none')
leg.get_title().set_fontweight('bold')
for txt in leg.get_texts():
    txt.set_color(INK)

ax.set_title("Relative Abundance of Microbial Phyla\nLower (Inner) vs Upper (Outer)",
             fontsize=14, fontweight='bold', color=INK, pad=6)

plt.tight_layout()
plt.savefig("Figure3_phyla_donut.png", bbox_inches="tight", facecolor=BG)
print("Saved Figure3_phyla_donut.png")
