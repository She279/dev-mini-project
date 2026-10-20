# %% [markdown]
# # Exploratory Data Analysis and Visualization of the Auto MPG Dataset
# Capstone project - Data Exploration and Visualization (Units 1-5)
#
# How to run in VS Code:
#   1. pip install -r requirements.txt
#   2. Open this file; click "Run Cell" above any `# %%` line (Jupyter extension),
#      or run the whole file:  python autompg_eda.py
# Figures are saved in ./figures, tables in ./outputs/tables,
# interactive HTML in ./outputs/interactive.

# %% Setup
import os
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

SHOW = True  # set False to only save figures (no pop-up windows)
for d in ["data", "figures", "outputs/tables", "outputs/interactive"]:
    os.makedirs(d, exist_ok=True)

PAL = {"USA": "#F28E2B", "Europe": "#B07AA1", "Japan": "#4E79A7"}
ORIGINS = ["USA", "Europe", "Japan"]


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)


def save_table(df, name):
    df.to_csv(f"outputs/tables/{name}.csv")


# %% [markdown]
# # UNIT 1 - Exploratory Data Analysis

# %% 3.2 Data loading and initial inspection
CSV = "data/mpg.csv"
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
if not os.path.exists(CSV):
    print("Downloading dataset ...")
    urllib.request.urlretrieve(URL, CSV)

raw = pd.read_csv(CSV)
raw["origin"] = raw["origin"].str.lower().map({"usa": "USA", "europe": "Europe", "japan": "Japan"})
names = raw["name"]
df = raw.drop(columns="name")

print(df.shape)  # (398, 8)
df.info()
print(df.describe().round(2))
print(df.head())

num_cols = ["mpg", "horsepower", "weight", "acceleration"]
print("\ndisplacement vs weight r =", round(df.displacement.corr(df.weight), 2),
      "| vs horsepower r =", round(df.displacement.corr(df.horsepower), 2))

# %% 3.3 Missing values and duplicates
missing = df.isna().sum()
print("Missing per column:\n", missing)
print("Rows with missing:", df.isna().any(axis=1).sum(), "| duplicates:", df.duplicated().sum())
print(df[df.horsepower.isna()].origin.value_counts())

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=missing.index, y=missing.values, color="#E15759", ax=ax[0])
ax[0].set_title("Missing values per column"); ax[0].set_ylabel("count")
ax[0].tick_params(axis="x", rotation=45)
sns.heatmap(df.isna(), cbar=False, cmap=["#fdf0ee", "#E15759"], yticklabels=False, ax=ax[1])
ax[1].set_title("Missing-value map (red = missing)")
save(fig, "fig3_2_missing")

# Compare imputation vs dropping
imputed = df.copy()
imputed["horsepower"] = imputed.groupby("origin")["horsepower"].transform(lambda s: s.fillna(s.median()))
print(f"Mean horsepower: before {df.horsepower.mean():.2f} -> after imputation {imputed.horsepower.mean():.2f}")

df_clean = df.dropna().reset_index(drop=True)
df_clean["period"] = np.where(df_clean.model_year <= 76, "Early", "Late")
print("Clean shape:", df_clean.shape, "| missing:", df_clean.isna().sum().sum())
df_clean.to_csv("data/mpg_clean.csv", index=False)

# %% 3.4 Encoding and slicing
enc = df_clean.copy()
enc["origin_code"] = LabelEncoder().fit_transform(enc["origin"])  # Europe=0, Japan=1, USA=2
enc["period_code"] = enc["period"].map({"Early": 0, "Late": 1})
enc = pd.concat([enc, pd.get_dummies(enc["cylinders"], prefix="cyl", dtype=int)], axis=1)
print(enc.head())

print(df_clean.loc[0:4, ["origin", "cylinders", "weight"]])      # label-based
print(df_clean.iloc[10:15, 2:5])                                  # position-based
heavy_usa = df_clean[(df_clean.origin == "USA") & (df_clean.weight > 4500)]
print("Heavy US cars:", len(heavy_usa), "| cylinders:", heavy_usa.cylinders.unique())
efficient_jp = df_clean.query("origin == 'Japan' and cylinders == 4 and mpg > 36")
print("Efficient Japanese 4-cyl:", len(efficient_jp), "| min year:", efficient_jp.model_year.min())
print(df_clean.sample(5, random_state=42))

# %% 3.5 Transformations
w = df_clean["weight"]
transforms = {
    "Original (lb)": w,
    "Log transform": np.log(w),
    "Z-score standardisation": (w - w.mean()) / w.std(),
    "Min-Max scaling [0,1]": (w - w.min()) / (w.max() - w.min()),
}
fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
for ax, (t, s), c in zip(axes, transforms.items(), sns.color_palette("deep", 4)):
    sns.histplot(s, kde=True, color=c, ax=ax); ax.set_title(t); ax.set_xlabel("")
save(fig, "fig3_3_transformations")
print(f"Skewness weight: original {w.skew():.2f} -> log {np.log(w).skew():.2f}")

df_clean["weight_class"] = pd.qcut(df_clean.weight, 3, labels=["Light", "Medium", "Heavy"])
df_clean["power_to_weight"] = df_clean.horsepower / df_clean.weight * 1000
print(df_clean.weight_class.value_counts())
print("Mean power-to-weight (hp per 1000 lb):", round(df_clean.power_to_weight.mean(), 1))

# %% 3.6 Grouping
g = df_clean.groupby("origin")[num_cols].mean().round(2).loc[ORIGINS]
print(g); save_table(g, "table3_2_origin_means")
print(df_clean.pivot_table(values="mpg", index="origin", columns="cylinders", aggfunc="mean").round(1))
print(pd.crosstab(df_clean.origin, df_clean.cylinders, margins=True))

# %% [markdown]
# # UNIT 2 - Visualizing using Matplotlib

# %% 4.1 Line, scatter and error plots
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
s = df_clean.groupby("model_year")["mpg"].mean()
ax[0].plot(s.index, s.values, marker="o", ls="-", label="Mean mpg")
ax[0].plot(s.index, s.rolling(3, min_periods=1).mean(), "r--", label="3-year rolling mean")
ax[0].set(title="(a) Simple line plot", xlabel="Model year (19xx)", ylabel="Mean mpg"); ax[0].legend()

for o in ORIGINS:
    d = df_clean[df_clean.origin == o]
    ax[1].scatter(d.weight, d.mpg, c=PAL[o], s=d.horsepower / 3, alpha=0.7, label=o)
ax[1].set(title="(b) Scatter (size = horsepower)", xlabel="Weight (lb)", ylabel="mpg")
ax[1].legend(title="Origin")

st = df_clean.groupby("origin")["mpg"].agg(["mean", "std", "count"]).loc[ORIGINS]
ci = 1.96 * st["std"] / np.sqrt(st["count"])
xpos = np.arange(3)
ax[2].errorbar(xpos - 0.1, st["mean"], yerr=st["std"], fmt="o", capsize=7, label="Mean ± 1 SD")
ax[2].errorbar(xpos + 0.1, st["mean"], yerr=ci, fmt="s", color="crimson", capsize=3, label="Mean ± 95% CI")
ax[2].set_xticks(xpos); ax[2].set_xticklabels(ORIGINS); ax[2].set(title="(c) Error bars", ylabel="mpg")
ax[2].legend()
save(fig, "fig4_1_line_scatter_error")
print(st.round(2)); print("95% CI half-widths:", ci.round(2).to_dict())
print("mpg vs model_year r =", round(df_clean.mpg.corr(df_clean.model_year), 2))

# %% 4.2 Density, contour and histograms
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
hb = ax[0].hexbin(df_clean.weight, df_clean.mpg, gridsize=25, cmap="YlGnBu", mincnt=1)
fig.colorbar(hb, ax=ax[0], label="Count"); ax[0].set(title="(a) Density: hexbin", xlabel="Weight (lb)", ylabel="mpg")
sns.kdeplot(x=df_clean.weight, y=df_clean.mpg, fill=True, cmap="viridis", levels=15, thresh=0, ax=ax[1])
ax[1].set(title="(b) Contour plot (2-D KDE)", xlabel="Weight (lb)", ylabel="mpg")
for b, ls in zip([8, 20, 40], [":", "--", "-"]):
    ax[2].hist(df_clean.mpg, bins=b, histtype="step", ls=ls, lw=1.8, label=f"{b} bins")
ax[2].set(title="(c) Histogram: effect of bin count", xlabel="mpg", ylabel="Frequency"); ax[2].legend()
save(fig, "fig4_2_density_contour_hist")

# %% 4.3 Legends, colours, text and annotation
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for o in ORIGINS:
    d = df_clean[df_clean.origin == o]
    ax[0, 0].scatter(d.weight, d.horsepower, c=PAL[o], s=14, label=o)
ax[0, 0].legend(title="Origin", loc="upper left", frameon=True, shadow=True)
ax[0, 0].set(title="(a) Legend with title, location, shadow", xlabel="Weight (lb)", ylabel="Horsepower")

sc = ax[0, 1].scatter(df_clean.horsepower, df_clean.acceleration, c=df_clean.mpg, cmap="plasma", s=15)
fig.colorbar(sc, ax=ax[0, 1], label="mpg")
ax[0, 1].set(title="(b) Colour mapping ('plasma')", xlabel="Horsepower", ylabel="Acceleration (s)")

ax[1, 0].hist([df_clean[df_clean.origin == o].weight for o in ORIGINS], bins=15, stacked=True,
              color=[PAL[o] for o in ORIGINS], label=ORIGINS)
ax[1, 0].legend(title="Origin"); ax[1, 0].set(title="(c) Stacked histogram", xlabel="Weight (lb)", ylabel="Frequency")

ax[1, 1].scatter(df_clean.weight, df_clean.mpg, c=df_clean.origin.map(PAL), s=12)
best = df_clean.loc[df_clean.mpg.idxmax()]
mean_mpg = df_clean.mpg.mean()
ax[1, 1].annotate(f"Best: {best.mpg} mpg ({best.origin})", xy=(best.weight, best.mpg),
                  xytext=(best.weight + 700, best.mpg - 3), arrowprops=dict(arrowstyle="->"), fontweight="bold")
ax[1, 1].axhline(mean_mpg, ls="--", color="grey"); ax[1, 1].text(4500, mean_mpg + 0.6, f"Mean = {mean_mpg:.1f}")
ax[1, 1].axvspan(3500, 5200, alpha=0.1); ax[1, 1].text(4600, 44, "Heavy cars")
ax[1, 1].set(title="(d) Text, annotation, reference lines", xlabel="Weight (lb)", ylabel="mpg")
fig.tight_layout(); save(fig, "fig4_3_legends_colours_annotation")
print("Best car:", names[df.mpg.idxmax()], "| mean mpg:", round(mean_mpg, 2))

# %% 4.4 Subplots with GridSpec
fig = plt.figure(figsize=(9, 8))
gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)
axm = fig.add_subplot(gs[1:, :3])
axt = fig.add_subplot(gs[0, :3], sharex=axm)
axr = fig.add_subplot(gs[1:, 3], sharey=axm)
for o in ORIGINS:
    d = df_clean[df_clean.origin == o]
    axm.scatter(d.weight, d.mpg, c=PAL[o], s=14, alpha=0.8, label=o)
    axt.hist(d.weight, bins=15, color=PAL[o], alpha=0.7)
    axr.hist(d.mpg, bins=15, color=PAL[o], alpha=0.7, orientation="horizontal")
axm.set(xlabel="Weight (lb)", ylabel="mpg"); axm.legend(title="Origin")
plt.setp(axt.get_xticklabels(), visible=False); plt.setp(axr.get_yticklabels(), visible=False)
fig.suptitle("Subplots with GridSpec: scatter plot with marginal histograms", fontweight="bold")
save(fig, "fig4_4_gridspec")

# %% 4.5 Three-dimensional plotting
fig = plt.figure(figsize=(16, 7))
a1 = fig.add_subplot(121, projection="3d")
for o in ORIGINS:
    d = df_clean[df_clean.origin == o]
    a1.scatter(d.weight, d.horsepower, d.mpg, c=PAL[o], s=12, label=o)
a1.set(title="(a) 3-D scatter plot", xlabel="Weight (lb)", ylabel="Horsepower", zlabel="mpg"); a1.legend(title="Origin")

a2 = fig.add_subplot(122, projection="3d")
kde = gaussian_kde(np.vstack([df_clean.weight, df_clean.mpg]))
xg, yg = np.meshgrid(np.linspace(1500, 5500, 60), np.linspace(5, 50, 60))
zg = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
a2.plot_surface(xg, yg, zg * 1e4, cmap="viridis", linewidth=0)
a2.set(title="(b) 3-D surface of the 2-D KDE", xlabel="Weight (lb)", ylabel="mpg", zlabel="Density (x1e-4)")
save(fig, "fig4_5_3d")

# %% 4.6 Seaborn
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
sns.scatterplot(data=df_clean, x="weight", y="mpg", hue="origin", style="period", palette=PAL, ax=ax[0, 0])
ax[0, 0].set_title("(a) scatterplot: hue = origin, style = period")
sns.kdeplot(data=df_clean, x="mpg", hue="origin", fill=True, palette=PAL, ax=ax[0, 1])
ax[0, 1].set_title("(b) kdeplot by origin")
sns.regplot(data=df_clean, x="horsepower", y="mpg", scatter_kws=dict(s=10, alpha=0.6),
            line_kws=dict(color="red"), ax=ax[1, 0])
ax[1, 0].set_title("(c) regplot with 95% confidence band")
sns.histplot(data=df_clean, x="displacement", hue="origin", multiple="stack", palette=PAL, ax=ax[1, 1])
ax[1, 1].set_title("(d) histplot (stacked)")
fig.tight_layout(); save(fig, "fig4_6_seaborn")
lr = stats.linregress(df_clean.horsepower, df_clean.mpg)
print(f"Slope = {lr.slope:.3f} mpg/hp, R^2 = {lr.rvalue**2:.2f}")

# %% 4.6 Plotly and Bokeh (interactive HTML)
try:
    import plotly.express as px
    fig_p = px.scatter(df_clean, x="weight", y="mpg", color="origin", symbol="period",
                       size="acceleration", marginal_x="box", marginal_y="violin",
                       hover_data=["horsepower", "cylinders"], color_discrete_map=PAL)
    fig_p.write_html("outputs/interactive/plotly_scatter.html")
    fig3d = px.scatter_3d(df_clean, x="weight", y="horsepower", z="mpg", color="origin", color_discrete_map=PAL)
    fig3d.write_html("outputs/interactive/plotly_3d.html")
    print("Saved Plotly HTML files.")
except ImportError:
    print("plotly not installed - skipping")

try:
    from bokeh.plotting import figure, output_file, save as bokeh_save
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import factor_cmap
    src = ColumnDataSource(df_clean)
    output_file("outputs/interactive/bokeh_scatter.html")
    p = figure(width=760, height=480, tools="pan,wheel_zoom,box_zoom,reset,save",
               title="Auto MPG - weight vs mpg", x_axis_label="Weight (lb)", y_axis_label="mpg")
    p.scatter("weight", "mpg", source=src, legend_field="origin", size=7, alpha=0.7,
              fill_color=factor_cmap("origin", list(PAL.values()), list(PAL.keys())))
    p.add_tools(HoverTool(tooltips=[("Origin", "@origin"), ("mpg", "@mpg"), ("Weight", "@weight")]))
    bokeh_save(p)
    print("Saved Bokeh HTML file.")
except ImportError:
    print("bokeh not installed - skipping")

# %% [markdown]
# # UNIT 3 - Univariate Analysis

# %% 5.1 Numerical variables
rows = []
for c in num_cols:
    x = df_clean[c]; q1, q3 = x.quantile([.25, .75])
    rows.append([c, x.mean(), x.median(), x.std(), q1, q3, q3 - q1, x.skew(), x.kurt()])
summary = pd.DataFrame(rows, columns=["Variable", "Mean", "Median", "Std", "Q1", "Q3", "IQR",
                                      "Skewness", "Kurtosis"]).set_index("Variable").round(2)
print(summary); save_table(summary, "table5_1_summary")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, c, col in zip(axes.ravel(), num_cols, sns.color_palette("deep", 4)):
    sns.histplot(df_clean[c], kde=True, color=col, ax=ax)
    ax.axvline(df_clean[c].mean(), color="red", ls="--", label=f"Mean {df_clean[c].mean():.1f}")
    ax.axvline(df_clean[c].median(), color="black", ls=":", label=f"Median {df_clean[c].median():.1f}")
    ax.set_title(f"{c} (skew = {df_clean[c].skew():.2f})"); ax.legend()
fig.tight_layout(); save(fig, "fig5_1_histograms")

# %% 5.2 Skewness, outliers, normality
rows = []
for c in num_cols:
    x = df_clean[c]; q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr = ((x < lo) | (x > hi)).sum()
    n_z = (np.abs(stats.zscore(x)) > 3).sum()
    rows.append([c, lo, hi, n_iqr, n_z, stats.shapiro(x).pvalue])
out = pd.DataFrame(rows, columns=["Variable", "Lower fence", "Upper fence", "Outliers (IQR)",
                                  "Outliers (|z|>3)", "Shapiro p"]).set_index("Variable").round(3)
print(out); save_table(out, "table5_2_outliers")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, (c, col) in enumerate(zip(num_cols, sns.color_palette("deep", 4))):
    sns.boxplot(y=df_clean[c], color=col, ax=axes[0, i]); axes[0, i].set_title(f"Box plot: {c}")
    stats.probplot(df_clean[c], dist="norm", plot=axes[1, i]); axes[1, i].set_title(f"Q-Q plot: {c}")
fig.tight_layout(); save(fig, "fig5_2_box_qq")

# %% 5.3 Categorical variables
freq = pd.concat({v: df_clean[v].value_counts().to_frame("Count").assign(
    Percent=lambda t: (t.Count / len(df_clean) * 100).round(1)) for v in ["origin", "cylinders", "period"]})
print(freq); save_table(freq, "table5_3_frequency")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for i, v in enumerate(["origin", "cylinders", "period"]):
    vc = df_clean[v].value_counts()
    sns.barplot(x=vc.index.astype(str), y=vc.values, palette="Set2", ax=axes[0, i])
    for j, val in enumerate(vc.values): axes[0, i].text(j, val, val, ha="center", va="bottom")
    axes[0, i].set(title=f"Bar chart: {v}", ylabel="count")
    axes[1, i].pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%", startangle=90,
                   colors=sns.color_palette("Set2"))
    axes[1, i].set_title(f"Pie chart: {v}")
fig.tight_layout(); save(fig, "fig5_3_categorical")
print("Majority-class baseline accuracy:", round(df_clean.origin.value_counts(normalize=True).max(), 3))

# %% [markdown]
# # UNIT 4 - Bivariate Analysis

# %% 6.1 Numerical vs numerical
import itertools
rows = []
for a, b in itertools.combinations(num_cols, 2):
    r, p = stats.pearsonr(df_clean[a], df_clean[b]); rho, _ = stats.spearmanr(df_clean[a], df_clean[b])
    rows.append([a, b, round(r, 3), round(rho, 3), p])
corr_tbl = pd.DataFrame(rows, columns=["Variable 1", "Variable 2", "Pearson r", "Spearman rho", "p (Pearson)"])
print(corr_tbl.to_string(index=False)); save_table(corr_tbl.set_index("Variable 1"), "table6_1_correlations")

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
sns.regplot(data=df_clean, x="weight", y="mpg", scatter_kws=dict(s=10), line_kws=dict(color="red"), ax=ax[0])
ax[0].set_title(f"(a) Weight vs mpg (r = {df_clean.weight.corr(df_clean.mpg):.2f})")
sns.regplot(data=df_clean, x="weight", y="acceleration", scatter_kws=dict(s=10, color="grey"),
            line_kws=dict(color="black"), ax=ax[1])
ax[1].set_title(f"(b) Weight vs acceleration, all data (r = {df_clean.weight.corr(df_clean.acceleration):.2f})")
sub = df_clean[df_clean.cylinders.isin([4, 6, 8])]
for cyl, col in zip([4, 6, 8], ["#4E79A7", "#59A14F", "#E15759"]):
    d = sub[sub.cylinders == cyl]
    sns.regplot(data=d, x="weight", y="acceleration", scatter_kws=dict(s=10), color=col,
                label=f"{cyl} cyl", ax=ax[2])
ax[2].set_title("(c) Same variables per cylinder group"); ax[2].legend(title="Cylinders")
fig.tight_layout(); save(fig, "fig6_1_scatter_regression")

print("\nSimpson-type reversal (weight vs acceleration):")
for cyl in [4, 6, 8]:
    d = sub[sub.cylinders == cyl]
    print(f"  {cyl} cyl: r = {d.weight.corr(d.acceleration):+.2f}")

# %% 6.2 Categorical vs categorical
def cramers_v(ct):
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return chi2, dof, p, np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))

ct1 = pd.crosstab(df_clean.origin, df_clean.cylinders)
ct2 = pd.crosstab(df_clean.origin, df_clean.period)
for lab, ct in [("origin x cylinders", ct1), ("origin x period", ct2)]:
    chi2, dof, p, v = cramers_v(ct)
    print(f"{lab}: chi2 = {chi2:.2f}, df = {dof}, p = {p:.2e}, Cramer's V = {v:.2f}")
print((pd.crosstab(df_clean.origin, df_clean.period, normalize="index") * 100).round(0))

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
ct1.loc[ORIGINS].plot(kind="bar", stacked=True, ax=ax[0], colormap="Set2", rot=0)
ax[0].set(title="(a) Stacked bar: origin by cylinders", ylabel="count")
ct1.loc[ORIGINS].plot(kind="bar", ax=ax[1], colormap="Set2", rot=0); ax[1].set_title("(b) Grouped bar: origin by cylinders")
ct2.loc[ORIGINS].plot(kind="bar", ax=ax[2], color=["#E15759", "#4E79A7"], rot=0); ax[2].set_title("(c) Grouped bar: origin by period")
fig.tight_layout(); save(fig, "fig6_2_categorical_bars")

# %% 6.3 Numerical vs categorical
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, c in enumerate(num_cols):
    sns.boxplot(data=df_clean, x="origin", y=c, order=ORIGINS, palette=PAL, ax=axes[0, i])
    axes[0, i].set_title(f"Box: {c}")
    sns.violinplot(data=df_clean, x="origin", y=c, hue="period", split=True, order=ORIGINS,
                   palette=["#E15759", "#4E79A7"], inner="quartile", ax=axes[1, i])
    axes[1, i].set_title(f"Violin (split by period): {c}")
    if i: axes[1, i].legend_.remove()
fig.tight_layout(); save(fig, "fig6_3_box_violin")

rows = []
for c in num_cols:
    groups = [df_clean[df_clean.origin == o][c] for o in ORIGINS]
    F, p = stats.f_oneway(*groups)
    ss_b = sum(len(g) * (g.mean() - df_clean[c].mean()) ** 2 for g in groups)
    ss_t = ((df_clean[c] - df_clean[c].mean()) ** 2).sum()
    rows.append([c, F, p, ss_b / ss_t])
anova = pd.DataFrame(rows, columns=["Variable", "F", "p-value", "eta^2"]).set_index("Variable").round(4)
print(anova); save_table(anova, "table6_2_anova")

from scipy.stats import tukey_hsd
th = tukey_hsd(*[df_clean[df_clean.origin == o].mpg for o in ORIGINS])
print("\nTukey HSD on mpg (order: USA, Europe, Japan):\n", th)
early = df_clean[df_clean.period == "Early"].mpg; late = df_clean[df_clean.period == "Late"].mpg
t, p = stats.ttest_ind(late, early, equal_var=False)
print(f"Late {late.mean():.2f} vs Early {early.mean():.2f} mpg | Welch t = {t:.1f}, p = {p:.1e}")

# %% [markdown]
# # UNIT 5 - Multivariate Analysis and Insights

# %% 7.1 Pair plot and correlation heatmaps
pp = sns.pairplot(df_clean, vars=num_cols, hue="origin", palette=PAL, diag_kind="kde",
                  plot_kws=dict(s=15, alpha=0.7), hue_order=ORIGINS)
pp.fig.suptitle("Pair plot (scatter matrix) of the four measurements", y=1.02, fontweight="bold")
save(pp.fig, "fig7_1_pairplot")

corr = df_clean[num_cols].corr().round(3); print(corr); save_table(corr, "table7_1_corr_matrix")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0, 0]); axes[0, 0].set_title("All cars")
for ax, o in zip(axes.ravel()[1:], ORIGINS):
    sns.heatmap(df_clean[df_clean.origin == o][num_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, ax=ax); ax.set_title(o)
fig.tight_layout(); save(fig, "fig7_2_heatmaps")

# %% 7.2 Grouped visualizations
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=df_clean, x="origin", y="mpg", hue="period", order=ORIGINS,
            palette=["#E15759", "#4E79A7"], ax=ax[0]); ax[0].set_title("(a) mpg by origin and period")
sns.barplot(data=df_clean[df_clean.cylinders.isin([4, 6, 8])], x="cylinders", y="mpg", hue="origin",
            hue_order=ORIGINS, palette=PAL, errorbar="sd", ax=ax[1]); ax[1].set_title("(b) Mean mpg by cylinders and origin (±SD)")
piv = df_clean.pivot_table(values="mpg", index="origin", columns="period", aggfunc="mean").loc[ORIGINS][["Early", "Late"]]
sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[2]); ax[2].set_title("(c) Mean mpg: origin x period")
fig.tight_layout(); save(fig, "fig7_3_grouped")
print(piv.round(1))

fg = sns.FacetGrid(sub, col="cylinders", row="period", hue="origin", palette=PAL, hue_order=ORIGINS,
                   height=3, aspect=1.2, row_order=["Early", "Late"])
fg.map_dataframe(sns.scatterplot, x="weight", y="mpg", s=15).add_legend(title="Origin")
fg.fig.suptitle("FacetGrid: weight vs mpg by cylinders (columns) and period (rows)", y=1.03, fontweight="bold")
save(fg.fig, "fig7_4_facetgrid")

# %% 7.3 Dimensionality reduction: PCA and t-SNE
Xs = StandardScaler().fit_transform(df_clean[num_cols])
pca = PCA(n_components=4).fit(Xs); Z = pca.transform(Xs)
ev = pca.explained_variance_ratio_ * 100
pca_tbl = pd.DataFrame(pca.components_, columns=num_cols, index=[f"PC{i+1}" for i in range(4)]).round(3)
pca_tbl.insert(0, "Cumulative %", ev.cumsum().round(2)); pca_tbl.insert(0, "Variance %", ev.round(2))
print(pca_tbl); save_table(pca_tbl, "table7_2_pca")
print("Mean PC1 per origin:\n", pd.Series(Z[:, 0]).groupby(df_clean.origin.values).mean().round(2))

tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(Xs)

fig, ax = plt.subplots(2, 2, figsize=(14, 11))
ax[0, 0].bar(range(1, 5), ev, color="#4E79A7"); ax[0, 0].plot(range(1, 5), ev.cumsum(), "ro-")
ax[0, 0].set(title="(a) Scree plot", xlabel="Principal component", ylabel="Explained variance (%)")
sns.heatmap(pca_tbl[num_cols].T, annot=True, cmap="coolwarm", center=0, ax=ax[0, 1]); ax[0, 1].set_title("(b) PCA loadings")
for o in ORIGINS:
    m = (df_clean.origin == o).values
    ax[1, 0].scatter(Z[m, 0], Z[m, 1], c=PAL[o], s=14, label=o)
    ax[1, 1].scatter(tsne[m, 0], tsne[m, 1], c=PAL[o], s=14, label=o)
ax[1, 0].set(title="(c) PCA projection", xlabel=f"PC1 ({ev[0]:.1f}%)", ylabel=f"PC2 ({ev[1]:.1f}%)"); ax[1, 0].legend(title="Origin")
ax[1, 1].set(title="(d) t-SNE projection (perplexity = 30)", xlabel="t-SNE 1", ylabel="t-SNE 2")
fig.tight_layout(); save(fig, "fig7_5_pca_tsne")

# %% 7.4 Scikit-learn model development
X = df_clean[num_cols + ["cylinders"]].copy()
X["period"] = df_clean["period"].map({"Early": 0, "Late": 1})
y = df_clean["origin"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print("Train/test sizes:", len(Xtr), len(Xte))

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "k-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    sc = cross_val_score(m, Xtr, ytr, cv=cv)
    m.fit(Xtr, ytr)
    rows.append([name, sc.mean() * 100, sc.std() * 100, accuracy_score(yte, m.predict(Xte)) * 100])
res = pd.DataFrame(rows, columns=["Model", "CV accuracy (mean) %", "CV std %", "Test accuracy %"]).set_index("Model").round(2)
print(res); save_table(res, "table7_3_classification")

rf = models["Random Forest"]
cm = confusion_matrix(yte, rf.predict(Xte), labels=ORIGINS)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ORIGINS, yticklabels=ORIGINS, ax=ax[0])
ax[0].set(title="(a) Confusion matrix - Random Forest", xlabel="Predicted label", ylabel="True label")
imp.plot(kind="barh", color="#59A14F", ax=ax[1]); ax[1].set(title="(b) Random Forest feature importance", xlabel="importance")
ax[2].bar(range(4), res["CV accuracy (mean) %"], yerr=res["CV std %"], color="#4E79A7", capsize=4)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(res.index, rotation=20); ax[2].set_ylim(50, 100)
ax[2].set(title="(c) Model comparison", ylabel="5-fold CV accuracy (%)")
fig.tight_layout(); save(fig, "fig7_6_models")
print("Per-class recall (RF):", dict(zip(ORIGINS, (cm.diagonal() / cm.sum(axis=1)).round(3))))

# %% [markdown]
# # Summary of key insights

# %% Summary
print("""
KEY INSIGHTS
1. American cars are heavier, more powerful and less fuel-efficient than European/Japanese cars.
2. European and Japanese cars are similar in size and power (horsepower Tukey p ~ 0.99).
3. Weight is the strongest driver of fuel economy (r = -0.83).
4. Weight vs acceleration shows a Simpson-type reversal (overall -0.42; within cylinder groups ~0 / +0.6 / ~0).
5. Fuel economy improved in every origin between the early and late 1970s.
6. Origin and engine size are strongly associated (Cramer's V ~ 0.48).
7. Distributions are non-normal; origin is only moderately predictable (~70-74% CV accuracy).
""")
print("Done. Figures -> ./figures | Tables -> ./outputs/tables | Interactive -> ./outputs/interactive")
