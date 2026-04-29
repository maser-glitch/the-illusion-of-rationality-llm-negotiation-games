import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from metrics import buy_sell_logs_to_df

"""
This script generates the scatter plot that compares different models in the
buyer-seller scenario including the optimized gpt4.1 model. 
"""

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["DejaVu Serif"],
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

log_name = "buysell-experiment"
df = buy_sell_logs_to_df(log_name=log_name)

model_map = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gpt-4.1-2025-04-14-cde-aia": "gpt-4.1",
    "gpt-4.1-mini-2025-04-12-cde-aia": "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-12-cde-aia_optimized": "gpt-4.1-mini-optimized",
    "gpt-4o-2024-08-06-cde-aia": "gpt-4o",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "sonnet4.5",
}

model_colors = {
    "gemini-2.5-pro": "#A46B9B",
    "gemini-2.5-flash": "#D9739E",
    "gpt-4.1": "#F1976D",
    "gpt-4.1-mini": "#76DECD",
    "gpt-4o": "#C7DB74",
    "sonnet4.5": "#FFCF50",
    "gpt-4.1-mini-optimized": "#7f7f7f",
}

model_colors_edge = {
    "gemini-2.5-pro": "#8E4782",
    "gemini-2.5-flash": "#BB4878",
    "gpt-4.1": "#D67041",
    "gpt-4.1-mini": "#4BC0AD",
    "gpt-4o": "#A7BD49",
    "sonnet4.5": "#E5AF21",
    "gpt-4.1-mini-optimized": "#7f7f7f",
}

model_colors_markers = {
    "gemini-2.5-pro": "o",
    "gemini-2.5-flash": "s",
    "gpt-4.1": "^",
    "gpt-4.1-mini": "D",
    "gpt-4o": "v",
    "sonnet4.5": "X",
    "gpt-4.1-mini-optimized": "",
}


pretty = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 mini",
    "gpt-4.1-mini-optimized": "GPT-4.1 mini + GEPA",
    "gpt-4o": "GPT-4o",
    "sonnet4.5": "Claude Sonnet 4.5",
}


df["buyer_model_short"]  = df["buyer_model"].map(model_map).fillna(df["buyer_model"])
df["seller_model_short"] = df["seller_model"].map(model_map).fillna(df["seller_model"])


seller_stats = (
    df.groupby("seller_model_short")["seller_payoff"]
      .agg(['mean', 'var'])
      .rename(columns={'mean': 'seller_mean', 'var': 'seller_var'})
)
buyer_stats = (
    df.groupby("buyer_model_short")["buyer_payoff"]
      .agg(['mean', 'var'])
      .rename(columns={'mean': 'buyer_mean', 'var': 'buyer_var'})
)

combined = pd.concat(
    objs=[seller_stats, buyer_stats],
    axis=1).dropna(subset=['seller_mean', 'buyer_mean'])

plt.figure(figsize=(10, 10))
for model, row in combined.iterrows():
    color = model_colors.get(model, "#999999")
    edge_color = model_colors_edge.get(model, "#999999")
    label = pretty.get(model, model)
    marker = model_colors_markers.get(model, "o")
    plt.scatter(row["seller_mean"], row["buyer_mean"],
                label=label,
                s=120,
                color=color,
                marker=marker,
                edgecolors=edge_color,
                linewidths=1.5,
                zorder=3)

# Equal axes
min_val = min(combined["seller_mean"].min(), combined["buyer_mean"].min())
max_val = max(combined["seller_mean"].max(), combined["buyer_mean"].max())


zoom_out = 0.6
x_min = min(combined["seller_mean"])
x_max = max(combined["seller_mean"])
x_median = (x_min + x_max)/2
x_range = x_max - x_min

y_min = min(combined["buyer_mean"])
y_max = max(combined["buyer_mean"])
y_median = (y_min + y_max)/2
y_range = y_max - y_min

plotting_range = max([x_range, y_range]) + zoom_out

plt.xlim(x_median - plotting_range/2, x_median + plotting_range/2)
plt.ylim(2.5, 12.5)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel(r"Mean Seller Payoff ($P - v_s$)")
plt.ylabel(r"Mean Buyer Payoff ($v_b - P$)")
plt.title("Mean Payoffs in Buyer–Seller")

handles, labels = plt.gca().get_legend_handles_labels()

plt.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=False
)


plt.grid(visible=True, which='major',
         linestyle='-', linewidth=0.75, alpha=0.25)
plt.minorticks_on()
plt.grid(visible=True, which='minor', linestyle='-',
         linewidth=0.25, alpha=0.15)

file_dir = Path(__file__).resolve().parent
plots_dir = file_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / f"{log_name}-scatter.pdf", dpi=300,
            bbox_inches='tight')
