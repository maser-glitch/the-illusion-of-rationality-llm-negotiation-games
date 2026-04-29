import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from metrics import buy_sell_logs_to_df

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


def load_data(log_name: str) -> Tuple[np.ndarray, np.ndarray]:

    df = buy_sell_logs_to_df(log_name=log_name)
    print(f"Seller valuation stats for log {log_name}")
    print(df["seller_valuation"].describe())
    df["gap"] = df["buyer_valuation"] - df["seller_valuation"]
    df["ip_norm"] = (df["first_proposal"] - df["seller_valuation"]) / df["gap"]
    df["fp_norm"] = (df["price"] - df["seller_valuation"]) / df["gap"]
    ip_norm = df["ip_norm"].to_numpy()
    fp_norm = df["fp_norm"].to_numpy()
    return ip_norm,  fp_norm


fig, ax = plt.subplots(figsize=(10, 10))

gemini_ip, gemini_fp = load_data(log_name="buysell-anchoring-gemini-25-pro")
rho_gemini, _ = spearmanr(gemini_ip, gemini_fp)
ax.scatter(gemini_ip, gemini_fp,
           label="Gemini 2.5 Pro",
           s=35,
           color="#9671bd",
           edgecolors="#6a408d")

sonnet_ip, sonnet_fp = load_data(log_name="buysell-anchoring-sonnet-45")
rho_sonnet, _ = spearmanr(sonnet_ip, sonnet_fp)
ax.scatter(sonnet_ip, sonnet_fp,
           label="Claude Sonnet 4.5",
           s=35,
           color="#77b5b6",
           edgecolors="#378d94")

zoom_out = 0.1
x_merged = np.concatenate([gemini_ip, sonnet_ip])
x_min = min(x_merged)
x_max = max(x_merged)
x_median = (x_min + x_max)/2
x_range = x_max - x_min

y_merged = np.concatenate([gemini_fp, sonnet_fp])
y_min = min(y_merged)
y_max = max(y_merged)
y_median = (y_min + y_max)/2
y_range = y_max - y_min

plotting_range = max([x_range, y_range]) + zoom_out

plt.xlim(0.1, 2.1)
plt.ylim(0.04, 1.04)

lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([lo, hi], [lo, hi],
        color="#7e7e7e", linewidth=1.5, linestyle="--", zorder=2)


ax.set_aspect('equal', adjustable='box')

ax.set_title("Normalized Initial Proposal vs. Final Price")
ax.set_xlabel(r"Initial Proposal $\tilde{p}_1$")
ax.set_ylabel(r"Final Price $\tilde{p}_{\mathrm{final}}$")


plt.grid(visible=True, which='major',
         linestyle='-', linewidth=0.75, alpha=0.25)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle='-',
         linewidth=0.25, alpha=0.15)

handles, labels = plt.gca().get_legend_handles_labels()

plt.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=False
)

project_root = Path(__file__).resolve().parent
out_dir = project_root / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Spearman Correlation Sonnet: {rho_sonnet}")
print(f"Spearman Correlation Gemini: {rho_gemini}")
fig.savefig(out_dir / "numerical_anchoring.pdf", dpi=300, bbox_inches='tight')

