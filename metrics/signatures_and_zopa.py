import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from metrics import buy_sell_logs_to_df

"""
This script generates the graph that compares model performance the strategic 
signatures of Gemini 2.5 pro and Claude Sonnet 4.5 as the ZOPA changes.
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
    "legend.fontsize": 18,
})


def get_data(log_name: str):
    """
    This function computes buyer-seller metrics from a log file
    """
    df = buy_sell_logs_to_df(log_name=log_name)
    df["gap"] = df["buyer_valuation"] - df["seller_valuation"]
    df.sort_values(by="gap", inplace=True)
    df["seller_score"] = df["seller_payoff"] / df["gap"]
    df["buyer_score"] = df["buyer_payoff"] / df["gap"]
    seller_stats = df[["gap", "seller_score"]].groupby("gap").agg(
        ["mean", "var"])
    buyer_stats = df[["gap", "buyer_score"]].groupby("gap").agg(
        ["mean", "var"])
    gap = seller_stats.index
    seller_mean = seller_stats["seller_score"]["mean"]
    seller_std = np.sqrt(seller_stats["seller_score"]["var"])
    buyer_mean = buyer_stats["buyer_score"]["mean"]
    buyer_std = np.sqrt(buyer_stats["buyer_score"]["var"])
    return gap, seller_mean, seller_std, buyer_mean, buyer_std


def plot_on_axis(ax, gap, s_mean, s_std, b_mean, b_std, color,
                 show_xlabel=False):
    """
    Plot a single curve over an axis
    """
    ax.plot(gap, s_mean, color=color, linestyle="--", linewidth=2.5)
    ax.fill_between(gap, s_mean - s_std, s_mean + s_std, color=color,
                    alpha=0.2)

    ax.plot(gap, b_mean, color=color, linestyle="-", linewidth=2.5)
    ax.fill_between(gap, b_mean - b_std, b_mean + b_std, color=color,
                    alpha=0.2)

    ax.set_ylabel("Payoff")

    ax.set_xlim(10, 90)
    ax.set_ylim(0, 1.25)

    if show_xlabel:
        ax.set_xlabel("Zone of Possible Agreement")

    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.75,
            alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)



gap_s, s_mean_sonnet, s_std_sonnet, b_mean_sonnet, b_std_sonnet = get_data(
    log_name="zopa-claude-sonnet"
)
gap_g, s_mean_gemini, s_std_gemini, b_mean_gemini, b_std_gemini = get_data(
    log_name="zopa-gemini-25-pro"
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11), sharey=True,
                               sharex=True)

col_sonnet = "#77b5b6"
col_gemini = "#9671bd"

plot_on_axis(
    ax1, gap_s, s_mean_sonnet, s_std_sonnet, b_mean_sonnet, b_std_sonnet,
    color=col_sonnet,
    show_xlabel=False
)

plot_on_axis(
    ax2, gap_g, s_mean_gemini, s_std_gemini, b_mean_gemini, b_std_gemini,
    color=col_gemini,
    show_xlabel=True
)

legend_elements = [
    Line2D(xdata=[0],
           ydata=[0],
           marker='o',
           color='w',
           label='Claude 4.5 Sonnet',
           markerfacecolor=col_sonnet,
           markersize=12
           ),
    Line2D(xdata=[0],
           ydata=[0],
           marker='o',
           color='w',
           label='Gemini 2.5 Pro',
           markerfacecolor=col_gemini,
           markersize=12),

    Line2D(xdata=[0],
           ydata=[0],
           color='gray',
           lw=3,
           linestyle='-',
           label='Buyer'),
    Line2D(xdata=[0],
           ydata=[0],
           color='gray',
           lw=3,
           linestyle='--',
           label='Seller'),
]

plt.suptitle(t="Buyer and Seller Payoffs vs. Zone of Possible Agreement",
             y=0.96,
             fontsize=22)
plt.tight_layout(rect=[0, 0.12, 1, 0.96])

fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.5, 0.02),
           ncol=2, frameon=False)

project_root = Path(__file__).resolve().parent
out_dir = project_root / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "zopa-subplots.pdf", dpi=300,
            bbox_inches='tight')
