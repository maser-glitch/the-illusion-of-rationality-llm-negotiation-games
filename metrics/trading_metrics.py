import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from metrics.metrics_utils import trading_logs_to_df

"""
This script generates the metrics for the multi-turn ultimatum game.
"""

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["DejaVu Serif"],
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
})


NAMES_MAP = {
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'gpt-4.1-2025-04-14-cde-aia': 'GPT-4.1',
    'gpt-4.1-mini-2025-04-12-cde-aia': 'GPT-4.1 mini',
    'gpt-4o-2024-08-06-cde-aia': 'GPT-4o',
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0': 'Claude Sonnet 4.5'
}

colormap = sns.light_palette(color="#9671bd", as_cmap=True)

df = trading_logs_to_df("trading-experiment")

df["models"] = df["player_one_model"] + "@" + df["player_two_model"]
target_cols = ["player_one_win", "player_two_win", "draw",
               "player_one_delta", "player_two_delta"]
df_grouped = df.groupby(by="models")[target_cols].mean().reset_index()
df_grouped[['model_1', 'model_2']] = df_grouped['models'].str.split(
    '@',
    n=1,
    expand=True)

win_sum = df_grouped['player_one_win'] + df_grouped['player_two_win']
df_grouped['player_one_win_rate'] = df_grouped['player_one_win'].divide(
    win_sum
).fillna(0)
df_grouped['player_two_win_rate'] = df_grouped['player_two_win'].divide(
    win_sum
).fillna(0)


win_rate_data = df_grouped.pivot(index="model_2",
                                 columns="model_1",
                                 values="player_two_win_rate")
win_rate_data = win_rate_data.rename(index=NAMES_MAP,
                                     columns=NAMES_MAP)

payoff_data = df_grouped.pivot(index="model_2",
                               columns="model_1",
                               values="player_two_delta")
payoff_data = payoff_data.rename(index=NAMES_MAP,
                                 columns=NAMES_MAP)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

sns.heatmap(
    win_rate_data,
    annot=True,
    cmap=colormap,
    ax=axes[0],
    fmt=".2f",
    vmin=0.0,
    vmax=1.0,
    linewidths=1.0,
    linecolor='white',
    annot_kws={'size': 18},
    cbar=False
)
axes[0].set_ylabel('Player 2')
axes[0].set_xlabel('Player 1')
axes[0].set_title('Player 2 Win Rate')
axes[0].tick_params(axis='y', rotation=0)
axes[0].tick_params(axis='x')

sns.heatmap(
    payoff_data,
    annot=True,
    cmap=colormap,
    ax=axes[1],
    fmt=".2f",
    linewidths=1.0,
    linecolor='white',
    annot_kws={'size': 18},
    cbar=False
)
axes[1].set_ylabel('')
axes[1].set_xlabel('Player 1')
axes[1].set_title('Player 2 Average Payoff')
axes[1].tick_params(axis='y',
                    rotation=0)
axes[1].tick_params(axis='x')

plt.setp(axes[0].get_xticklabels(),
         rotation=45,
         ha='right',
         rotation_mode='anchor')
plt.setp(axes[1].get_xticklabels(),
         rotation=45,
         ha='right',
         rotation_mode='anchor')

plt.tight_layout()

project_root = Path(__file__).resolve().parent
out_dir = project_root / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "trading_metrics.pdf", dpi=300, bbox_inches='tight')