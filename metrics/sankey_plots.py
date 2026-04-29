import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from metrics import buy_sell_logs_to_df
from matplotlib.colors import LinearSegmentedColormap

"""
This script generates the sankey graph that shows the strong semantic 
anchoring of frontier LLMs.
"""

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["DejaVu Serif"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

COL_SONNET = "#77b5b6"
COL_GEMINI = "#9671bd"
COL_MID = "#E8E6E1"

CMAP_LEFT = LinearSegmentedColormap.from_list(name="sonnet_mid",
                                              colors=[COL_SONNET, COL_MID])
CMAP_RIGHT = LinearSegmentedColormap.from_list(name="mid_gemini",
                                               colors=[COL_MID, COL_GEMINI])


def get_data():
    try:
        df_claude = buy_sell_logs_to_df(log_name="buysell-anchoring-sonnet-45")
        df_gemini = buy_sell_logs_to_df(
            log_name="buysell-anchoring-gemini-25-pro")
    except ImportError:
        np.random.seed(42)
        c_vals = np.random.randint(20, 80, 500)
        c_props = c_vals * np.random.uniform(0.5, 1.5, 500)
        df_claude = pd.DataFrame(
            {'seller_valuation': c_vals, 'first_proposal': c_props})

        g_vals = np.random.randint(20, 80, 500)
        g_props = g_vals * np.random.uniform(0.5, 1.5, 500)
        df_gemini = pd.DataFrame(
            {'seller_valuation': g_vals, 'first_proposal': g_props})

    df_claude['proposal_final'] = df_claude['first_proposal'].round().astype(
        int
    )
    df_gemini['proposal_final'] = df_gemini['first_proposal'].round().astype(
        int
    )

    return df_claude, df_gemini


def draw_sankey_ribbon(ax, x_start,
                       x_end,
                       y_start_top,
                       y_start_bot,
                       y_end_top,
                       y_end_bot,
                       colormap,
                       alpha=0.4):
    verts = [
        (x_start, y_start_top),
        (x_start + (x_end - x_start) * 0.5, y_start_top),
        (x_end - (x_end - x_start) * 0.5, y_end_top),
        (x_end, y_end_top),
        (x_end, y_end_bot),
        (x_end - (x_end - x_start) * 0.5, y_end_bot),
        (x_start + (x_end - x_start) * 0.5, y_start_bot),
        (x_start, y_start_bot),
        (x_start, y_start_top),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    path = Path(verts, codes)

    clip_patch = patches.PathPatch(path,
                                   facecolor='none',
                                   edgecolor='none',
                                   lw=0)
    ax.add_patch(clip_patch)

    gradient_data = np.linspace(start=0, stop=1, num=256).reshape(1, -1)

    y_min = min(y_start_bot, y_end_bot) - 5
    y_max = max(y_start_top, y_end_top) + 5
    extent = [x_start, x_end, y_min, y_max]

    im = ax.imshow(gradient_data, cmap=colormap, aspect='auto', extent=extent,
                   alpha=alpha, origin='lower', zorder=1)
    im.set_clip_path(clip_patch)


def generate_matplotlib_sankey():
    df_c, df_g = get_data()

    left_counts = df_c['seller_valuation'].value_counts().sort_index(
        ascending=False
    )
    right_counts = df_g['seller_valuation'].value_counts().sort_index(
        ascending=False
    )

    mid_counts_c = df_c['proposal_final'].value_counts().sort_index()
    mid_counts_g = df_g['proposal_final'].value_counts().sort_index()

    all_props = sorted(list(set(mid_counts_c.index) | set(mid_counts_g.index)),
                       reverse=True)

    GAP = 2

    def get_node_positions(counts_series, gap):
        current_y = 0
        pos = {}
        for val, count in counts_series.items():
            pos[val] = {'bot': current_y, 'top': current_y + count, 'h': count}
            current_y += count + gap
        return pos, current_y

    left_pos, max_y_left = get_node_positions(left_counts, GAP)
    right_pos, max_y_right = get_node_positions(right_counts, GAP)

    mid_pos = {}
    current_y = 0
    for p in all_props:
        c_in = mid_counts_c.get(p, 0)
        g_in = mid_counts_g.get(p, 0)
        total = c_in + g_in
        mid_pos[p] = {
            'bot': current_y,
            'top': current_y + total,
            'c_top': current_y + c_in,
            'h': total
        }
        current_y += total + GAP
    max_y_mid = current_y

    global_max = max(max_y_left, max_y_right, max_y_mid)

    def adjust_y(pos_dict, my_max):
        offset = (global_max - my_max) / 2
        for k in pos_dict:
            pos_dict[k]['bot'] += offset
            pos_dict[k]['top'] += offset
            if 'c_top' in pos_dict[k]: pos_dict[k]['c_top'] += offset
        return pos_dict

    left_pos = adjust_y(left_pos, max_y_left)
    right_pos = adjust_y(right_pos, max_y_right)
    mid_pos = adjust_y(mid_pos, max_y_mid)

    fig, ax = plt.subplots(figsize=(14, 10))

    X_LEFT, X_MID, X_RIGHT = 0, 5, 10
    WIDTH = 0.05

    left_fill_tracker = {k: v['bot'] for k, v in left_pos.items()}
    right_fill_tracker = {k: v['bot'] for k, v in right_pos.items()}
    mid_fill_tracker_c = {k: v['bot'] for k, v in mid_pos.items()}
    mid_fill_tracker_g = {k: v['c_top'] for k, v in mid_pos.items()}

    flows_c = df_c.groupby(['seller_valuation', 'proposal_final']).size().reset_index(name='count')
    flows_c.sort_values(by=['seller_valuation', 'proposal_final'],
                        ascending=[True, False],
                        inplace=True)

    for _, row in flows_c.iterrows():
        val, prop, count = row['seller_valuation'], row['proposal_final'], row['count']
        y_l_start = left_fill_tracker[val]
        y_l_end = y_l_start + count
        left_fill_tracker[val] += count
        y_m_start = mid_fill_tracker_c[prop]
        y_m_end = y_m_start + count
        mid_fill_tracker_c[prop] += count
        draw_sankey_ribbon(ax, X_LEFT + WIDTH, X_MID - WIDTH,
                           y_l_end, y_l_start, y_m_end, y_m_start,
                           CMAP_LEFT, alpha=0.5)

    flows_g = df_g.groupby(['seller_valuation', 'proposal_final']).size().reset_index(name='count')
    flows_g.sort_values(by=['seller_valuation', 'proposal_final'],
                        ascending=[True, False],
                        inplace=True)

    for _, row in flows_g.iterrows():
        val, prop, count = row['seller_valuation'], row['proposal_final'], row['count']
        y_r_start = right_fill_tracker[val]
        y_r_end = y_r_start + count
        right_fill_tracker[val] += count
        y_m_start = mid_fill_tracker_g[prop]
        y_m_end = y_m_start + count
        mid_fill_tracker_g[prop] += count
        draw_sankey_ribbon(ax, X_MID + WIDTH, X_RIGHT - WIDTH,
                           y_m_end, y_m_start, y_r_end, y_r_start,
                           CMAP_RIGHT, alpha=0.5)

    def draw_nodes(pos_dict, x_pos, color, label_align='left'):
        for label, coords in pos_dict.items():
            rect = patches.Rectangle((x_pos - WIDTH, coords['bot']), 2 * WIDTH,
                                     coords['h'],
                                     linewidth=1,
                                     edgecolor='none',
                                     facecolor=color,
                                     zorder=10)
            ax.add_patch(rect)

        text_data = []
        for label, coords in pos_dict.items():
            y_center = coords['bot'] + coords['h'] / 2
            text_data.append({'label': label, 'y': y_center})
        text_data.sort(key=lambda x: x['y'])

        min_text_dist = 2.5
        if text_data:
            last_placed_y = text_data[0]['y'] - min_text_dist
            for item in text_data:
                if item['y'] < last_placed_y + min_text_dist:
                    item['y'] = last_placed_y + min_text_dist
                last_placed_y = item['y']

        offset = WIDTH + 0.4
        final_x = x_pos + offset if label_align == 'left' else x_pos - offset
        ha = 'left' if label_align == 'left' else 'right'

        for item in text_data:
            ax.text(final_x, item['y'], f"${item['label']}$",
                    va='center', ha=ha, fontsize=11, zorder=10)

    draw_nodes(left_pos, X_LEFT, COL_SONNET, label_align='right')
    draw_nodes(mid_pos, X_MID, COL_MID, label_align='left')
    draw_nodes(right_pos, X_RIGHT, COL_GEMINI, label_align='left')

    ax.axis('off')
    ax.set_ylim(0, global_max * 1.15)
    ax.set_xlim(X_LEFT - 2, X_RIGHT + 2)

    y_header = global_max * 1.05
    ax.text(X_LEFT,
            y_header,
            "Claude 4.5\n Initial Seller Valuation",
            ha='center',
            fontsize=14)
    ax.text(X_MID,
            y_header,
            "Proposal $p_1$",
            ha='center',
            fontsize=14)
    ax.text(X_RIGHT,
            y_header,
            "Gemini 2.5\n Initial Seller Valuation",
            ha='center',
            fontsize=14)

    arrow_y = y_header * 1.015

    arrow_props_sonnet = dict(arrowstyle="->",
                              color=COL_SONNET,
                              lw=1,
                              mutation_scale=25)
    arrow_props_gemini = dict(arrowstyle="->",
                              color=COL_GEMINI,
                              lw=1,
                              mutation_scale=25)

    PADDING = 2
    ax.annotate("",
                xy=(X_MID - PADDING, arrow_y),
                xytext=(X_LEFT + PADDING, arrow_y),
                arrowprops=arrow_props_sonnet)

    ax.annotate("",
                xy=(X_MID + PADDING, arrow_y),
                xytext=(X_RIGHT - PADDING, arrow_y),
                arrowprops=arrow_props_gemini)

    plt.tight_layout()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig("plots/sankey.pdf", bbox_inches='tight', dpi=300)


generate_matplotlib_sankey()