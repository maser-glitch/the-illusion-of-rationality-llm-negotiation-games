from core.player import Player
from runners import BuySellRunner

"""
This scripts runs the gap analysis in the buyer and seller scenario
"""

players = [
    Player(model="gemini-2.5-pro")
]

player_pairs = [(a, b) for a in players for b in players]

mid_point = 100
for num_exp, gap in enumerate([5, 10, 15, 20, 25, 30, 35, 40, 45]):
    print(f" === Experiment No. {num_exp + 1} ===""")
    buyer_valuation = mid_point + gap
    seller_valuation = mid_point - gap
    print(f"seller_valuation: {seller_valuation} - "
          f"buyer_valuation: {buyer_valuation} ")
    runner = BuySellRunner(buyer_valuation=buyer_valuation,
                           seller_valuation=seller_valuation,
                           buyer_resources=200,
                           games_per_pair=20)
    runner.run(player_pairs=player_pairs,
               logs_dir=".logs/buysell-rare-large_values-gemini-2.5-pro/")
