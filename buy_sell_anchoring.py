import random
from core.player import Player
from runners import BuySellRunner

"""
This script simulates games for the Buyer and Seller Scenario
to study the anchoring bias
"""


players = [
    Player(model="anthropic.claude-opus-4-5-20251101-v1:0",
           region="us-east-1"),
]

player_pairs = [(a, b) for a in players for b in players]

for num_exp in range(100):
    print(f" === Experiment No. {num_exp + 1} ===""")
    runner = BuySellRunner(buyer_valuation=random.randint(a=60, b=80),
                           seller_valuation=random.randint(a=20, b=40),
                           games_per_pair=1)
    runner.run(player_pairs=player_pairs,
               logs_dir=".logs/buysell-anchoring-claude-4.5-opus/")
