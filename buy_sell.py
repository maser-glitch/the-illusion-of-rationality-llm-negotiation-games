from core.player import Player
from runners import BuySellRunner

"""
This script simulates games for the Buyer and Seller Scenario
"""

players = [
    Player(model="gemini-2.5-flash"),
    Player(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
           region="us-east-1"),
    Player(model="gpt-4.1-mini-2025-04-12-cde-aia"),
    Player(model="gpt-4o-2024-08-06-cde-aia"),
    Player(model="gpt-4.1-2025-04-14-cde-aia"),
    Player(model="gemini-2.5-pro")
]

player_pairs = [(a, b) for a in players for b in players]

runner = BuySellRunner(buyer_valuation=60,
                       seller_valuation=40,
                       games_per_pair=1)

games_states = runner.run(
    player_pairs=player_pairs,
    logs_dir=".logs/buysell-experiment-2/"
)
