from core.player import Player
from runners import UltimatumRunner


"""
This script runs the experiments for ultimatum game 
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

player_one_initial_resources = {"Dollars": 100}
player_two_initial_resources = {"Dollars": 0}

runner = UltimatumRunner(
    player_one_initial_resources=player_one_initial_resources,
    player_two_initial_resources=player_two_initial_resources,
    games_per_pair=20)

runner.run(player_pairs=player_pairs,
           logs_dir=".logs/ultimatum-experiment/")

