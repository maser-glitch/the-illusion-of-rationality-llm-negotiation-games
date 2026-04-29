from core.player import Player
from optimization import Agent
from runners import BuySellRunner

"""
This script executes the games pairs for the optimized agent.

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

agent = Agent(env=None)
agent.load("optimized_002.pkl")


new_player = Player(model="gpt-4.1-mini-2025-04-12-cde-aia",
                    optimized=True)
player_pairs = ([(new_player, player) for player in players] +
                [(player, new_player) for player in players] +
                [(new_player, new_player)])


buyer_valuation = 60
seller_valuation = 40
for seller, buyer in player_pairs:
    for _ in range(20):
        if seller.optimized:
            strategy =  agent.strategy(valuation=seller_valuation,
                                       role="seller").strategy
            social_behaviors = [strategy, ""]
        else:
            strategy = agent.strategy(valuation=buyer_valuation,
                                      role="buyer").strategy
            social_behaviors = ["", strategy]
        runner = BuySellRunner(buyer_valuation=buyer_valuation,
                               seller_valuation=seller_valuation,
                               games_per_pair=1)
        games_states = runner.run(
            player_pairs=[(seller, buyer)],
            logs_dir=".logs/buysell-experiment-optimized/",
            social_behaviors=social_behaviors
        )


