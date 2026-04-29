from core.player import Player
from typing import List, Tuple
from runners.runner import Runner
from games.buy_sell_game.game import BuySellGame
from ratbench.game_objects.resource import Resources
from ratbench.game_objects.valuation import Valuation
from ratbench.game_objects.goal import BuyerGoal, SellerGoal
from ratbench.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN


class BuySellRunner(Runner):
    """
    Runs Buy/Sell games between pairs of players.
    """
    GAME_NAME = "Buy/Sell"
    def __init__(
        self,
        buyer_valuation: int,
        seller_valuation: int,
        games_per_pair: int,
        buyer_resources: int = 100,
        seller_resources: int = 1
    ) -> None:

        self.buyer_valuation = buyer_valuation
        self.seller_valuation = seller_valuation
        self.games_per_pair = games_per_pair
        self.buyer_resources = buyer_resources
        self.seller_resources = seller_resources

    def run(self,
            player_pairs: List[Tuple[Player, Player]],
            logs_dir: str,
            social_behaviors: List[str] = None,
            patience_value: int = 3) -> List[list]:
        """
        Execute a complete simulation of the Buy/Sell game for each pair.
        """
        total_games = len(player_pairs) * self.games_per_pair
        games_played = 0
        games_states = []
        for player_one, player_two in player_pairs:
            games_played_for_pair = 0
            patience = patience_value
            while games_played_for_pair < self.games_per_pair:
                self.log(
                    player_one=player_one,
                    player_two=player_two,
                    games_played=games_played,
                    total_games=total_games,
                )
                try:
                    agent_one, agent_two = self._get_agents(
                        player_one=player_one,
                        player_two=player_two
                    )
                    seller_goal, buyer_goal = self._get_goals()
                    seller_resources, buyer_resources = self._get_resources()
                    social = social_behaviors if social_behaviors else ["", ""]
                    game = BuySellGame(
                        players=[agent_one, agent_two],
                        iterations=10,
                        resources_support_set=Resources({"X": 0}),
                        player_goals=[seller_goal, buyer_goal],
                        player_initial_resources=[
                            seller_resources,
                            buyer_resources
                        ],
                        player_roles=[
                            f"You are {AGENT_ONE}.",
                            f"You are {AGENT_TWO}.",
                        ],
                        player_social_behaviour=social,
                        log_dir=logs_dir,
                    )
                    game.run()

                    games_played_for_pair += 1
                    games_played += 1
                    players = [player.get_state() for player in game.players]
                    game_state = {
                        "game_state": game.game_state,
                        "players": players
                    }
                    games_states.append(game_state)
                except ValueError:
                    patience -= 1
                    if not patience:
                        game_state = {
                            "fail": True
                        }
                        games_states.append(game_state)
                        patience = patience_value
                        games_played_for_pair += 1
                        games_played += 1
                except Exception as e:
                    print(f"Exception occurred while running game"
                          f" {games_played + 1}/{total_games}: {e!r}")
        return games_states

    def _get_goals(self) -> Tuple[SellerGoal, BuyerGoal]:
        """
        Return seller and buyer goals.
        """
        seller_goal = SellerGoal(
            cost_of_production=Valuation({"X": self.seller_valuation})
        )
        buyer_goal = BuyerGoal(
            willingness_to_pay=Valuation({"X": self.buyer_valuation})
        )
        return seller_goal, buyer_goal

    def _get_resources(self) -> Tuple[Resources, Resources]:
        """
        Returns seller and buyer initial resources
        """
        seller_resources = Resources({"X": self.seller_resources})
        buyer_resources = Resources({MONEY_TOKEN: self.buyer_resources})
        return seller_resources, buyer_resources
