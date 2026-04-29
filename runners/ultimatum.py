from typing import List, Tuple
from core.player import Player
from runners.runner import Runner
from ratbench.constants import AGENT_ONE, AGENT_TWO
from ratbench.game_objects.goal import UltimatumGoal
from ratbench.game_objects.resource import Resources
from games.ultimatum.ultimatum_multi_turn.game import MultiTurnUltimatumGame
from games.ultimatum.ultimatum_multi_turn.interface import (
    UltimatumGameInterface
)


class UltimatumRunner(Runner):
    """
        Runs Ultimatum games between pairs of players
    """

    def __init__(
        self,
        player_one_initial_resources: dict,
        player_two_initial_resources: dict,
        games_per_pair: int,
    ) -> None:
        self.player_one_initial_resources = player_one_initial_resources
        self.player_two_initial_resources = player_two_initial_resources
        self.games_per_pair = games_per_pair

    def run(self,
            player_pairs: List[Tuple[Player, Player]],
            logs_dir: str) -> List[list]:
        """
        Execute a complete simulation of the Trading game for each pair.
        """
        total_games = len(player_pairs) * self.games_per_pair
        games_played = 0
        games_states = []
        for player_one, player_two in player_pairs:
            games_played_for_pair = 0
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
                    player_one_goals, player_two_goals = self._get_goals()
                    player_one_res, player_two_res = self._get_resources()
                    game = MultiTurnUltimatumGame(
                    iterations=6,
                    players=[agent_one, agent_two],
                    game_interface=UltimatumGameInterface(),
                    resources_support_set=Resources({"ZUP": 0}),
                    player_goals=[
                        player_one_goals,
                        player_two_goals
                    ],
                    player_initial_resources=[
                        player_one_res,
                        player_two_res,
                    ],
                    player_social_behaviour=[
                        "",
                        "",
                    ],
                    player_roles=[
                        f"You are {AGENT_ONE}",
                        f"You are {AGENT_TWO}",
                    ],
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

                except Exception as e:
                    print(f"Exception occurred while running game"
                          f" {games_played + 1}/{total_games}: {e!r}")

        return games_states

    @staticmethod
    def _get_goals() -> List[UltimatumGoal]:
        """
        Returns first and second player goals.
        """
        player_one_goals = UltimatumGoal()
        player_two_goals = UltimatumGoal()

        return [player_one_goals, player_two_goals]

    def _get_resources(self) -> List[Resources]:
        """
        Returns first and second player initial resources
        """
        player_one_resources = Resources(self.player_one_initial_resources)
        player_two_resources = Resources(self.player_two_initial_resources)

        return [player_one_resources, player_two_resources]

