from core.player import Player
from ratbench.agents import Agent
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from utils.agent_factory import agent_factory
from ratbench.constants import AGENT_ONE, AGENT_TWO

class Runner(ABC):
    GAME_NAME: Optional[str] = None
    @abstractmethod
    def run(self,  *args, **kwargs):
        """Runs an experiment"""

    @staticmethod
    def _get_agents(
            player_one: Player,
            player_two: Player) -> Tuple[Agent, Agent]:
        agent_one = agent_factory(
            agent_name=AGENT_ONE,
            model=player_one.model,
            region=player_one.region,
        )
        agent_two = agent_factory(
            agent_name=AGENT_TWO,
            model=player_two.model,
            region=player_two.region,
        )

        return agent_one, agent_two

    @classmethod
    def log(
            cls,
            player_one: Player,
            player_two: Player,
            games_played: int,
            total_games: int,
    ) -> None:
        """
        Prints the simulation progress for the current game.
        """
        separator = "=" * 40
        print(f"\n{separator}")
        print(f"{cls.GAME_NAME} Game Progress")
        print(f"{separator}")
        print(f"Player One : {player_one}")
        print(f"Player Two : {player_two}")
        print(f"Progress   : Game {games_played + 1} of {total_games}")
        print(separator)
