from envs.pool import EnvPool
from core.player import Player
from typing import List, Tuple
from runners import BuySellRunner
from envs.parsers import game_state_parser

COMMUNICATION_PROTOCOL_ERROR = ("Player did not follow the communication "
                                 "protocol. Maybe it did not follow the "
                                 "communication pattern. The simulation "
                                 "could not be completed")

class BuySellEnv:
    """Buy sell environment used for the prompt optimization part"""
    def __init__(self, logs_dir: str) -> None:
        self.logs_dir = logs_dir
        self.runner = None
        self.learner = None
        self.seller = None
        self.buyer = None
        self.other_social_behavior = None
        self.pool = EnvPool(self)

    def init(self,
             buyer_valuation: int,
             seller_valuation: int,
             seller: Player,
             buyer: Player,
             other_social_behavior: str,
             learner: str) -> None:
        self.runner = BuySellRunner(
            buyer_valuation=buyer_valuation,
            seller_valuation=seller_valuation,
            games_per_pair=1
        )
        self.learner = learner
        self.seller = seller
        self.buyer = buyer
        self.other_social_behavior = other_social_behavior

    def run(self,
            social_behavior: str) -> Tuple[List[str], float]:
        if self.learner == "seller":
            social_behaviors = [social_behavior, self.other_social_behavior]
        else:
            social_behaviors = [self.other_social_behavior, social_behavior]
        games_states = self.runner.run(
            player_pairs=[(self.seller, self.buyer)],
            social_behaviors=social_behaviors,
            logs_dir=self.logs_dir,
        )
        if "fail" in games_states[0]:
            return [COMMUNICATION_PROTOCOL_ERROR], -10
        game_states = games_states[0]["game_state"]
        players = games_states[0]["players"]
        summary_state, players_obs = game_state_parser(game_states=game_states,
                                                       players=players)

        players_obs, reward = self._get_obs_and_rewards(
            summary_state=summary_state,
            players_obs=players_obs,
            players=players
        )
        return players_obs, reward

    def reset(self) -> None:
        self.runner = None
        self.learner = None
        self.seller = None
        self.buyer = None

    def _get_obs_and_rewards(self,
                             summary_state: dict,
                             players_obs: dict,
                             players: dict):
        names = [players[0]["agent_name"], players[1]["agent_name"]]

        beliefs = summary_state["summary"]["player_valuation"]
        player_one_belief = beliefs[0].json()["_value"]["X"]
        player_two_belief = beliefs[1].json()["_value"]["X"]
        zopa = player_two_belief - player_one_belief

        beliefs_str = (
            f"# END OF THE GAME - SUMMARY\n"
            f"{names[0]} had a prior belief of {player_one_belief}\n"
            f"{names[1]} had a prior belief of {player_two_belief}\n"
            f"Zone Of Possible Agreement {zopa}"
        )

        players_obs[0].append(beliefs_str)
        players_obs[1].append(beliefs_str)

        player_outcomes = summary_state["summary"]["player_outcome"]
        if self.learner == "seller":
            reward = player_outcomes[0] / zopa
            obs = players_obs[0]
        else:
            reward = player_outcomes[1] / zopa
            obs = players_obs[1]

        return obs, reward
