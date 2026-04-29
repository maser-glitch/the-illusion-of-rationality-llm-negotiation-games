import glob
import pandas as pd
from typing import List
from pathlib import Path
from utils.file_management import load_json


def get_player_model(player_data: dict) -> str | None:
    """Get the model used by a given player from its log data"""
    for k in ["model"]:
        if (k in player_data and isinstance(player_data[k], str)
                and player_data[k].strip()):
            return player_data[k].strip()
    return player_data.get("class", None)

def buy_sell_get_first_proposal(game_states: List[dict]) -> int | None:
    """Get the first proposal of a player in the buyer-seller scenario"""
    for i in range(1, len(game_states)):
        state = game_states[i]
        if "player_public_info_dict" in state:
            public_info = state["player_public_info_dict"]
            if public_info["player answer"] == "PROPOSAL":
                return public_info["newly proposed trade"]["_value"]["BLUE"][
                          "_value"]["ZUP"]

def buy_sell_get_proposals(game_states: List[dict]) -> List[int]:
    """Get the all the proposals done during a simulation in the buy and sell
      game"""
    proposals = []
    for i in range(1, len(game_states)):
        state = game_states[i]
        if "player_public_info_dict" in state:
            public_info = state["player_public_info_dict"]
            if public_info["player answer"] == "PROPOSAL":
                proposal_value = \
                public_info["newly proposed trade"]["_value"]["BLUE"][
                    "_value"]["ZUP"]
                proposals.append(proposal_value)
    return proposals

def buy_sell_logs_to_df(log_name: str) -> pd.DataFrame:
    """Transforms buy and sell logs into a dataframe """
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / ".logs" / log_name
    games_state_files = glob.glob(str(logs_dir / "*/game_state.json"))
    rows = []
    for game_state_file in games_state_files:
        row = {
            "log_path": game_state_file,
            "seller_model": None,
            "buyer_model": None,
            "seller_valuation": None,
            "buyer_valuation": None,
            "final_response": None,
            "price": None,
            "seller_payoff": None,
            "buyer_payoff": None,
        }
        try:
            game_state = load_json(game_state_file)

            row["seller_model"] = get_player_model(game_state["players"][0])
            row["buyer_model"] = get_player_model(game_state["players"][1])
            row["seller_social_behavior"] = \
                game_state["player_social_behaviour"][0]
            row["buyer_social_behavior"] = \
            game_state["player_social_behaviour"][1]
            row["seller_valuation"] = \
            game_state["player_goals"][0]["_value"]["_value"]["_value"]["X"]
            row["buyer_valuation"] = \
            game_state["player_goals"][1]["_value"]["_value"]["_value"]["X"]

            row["first_proposal"] = buy_sell_get_first_proposal(
                game_states=game_state["game_state"]
            )
            row["proposals"] = buy_sell_get_proposals(
                game_states=game_state["game_state"]
            )
            # Summary
            summary = game_state["game_state"][-1]["summary"]
            row["final_response"] = str(
                summary.get("final_response", "")).upper() or None

            if row["final_response"] == "ACCEPT":
                trade = summary["proposed_trade"]["_value"]["BLUE"]["_value"]
                row["price"] = trade.get("ZUP", None)

            if ("player_outcome" in summary and isinstance(
                    summary["player_outcome"],
                    (list, tuple)) and
                    len(summary["player_outcome"]) == 2):
                row["seller_payoff"], row["buyer_payoff"] = summary[
                    "player_outcome"]

            rows.append(row)

        except Exception:
            print(f"Error computing loading log {game_state_file}")

    return pd.DataFrame(rows)


def trading_purify_outcomes(outcome: dict):
    """
    For some reason, sometimes, the library returns corrupted outcomes.
    For instance: {'X': 0, 'Y': 0, 'item Y': 5, 'item X': -7}.
    This function purifies these cases by keeping the non-zero elements while
    keeping the original names X and Y
    """
    if len(outcome) > 2:

        x_keys = [k for k in outcome if 'x' in k.lower()]
        y_keys = [k for k in outcome if 'y' in k.lower()]

        purified = {
            "X": sum(outcome[k] for k in x_keys),
            "Y": sum(outcome[k] for k in y_keys)
        }
        return purified

    return outcome

def trading_logs_to_df(log_name: str) -> pd.DataFrame:
    """Transform trading logs into a pdf"""
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / ".logs" / log_name
    games_state_files = glob.glob(str(logs_dir / "*/game_state.json"))
    rows = []
    for game_state_file in games_state_files:
        try:
            row = {"log_path": game_state_file}
            game_state = load_json(game_state_file)
            players = game_state["players"]
            row["player_one_model"] = get_player_model(players[0])
            row["player_two_model"] = get_player_model(players[1])
            states = game_state["game_state"]
            outcomes = states[-1]["summary"]["player_outcome"]
            row["player_one_outcome"] = trading_purify_outcomes(
                outcomes[0]["_value"]
            )
            row["player_two_outcome"] = trading_purify_outcomes(
                outcomes[1]["_value"]
            )
            row["player_one_delta"] = sum(row["player_one_outcome"].values())
            row["player_two_delta"] = sum(row["player_two_outcome"].values())

            row["player_one_win"] = (row["player_one_delta"] >
                                     row["player_two_delta"])
            row["player_two_win"] = (row["player_one_delta"] <
                                     row["player_two_delta"])
            row["draw"] = row["player_one_delta"] == row["player_two_delta"]
            rows.append(row)
        except Exception:
            print(f"Error computing loading log {game_state_file}")
    return pd.DataFrame(rows)


def ultimatum_logs_to_df(log_name: str) -> pd.DataFrame:
    """Transforms ultimatum logs into a pdf"""
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / ".logs" / log_name
    games_state_files = glob.glob(str(logs_dir / "*/game_state.json"))
    rows = []
    for game_state_file in games_state_files:
        try:
            row = {"log_path": game_state_file}
            game_state = load_json(game_state_file)
            players = game_state["players"]
            row["player_one_model"] = get_player_model(players[0])
            row["player_two_model"] = get_player_model(players[1])
            states = game_state["game_state"]
            outcomes = states[-1]["summary"]["player_outcome"]
            row["player_one_outcome"] = outcomes[0]["_value"]
            row["player_two_outcome"] = outcomes[1]["_value"]

            row["player_one_delta"] = sum(row["player_one_outcome"].values())
            row["player_two_delta"] = sum(row["player_two_outcome"].values())

            row["player_one_win"] = (row["player_one_delta"] >
                                     row["player_two_delta"])
            row["player_two_win"] = (row["player_one_delta"] <
                                     row["player_two_delta"])
            row["draw"] = row["player_one_delta"] == row["player_two_delta"]
            rows.append(row)
        except Exception:
            print(f"Error computing loading log {game_state_file}")
    return pd.DataFrame(rows)
