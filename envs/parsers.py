from typing import List, Tuple

def get_system(player: dict) -> str:
    """
    Gets the system prompt for a given player
    """
    if "Bedrock" in player["class"]:
        system_message = player["_system_blocks"][0]["text"]
    else:
        system_message = player["conversation"][0]["content"]
    return  system_message


def get_player_last_message(player_state: dict, player: dict):
    """
    Gets the last message for a given player
    """
    if "Bedrock" in player["class"]:
        player_message = player_state["conversation"][-1]["content"][0]["text"]
    else:
        player_message = player_state["conversation"][-1]["content"]
    return player_message


def game_state_parser(game_states: List[dict],
                      players: List[dict]) -> Tuple[dict, dict]:
    """
    Transforms games states into egocentric observations.
    """
    players_obs = {
        0: [get_system(player=players[0])],
        1: [get_system(player=players[1])]

    }

    config_state = game_states.pop(0)
    summary_state = game_states.pop()
    for i, player_role in enumerate(config_state["settings"]["player_roles"]):
        if player_role not in players_obs[i][0]:
            players_obs[i][0] += f"\n{player_role}\n"

    for game_state in game_states:
        player_id = game_state["turn"]
        other_player_id = 0 if player_id else 1
        public_info = game_state["player_public_info_dict"]

        player_raw_output = get_player_last_message(
            player_state=game_state["player_state"][player_id],
            player=players[player_id]
        )
        message = public_info["message"]

        player_action = f"# ACTION\n{player_raw_output}"
        other_observation = (f"# OBSERVATION\n"
                             f"<other_player_message> {message} "
                             f"</other_player_message>")
        players_obs[player_id].append(player_action)
        players_obs[other_player_id].append(other_observation)


    return summary_state, players_obs
