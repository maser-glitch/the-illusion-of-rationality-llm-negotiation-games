from utils.file_management import load_json

def get_system(player: dict) -> str:
    if "Bedrock" in player["class"]:
        system_message = player["_system_blocks"][0]["text"]
    else:
        system_message = player["conversation"][0]["content"]
    return  system_message


def get_player_last_message(player_state: dict, player: dict):
    if "Bedrock" in player["class"]:
        player_message = player_state["conversation"][-1]["content"][0]["text"]
    else:
        player_message = player_state["conversation"][-1]["content"]
    return player_message

log_path = "/Users/manrios/Documents/projects/negotiation-strategies/.logs/buysell-experiment-2/1762021162301/game_state.json"

game_states = load_json(log_path)["game_state"]
players = load_json(log_path)["players"]
names = [players[0]["agent_name"], players[1]["agent_name"]]
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

beliefs = summary_state["summary"]["player_valuation"]
player_one_belief = beliefs[0]["_value"]["X"]
player_two_belief = beliefs[1]["_value"]["X"]
zopa = player_two_belief - player_one_belief

beliefs_str = (f"# END OF THE GAME - SUMMARY"
               f"{names[0]} had a prior belief of {player_one_belief}\n"
               f"{names[1]} had a prior belief of {player_two_belief}\n"
               f"Zone Of Possible Agreement {zopa}")

players_obs[0].append(beliefs_str)
players_obs[1].append(beliefs_str)

player_outcomes = summary_state["summary"]["player_outcome"]
player_one_reward = player_outcomes[0]/zopa
player_two_reward = player_outcomes[1]/zopa

print("Player One Reward", player_one_reward)
print("Player Two Reward", player_two_reward)


