import os
import json
from player_tracker import PlayerTracker
from role_assignment_refactored import RoleAssigner

def process_action(action_name: str, input_path: str):
    # Player Detection
    use_cache = False

    player_detections_path = f"player_tracker_results/{action_name}/player_detections.json"
    if use_cache and os.path.exists(player_detections_path):
        print(f"Player detections already exist for {action_name}, using cached results")
        with open(player_detections_path, "r") as f:
            player_detections = json.load(f)
    else:
        player_detections = player_tracker.track_players(input_path, store_results=True)


    # Role Assignment
    use_cache = False

    role_assignment_path = f"role_assignment_results/{action_name}/role_assignments.json"
    if use_cache and os.path.exists(role_assignment_path):
        print(f"Role assignment already exists for {action_name}, using cached results")
        with open(role_assignment_path, "r") as f:
            role_assignment = json.load(f)
    else:
        role_assignment = role_assigner.process_detections(input_path, player_detections, store_results=True, visualize_colors=False)


# Modules
player_tracker = PlayerTracker()
role_assigner = RoleAssigner()

for action_path in os.listdir("input"):
    if action_path == "SNGS-173_video.mp4":
        continue

    input_path = f"input/{action_path}"
    action_name = action_path.split(".")[0]

    action_id = int(action_name.split("_")[1])
    if action_id != 123:
        continue

    print(f"Processing action {action_name}")
    process_action(action_name, input_path)


# action_name = "SNGS-173_video"
# input_path = f"input/{action_name}.mp4"
# process_action(action_name, input_path)
