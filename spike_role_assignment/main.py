import os
import json
from player_tracker import PlayerTracker
from role_assignment import RoleAssigner

action_name = "action_123_offence"
input_path = f"input/{action_name}.mp4"


# Modules
player_tracker = PlayerTracker()
role_assigner = RoleAssigner()


# Player Detection
player_detections_path = f"player_tracker_results/{action_name}/player_detections.json"
if os.path.exists(player_detections_path):
    print(f"Player detections already exist for {action_name}, using cached results")
    with open(player_detections_path, "r") as f:
        player_detections = json.load(f)
else:
    player_detections = player_tracker.track_players(input_path, store_results=True)


# Role Assignment
role_assignment_path = f"role_assignment_results/{action_name}/role_assignment.json"
if os.path.exists(role_assignment_path):
    print(f"Role assignment already exists for {action_name}, using cached results")
    with open(role_assignment_path, "r") as f:
        role_assignment = json.load(f)
else:
    role_assignment = role_assigner.process_detections(input_path, player_detections)








