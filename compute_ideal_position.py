#!/usr/bin/env python
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
import math
import os
import cv2
from pathlib import Path
import sys


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distance_to_line(x: float, y: float, minimap_width: int, minimap_height: int) -> float:
    """
    Calculate the distance from a point to the ideal diagonal.
    The ideal diagonal is from bottom left to top right of the pitch.
    """
    # Get the closest point on the diagonal
    closest_x, closest_y = get_closest_point_on_diagonal(x, y, minimap_width, minimap_height)
    
    # Calculate distance to that point
    return euclidean(x, y, closest_x, closest_y)


def get_closest_point_on_diagonal(x: float, y: float, minimap_width: int, minimap_height: int) -> Tuple[float, float]:
    """
    Find the closest point on the ideal diagonal to the given point.
    """
    # Define the diagonal line (from bottom left to top right)
    x1, y1 = 0, minimap_height  # Bottom left
    x2, y2 = minimap_width, 0   # Top right
    
    # Calculate the closest point on the line to the given point
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    
    if det == 0:
        return x1, y1
    
    a = ((x - x1) * dx + (y - y1) * dy) / det
    
    # Clamp a to ensure the point is on the line segment
    a = max(0, min(1, a))
    
    closest_x = x1 + a * dx
    closest_y = y1 + a * dy
    
    return closest_x, closest_y


def get_point_of_action(player_positions: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the centroid of all player positions, which represents
    the point of action on the pitch.
    """
    if not player_positions:
        return 0, 0
    
    # Filter out any extreme outliers that might skew the centroid
    # This is important in sports analysis where some detections might be far off
    
    # Calculate median values for robust centroid approximation
    x_values = [pos[0] for pos in player_positions]
    y_values = [pos[1] for pos in player_positions]
    
    # If we have enough players, remove outliers
    if len(player_positions) > 5:
        # Sort values
        x_values.sort()
        y_values.sort()
        
        # Find quartiles for outlier detection
        q1_x = x_values[len(x_values) // 4]
        q3_x = x_values[3 * len(x_values) // 4]
        q1_y = y_values[len(y_values) // 4]
        q3_y = y_values[3 * len(y_values) // 4]
        
        # Calculate IQR (Interquartile Range)
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y
        
        # Define bounds for outlier detection
        lower_bound_x = q1_x - 1.5 * iqr_x
        upper_bound_x = q3_x + 1.5 * iqr_x
        lower_bound_y = q1_y - 1.5 * iqr_y
        upper_bound_y = q3_y + 1.5 * iqr_y
        
        # Filter positions to exclude outliers
        filtered_positions = [
            (x, y) for x, y in player_positions 
            if (lower_bound_x <= x <= upper_bound_x and lower_bound_y <= y <= upper_bound_y)
        ]
        
        # Use filtered positions if we didn't filter out too many
        if len(filtered_positions) >= len(player_positions) // 2:
            player_positions = filtered_positions
    
    # Calculate mean for the filtered positions
    sum_x = sum(pos[0] for pos in player_positions)
    sum_y = sum(pos[1] for pos in player_positions)
    
    return sum_x / len(player_positions), sum_y / len(player_positions)


def calculate_pitch_dimensions(width: int, height: int) -> dict:
    """
    Calculate pitch dimensions based on image size.
    
    Standard soccer pitch size is 105x68 meters (FIFA regulations).
    We'll scale this to fit our image.
    """
    # Define standard FIFA pitch dimensions (in meters)
    fifa_width = 105
    fifa_height = 68
    
    # Scale factor (assuming the width is the limiting dimension)
    scale = width / fifa_width
    
    return {
        "pitch_width": width,
        "pitch_height": height,
        "scale": scale,
        "fifa_width": fifa_width,
        "fifa_height": fifa_height
    }


def parse_json_data(json_file_path: str) -> Dict[int, List[Dict]]:
    """
    Parse the JSON file and extract frame information.
    
    Args:
        json_file_path: Path to the JSON file with player detections
        
    Returns:
        Dict mapping frame_id to list of player detections
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    frames_dict = {}
    
    for frame_data in data:
        frame_id = frame_data.get("frame_id", 0)
        frames_dict[frame_id] = frame_data.get("detections", [])
    
    return frames_dict


def analyze_ref_positions(json_file_path: str, output_dir: str):
    """
    Analyze referee positions for all frames in a JSON file and export the results.
    
    Args:
        json_file_path: Path to the JSON file containing predictions
        output_dir: Directory to save the results
    """
    # Parse JSON data
    frames_dict = parse_json_data(json_file_path)
    print(f"Found {len(frames_dict)} frames in {json_file_path}")

    minimap = cv2.imread('pitch.jpg')
    minimap_height, minimap_width, _ = minimap.shape
    
    # Analyze each frame
    results = []
    for frame_id, detections in frames_dict.items():
        # Extract referee and player positions
        ref_pos = None
        player_positions = []
        referee_positions = []

        for detection in detections:
            # Use minimap_position if available, otherwise use bbox center
            x_center, y_center = detection["minimap_position"]
            team = detection["team"]

            if team == "Referee":
                ref_pos = (x_center, y_center)
            else:
                player_positions.append((x_center, y_center))
        
        if not player_positions or not referee_positions:
            continue

        referee_pos = None
        if len(referee_positions) > 1:
            print(f"Warning: Found multiple referee positions in frame {frame_id}, taking the closest to the center of the pitch")
            center = (minimap_width / 2, minimap_height / 2)
            min_dist = float('inf')
            for pos in referee_positions:
                dist = euclidean(pos[0], pos[1], center[0], center[1])
                if dist < min_dist:
                    min_dist = dist
                    referee_pos = pos
        else:
            referee_pos = referee_positions[0]
        
        # Calculate point of action (player centroid)
        point_of_action = get_point_of_action(player_positions)


        
        # Calculate distances
        dist_to_diagonal = distance_to_line(ref_pos[0], ref_pos[1], minimap_width, minimap_height)
        dist_to_action = euclidean(ref_pos[0], ref_pos[1], point_of_action[0], point_of_action[1])
        
        # Store results
        results.append({
            "frame_id": frame_id,
            "dist_to_diagonal": dist_to_diagonal,
            "dist_to_point_of_action": dist_to_action
        })
    
    print(f"Analyzed {len(results)} frames with referee positions")
    
    # Calculate statistics
    if results:
        avg_dist_diagonal = sum(r["dist_to_diagonal"] for r in results) / len(results)
        avg_dist_action = sum(r["dist_to_point_of_action"] for r in results) / len(results)
        
        print(f"Average distance to ideal diagonal: {avg_dist_diagonal:.2f}")
        print(f"Average distance to point of action: {avg_dist_action:.2f}")
    
    # Save results
    output_file = os.path.join(output_dir, 'ref_position_analysis.json')
    if results:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        
        # Generate plots
        plot_results(results, os.path.join(output_dir, 'analysis'))


def plot_results(results: List[Dict], output_prefix: str):
    """
    Generate plots visualizing the analysis results.
    
    Args:
        results: List of analysis results per frame
        output_prefix: Prefix for output file paths
    """
    # Extract data for plotting
    frame_ids = [r["frame_id"] for r in results]
    dist_diagonals = [r["dist_to_diagonal"] for r in results]
    dist_actions = [r["dist_to_point_of_action"] for r in results]
    
    # Convert frame_ids to numeric for better x-axis
    x = list(range(len(frame_ids)))
    
    # Plot distances over time
    plt.figure(figsize=(12, 6))
    plt.plot(x, dist_diagonals, label='Distance to Ideal Diagonal')
    plt.plot(x, dist_actions, label='Distance to Player Centroid')
    plt.xlabel('Frame Number')
    plt.ylabel('Distance')
    plt.title('Referee Positioning Analysis')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f"{output_prefix}_distances.png")
    print(f"Plot saved to {output_prefix}_distances.png")
    
    # Plot histogram of distances
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(dist_diagonals, bins=20, alpha=0.7)
    plt.xlabel('Distance to Ideal Diagonal')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Ideal Diagonal')
    
    plt.subplot(1, 2, 2)
    plt.hist(dist_actions, bins=20, alpha=0.7)
    plt.xlabel('Distance to Player Centroid')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Player Centroid')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_histograms.png")
    print(f"Histograms saved to {output_prefix}_histograms.png")


def create_augmented_minimaps(json_file_path: str, output_dir: str):
    """
    Create enhanced minimaps showing the ideal diagonal and player centroid,
    saving them next to the original minimap frames.
    
    Args:
        json_file_path: Path to the JSON file with player detections
        output_dir: Directory for output (minimap dir is a subdirectory)
    """
    minimap_dir = os.path.join(output_dir, 'minimap')
    # Parse JSON data
    frames_dict = parse_json_data(json_file_path)
    print(f"Processing {len(frames_dict)} frames")
    
    # Process each frame
    for frame_id, detections in frames_dict.items():
        # Load pitch.jpg as minimap
        minimap = cv2.imread('pitch.jpg')
        minimap_height, minimap_width, _ = minimap.shape
        
        # Extract player positions (excluding referee)
        player_positions = []
        for detection in detections:
            team = detection["team"]

            x = int(detection["minimap_position"][0])
            y = int(detection["minimap_position"][1])

            # Draw player position as black dot
            circle_radius = max(2, int(minimap_width / 115))
            cv2.circle(minimap, (x, y), circle_radius, (0, 0, 0), -1)
            player_positions.append((x, y))
        
        # Calculate player centroid
        if player_positions:
            centroid_x, centroid_y = get_point_of_action(player_positions)
            
            # Draw cross at player centroid (red)
            cross_size = max(10, int(minimap_width * 0.01))
            cv2.line(minimap, 
                    (int(centroid_x - cross_size), int(centroid_y)), 
                    (int(centroid_x + cross_size), int(centroid_y)), 
                    (0, 0, 255), 2)
            cv2.line(minimap, 
                    (int(centroid_x), int(centroid_y - cross_size)), 
                    (int(centroid_x), int(centroid_y + cross_size)), 
                    (0, 0, 255), 2)
        
        # Draw ideal diagonal (green) from bottom left to top right
        cv2.line(minimap, (0, minimap_height), (minimap_width, 0), (0, 255, 0), 2)
        
        # Optionally, draw referee position if available (yellow)
        for detection in detections:
            team = detection["team"]
            if team == "Referee":  # This is the referee
                # Use minimap_pos if available
                ref_x = int(detection["minimap_position"][0])
                ref_y = int(detection["minimap_position"][1])

                circle_radius = max(2, int(minimap_width / 115))
                cv2.circle(minimap, (ref_x, ref_y), circle_radius, (0, 255, 255), -1)
        
        # Save enhanced minimap next to the original one with "_augmented" suffix
        output_filename = f'frame_{frame_id:06d}_augmented.jpg'
        output_path = os.path.join(minimap_dir, output_filename)
        cv2.imwrite(output_path, minimap)
        
        # Print progress
        if int(frame_id) % 50 == 0:
            print(f"Processed frame {frame_id}")
    
    print(f"Enhanced minimaps saved to {minimap_dir} with '_augmented' suffix")
    
    # Create a video from the enhanced minimaps
    create_video_from_frames(minimap_dir, os.path.join(output_dir, 'minimap_augmented.mp4'), suffix="_augmented")
    

def create_video_from_frames(frames_dir: str, output_path: str, fps: int = 30, suffix: str = ""):
    """
    Create a video from a sequence of image frames.
    
    Args:
        frames_dir: Directory containing the image frames
        output_path: Path to save the output video
        fps: Frames per second for the output video
        suffix: Suffix to filter frame files (e.g., "_augmented")
    """
    # Get frame files
    if suffix:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and suffix in f and f.endswith('.jpg')])
    else:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    
    if not frame_files:
        print(f"Error: No frame files found in {frames_dir} with suffix '{suffix}'")
        return
    
    # Read the first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    print(f"Creating video from {len(frame_files)} frames...")
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video_writer.write(frame)
    
    # Release video writer
    video_writer.release()
    print(f"Video created: {output_path}")


def detect_minimap_dir(json_file_path: str) -> str:
    """
    Attempt to detect the minimap directory based on the JSON file path.
    
    Args:
        json_file_path: Path to the JSON file with player detections
        
    Returns:
        Detected minimap directory path
    """
    # Try to determine the minimap directory from the JSON path
    base_dir = os.path.dirname(json_file_path)
    
    # Check if the json is in a video-specific directory
    if os.path.basename(base_dir) == 'player_detections':
        video_dir = os.path.dirname(base_dir)
        video_name = os.path.basename(video_dir)
        minimap_dir = os.path.join(video_dir, 'minimap')
    elif 'intermediate_results' in base_dir:
        # Try to extract video name from path
        parts = base_dir.split(os.path.sep)
        for i, part in enumerate(parts):
            if part.startswith('intermediate_results'):
                if i+1 < len(parts):
                    video_name = parts[i+1]
                    minimap_dir = os.path.join(base_dir, '..', video_name, 'minimap')
                    minimap_dir = os.path.normpath(minimap_dir)
                    break
        else:
            # Default: Assume json file is in the base directory
            video_name = os.path.splitext(os.path.basename(json_file_path))[0]
            minimap_dir = os.path.join('intermediate_results2', video_name, 'minimap')
    else:
        # Default: Assume json file is in the base directory
        video_name = os.path.splitext(os.path.basename(json_file_path))[0]
        minimap_dir = os.path.join('intermediate_results2', video_name, 'minimap')
    
    # Check if the directory exists
    if os.path.exists(minimap_dir):
        return minimap_dir
    
    # Try alternative path with 'intermediate_results2'
    minimap_dir = minimap_dir.replace('intermediate_results', 'intermediate_results2')
    if os.path.exists(minimap_dir):
        return minimap_dir
    
    # Try alternative path with 'intermediate_results'
    minimap_dir = minimap_dir.replace('intermediate_results2', 'intermediate_results')
    if os.path.exists(minimap_dir):
        return minimap_dir
    
    raise FileNotFoundError(f"Could not locate minimap directory for {json_file_path}")


def main(json_file: str):
    """
    Main function to process a JSON file with player detections and create
    augmented minimaps.
    
    Args:
        json_file: Path to the JSON file with player detections
    """
    try:
        output_dir = os.path.dirname(json_file)

        minimap_dir = os.path.join(output_dir, 'minimap')
        print(f"Using minimap directory: {minimap_dir}")
        
        # Analyze referee positions and generate plots
        analyze_ref_positions(json_file, output_dir)
        
        # Create augmented minimaps
        create_augmented_minimaps(json_file, output_dir)
        
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze referee positioning and create enhanced minimaps')
    parser.add_argument('json_file', help='Path to the JSON file with player detections')
    
    args = parser.parse_args()
    main(args.json_file)
