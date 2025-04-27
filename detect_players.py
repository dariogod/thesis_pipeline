import json
import numpy as np
import cv2
from PIL import Image
import os
import argparse
from sklearn.cluster import KMeans
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter
from create_color_clusters import create_color_clusters_basic, create_color_clusters_lab

# def get_dominant_color(frame, bbox, top_percent=0.5, middle_x_percent=0.8, n_clusters=3, pixel_threshold=50):
#     x1, y1, x2, y2 = bbox
#     h = y2 - y1
#     w = x2 - x1

#     y_start = y1
#     y_end = y1 + int(h * top_percent)
#     x_start = x1 + int(w * (1 - middle_x_percent) / 2)
#     x_end = x2 - int(w * (1 - middle_x_percent) / 2)

#     cropped = frame[y_start:y_end, x_start:x_end]
    
#     # Convert BGR to RGB and then to HSV
#     rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
#     hsv = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2HSV)

#     # Initial green mask
#     lower_green = np.array([35, 40, 40])
#     upper_green = np.array([85, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     non_green_pixels = rgb_cropped[mask == 0]

#     # If not enough non-green pixels, relax the green mask
#     if len(non_green_pixels) < pixel_threshold:
#         lower_green = np.array([40, 50, 50])
#         upper_green = np.array([80, 255, 255])
#         mask = cv2.inRange(hsv, lower_green, upper_green)
#         non_green_pixels = rgb_cropped[mask == 0]

#     # Fallback if still too few
#     if len(non_green_pixels) < n_clusters:
#         non_green_pixels = rgb_cropped.reshape(-1, 3)

#     try:
#         kmeans = KMeans(n_clusters=n_clusters, n_init=10)
#         kmeans.fit(non_green_pixels)
#         cluster_centers = kmeans.cluster_centers_.astype(int)
#         labels, counts = np.unique(kmeans.labels_, return_counts=True)
#         # Convert numpy int values to Python int in the tuple
#         dominant_color = tuple(int(v) for v in cluster_centers[np.argmax(counts)])
#     except Exception:
#         dominant_color = (0, 0, 0)

#     return dominant_color

def get_dominant_color(frame, bbox, y_range=(0.1, 0.4), x_range=(0.1, 0.9), n_clusters=5, pixel_threshold=10):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    
    # Use specified ranges to crop the player
    y_start = y1 + int(h * y_range[0])
    y_end = y1 + int(h * y_range[1])
    x_start = x1 + int(w * x_range[0])
    x_end = x1 + int(w * x_range[1])
    
    # Make sure we have valid crop dimensions
    if y_end <= y_start or x_end <= x_start or y_end > frame.shape[0] or x_end > frame.shape[1]:
        return (0, 0, 0)
    
    cropped = frame[y_start:y_end, x_start:x_end]
    
    # Make sure we have a non-empty crop
    if cropped.size == 0:
        return (0, 0, 0)
    
    # Convert BGR to RGB and then to HSV
    rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2HSV)
    
    # Create multiple masks to better handle various field colors
    # Standard green field mask
    lower_green1 = np.array([35, 40, 40])
    upper_green1 = np.array([85, 255, 255])
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # Yellowish-green field mask
    lower_green2 = np.array([25, 40, 40])
    upper_green2 = np.array([35, 255, 255])
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    
    # Bluish-green field mask (for some artificial turfs)
    lower_green3 = np.array([85, 40, 40])
    upper_green3 = np.array([95, 255, 255])
    mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
    
    # Combine all masks
    combined_mask = mask1 | mask2 | mask3
    
    # Get non-green pixels
    non_green_pixels = rgb_cropped[combined_mask == 0]
    
    # If not enough non-green pixels, try a different approach - use brightness to filter
    if len(non_green_pixels) < pixel_threshold:
        # Get brighter pixels (likely to be jersey in well-lit conditions)
        brightness_mask = hsv[:,:,2] > 80
        bright_pixels = rgb_cropped[brightness_mask]
        
        if len(bright_pixels) >= pixel_threshold:
            non_green_pixels = bright_pixels
        else:
            # Just use all pixels as a fallback
            non_green_pixels = rgb_cropped.reshape(-1, 3)
    
    # Make sure we have enough pixels for clustering
    if len(non_green_pixels) < n_clusters:
        return (0, 0, 0)
    
    try:
        # Use more clusters for better color detection
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(non_green_pixels)
        cluster_centers = kmeans.cluster_centers_.astype(int)
        
        # Find the most dominant non-background color
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort clusters by count (descending)
        sorted_indices = np.argsort(-counts)
        sorted_centers = cluster_centers[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        # Choose the most saturated color among the top 3 most frequent clusters
        top_n = min(3, len(sorted_centers))
        max_saturation = 0
        dominant_color = sorted_centers[0]  # Default to most frequent
        
        for i in range(top_n):
            # Convert RGB to HSV to check saturation
            rgb_color = sorted_centers[i]
            hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
            
            # Higher weight for more frequent colors, but also consider saturation
            weighted_saturation = hsv_color[1] * (sorted_counts[i] / sorted_counts[0]) 
            
            if weighted_saturation > max_saturation:
                max_saturation = weighted_saturation
                dominant_color = rgb_color
        
        # Convert numpy int values to Python int in the tuple
        return tuple(int(v) for v in dominant_color)
    
    except Exception as e:
        print(f"Error in color clustering: {e}")
        return (0, 0, 0)

def get_bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def get_overlapping_bboxes(frame):
    detections = frame['detections']
    overlapping_pairs = []
    
    for i, detection1 in enumerate(detections):
        bbox1 = detection1['bbox']
        x1_min, y1_min, x1_max, y1_max = bbox1
        
        for j, detection2 in enumerate(detections[i+1:], i+1):
            bbox2 = detection2['bbox']
            x2_min, y2_min, x2_max, y2_max = bbox2
            
            # Check if bboxes overlap
            if not (x1_max < x2_min or x2_max < x1_min or 
                   y1_max < y2_min or y2_max < y1_min):
                overlapping_pairs.append((detection1['track_id'], detection2['track_id']))
                
    return overlapping_pairs

def process_player_detections(data):
    # Extract all unique track IDs
    all_track_ids = []
    for frame in data:
        for detection in frame['detections']:
            all_track_ids.append(detection['track_id'])
    
    all_unique_track_ids = list(set(all_track_ids))
    
    # Collect colors for each track ID
    track_id_to_colors = {}
    for track_id in all_unique_track_ids:
        track_id_to_colors[track_id] = []

        for frame in data:
            overlapping_pairs = get_overlapping_bboxes(frame)
            
            overlapping_track_ids = []
            for i, j in overlapping_pairs:
                overlapping_track_ids.append(i)
                overlapping_track_ids.append(j)
            overlapping_track_ids = list(set(overlapping_track_ids))
            
            for detection in frame['detections']:
                if detection['track_id'] == track_id:
                    track_id_to_colors[track_id].append(
                        {
                            "frame_id": frame['frame_id'],
                            "dominant_color_rgb": detection['dominant_color_rgb'], 
                            "bbox_size": get_bbox_size(detection['bbox']),
                            "has_overlap": track_id in overlapping_track_ids
                        }
                    )
    
    # Calculate confidence for each color detection
    for track_id, colors in track_id_to_colors.items():
        if len(colors) == 0:
            continue
        
        bbox_sizes = [color['bbox_size'] for color in colors]
        max_bbox_size = max(bbox_sizes)

        for color in colors:
            color["confidence"] = 1 - (color["bbox_size"] / max_bbox_size)
            if color["has_overlap"]:
                color["confidence"] = 0
    
    # Collect all non-overlapping player colors
    all_player_colors = []
    for track_id, colors in track_id_to_colors.items():
        if len(colors) == 0:
            continue
            
        non_overlapping_colors = [color for color in colors if not color['has_overlap']]
        
        if len(non_overlapping_colors) == 0:
            continue

        all_player_colors.extend(non_overlapping_colors)
    
    return all_player_colors, track_id_to_colors, all_unique_track_ids

def assign_teams_to_players(track_id_to_colors, kmeans, cluster_mapping, cluster_size_order):
    track_id_to_team = {}
    for track_id, colors in track_id_to_colors.items():
        if len(colors) == 0:
            continue
            
        # Filter out colors with overlap
        non_overlapping_colors = [color for color in colors if not color['has_overlap']]
        
        if len(non_overlapping_colors) == 0:
            continue
            
        # Calculate average RGB color from non-overlapping detections only
        rgb_values = np.array([color['dominant_color_rgb'] for color in non_overlapping_colors])

        team_assignments = []
        for rgb_color in rgb_values:
            # Normalize RGB values to [0,1] range
            rgb_color = [x/255.0 for x in rgb_color]
            
            # Find closest cluster center
            distances = np.linalg.norm(kmeans.cluster_centers_ - rgb_color, axis=1)
            closest_cluster_idx = np.argmin(distances)
            # Map to the team based on the cluster
            for i, idx in enumerate(cluster_size_order):
                if idx == closest_cluster_idx:
                    closest_cluster = cluster_mapping[idx]
                    break
            team_assignments.append(closest_cluster)

        # Use majority voting to determine the final team assignment
        most_common_team = Counter(team_assignments).most_common(1)[0][0]
        track_id_to_team[track_id] = most_common_team
    
    return track_id_to_team

def draw_color_legend(frame, track_id_colors, original_width, legend_width):
    # Define legend parameters
    square_size = 20
    padding = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    items_per_column = 7
    column_width = 60
    
    # Calculate how many rows we need for all track IDs
    track_ids = sorted(track_id_colors.keys())
    num_tracks = len(track_ids)
    num_rows = min(items_per_column, num_tracks)
    
    # Create a blank legend area on the right
    legend_height = min(num_rows * (square_size + 10) + 2 * padding, frame.shape[0])
    legend_y_start = (frame.shape[0] - legend_height) // 2
    
    # Draw legend background
    cv2.rectangle(frame, 
                 (original_width, legend_y_start), 
                 (original_width + legend_width, legend_y_start + legend_height), 
                 (240, 240, 240), -1)
    
    # Draw entries for each track ID in three columns
    for i, track_id in enumerate(track_ids):
        color = track_id_colors[track_id]
        
        # Calculate position based on which column and row
        column = i // num_rows
        row = i % num_rows
        
        if column >= 3:  # Only display up to 3 columns
            break
            
        # Column starting position
        col_x = original_width + padding + (column * column_width)
        
        # Row position
        y_pos = legend_y_start + padding + row * (square_size + 10)
        
        # Convert RGB to BGR for OpenCV
        bgr_color = (color[2], color[1], color[0])
        
        # Draw color square
        cv2.rectangle(frame, 
                     (col_x, y_pos), 
                     (col_x + square_size, y_pos + square_size), 
                     bgr_color, -1)
        
        # Draw ID text next to the square
        id_text = f"{track_id}"
        cv2.putText(frame, id_text, 
                   (col_x + square_size + 5, y_pos + square_size - 5), 
                   font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return frame

def process_video(input_path):
    os.makedirs('models', exist_ok=True)

    model_path = 'models/yolov8n.pt'
    
    # Check if model exists, if not it will be downloaded to the models directory
    if not os.path.exists(model_path) and not os.path.isabs(model_path):
        model_path = os.path.join('models', os.path.basename(model_path))
    
    # Load a pre-trained YOLO model
    model = YOLO(model_path)  # contains 'person' class

    # Open the input soccer video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define new width to include the legend
    legend_width = 250
    new_width = width + legend_width

    base_dir = 'intermediate_results/' + os.path.basename(input_path).split('.')[0]
    os.makedirs(base_dir, exist_ok=True)

    output_video_path = f"{base_dir}/player_detections.mp4"
    output_json_path = f"{base_dir}/player_detections.json"
    output_team_json_path = f"{base_dir}/player_detections_with_team.json"
    output_plot_path = f"{base_dir}/player_color_clusters.png"
    frames_dir = os.path.join(base_dir, 'frames_rgb')
    
    original_frames_dir = os.path.join(base_dir, 'frames')
    
    # Define video writer with the new width to accommodate the legend
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (new_width, height))

    # Create directories if they don't exist
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(original_frames_dir, exist_ok=True)

    # List to store all detections
    all_detections = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video
        
        # Save the original untouched frame
        original_frame_filename = os.path.join(original_frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(original_frame_filename, frame)
        
        # Create a copy of the original frame for color detection
        original_frame = frame.copy()
        
        # Use YOLO to detect and track persons in the frame
        results = model.track(frame, persist=True, verbose=False)

        frame_detections = []
        
        # For the color legend
        frame_track_id_colors = {}
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if model.names[cls_id] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                    
                    # Get dominant RGB color from the original unmodified frame
                    dominant_color = get_dominant_color(original_frame, (x1, y1, x2, y2))
                    
                    # Store color for legend
                    if track_id is not None:
                        frame_track_id_colors[track_id] = dominant_color
                    
                    # Draw bounding box
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_8)
                    
                    # Draw only track ID text
                    label = f"{track_id}"
                    font_scale = 0.9
                    font_thickness = 1
                    font = cv2.FONT_HERSHEY_PLAIN

                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    cv2.rectangle(frame, 
                                (x1, y1 - text_height - 10), 
                                (x1 + text_width + 10, y1), 
                                (0, 0, 0), -1)
                    cv2.putText(frame, 
                                label, 
                                (x1+5, y1 - 5), 
                                font, font_scale, 
                                (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'track_id': int(track_id) if track_id is not None else None,
                        'class': model.names[cls_id],
                        'dominant_color_rgb': [int(c) for c in dominant_color]
                    }
                    frame_detections.append(detection)
        
        # Create a wider frame to include the legend
        wide_frame = np.zeros((height, new_width, 3), dtype=np.uint8)
        wide_frame[:, :width] = frame
        
        # Draw the color legend on the wider frame
        if frame_track_id_colors:
            wide_frame = draw_color_legend(wide_frame, frame_track_id_colors, width, legend_width)

        all_detections.append({
            'frame_id': int(frame_count),
            'detections': frame_detections
        })
        frame_count += 1
        
        out.write(wide_frame)
        
        # Save processed frame with bounding boxes
        processed_frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(processed_frame_filename, wide_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save all detections to JSON
    with open(output_json_path, 'w') as f:
        json.dump(all_detections, f, indent=4)
        
    # Process player detections to assign teams
    all_player_colors, track_id_to_colors, all_unique_track_ids = process_player_detections(all_detections)
    
    # Create color clusters
    kmeans, cluster_mapping, cluster_size_order = create_color_clusters_lab(all_player_colors, output_plot_path)
    
    # Assign teams to players
    track_id_to_team = assign_teams_to_players(track_id_to_colors, kmeans, cluster_mapping, cluster_size_order)
    
    # Add team information to the original detections
    for frame in all_detections:
        for detection in frame['detections']:
            track_id = detection['track_id']
            if track_id in track_id_to_team:
                detection['team'] = track_id_to_team[track_id]
            else:
                detection['team'] = 'Unknown'
    
    # Save the updated detections to JSON
    with open(output_team_json_path, 'w') as f:
        json.dump(all_detections, f, indent=4)
        
    print(f"Processed {len(track_id_to_team)} players. Team assignments saved successfully.")

    return frame_count

def main():
    parser = argparse.ArgumentParser(description='Detect players in a soccer video')
    parser.add_argument('input', help='Path to input video file')
    
    args = parser.parse_args()
    
    try:
        frame_count = process_video(
            args.input, 
        )
        print(f"Processed {frame_count} frames. Detections saved successfully.")
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
