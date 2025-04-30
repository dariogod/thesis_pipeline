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
from create_color_clusters3 import create_color_clusters_lab_dynamic

def get_dominant_color(frame, bbox, base_dir, frame_id, y_range=(0.0, 0.5), x_range=(0.0, 1.0)):
    """
    Extract the dominant color from a player's bounding box using K-means clustering in LAB color space.
    
    Args:
        frame: The full video frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        y_range: Vertical range to crop (as percentage of bbox height)
        x_range: Horizontal range to crop (as percentage of bbox width)
        
    Returns:
        tuple: RGB color
    """
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    from skimage import color
    import matplotlib.pyplot as plt
    import os
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract the bounding box region
    x1, y1, x2, y2 = bbox
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    
    if roi.size == 0:
        return (0, 0, 0)
    
    # Apply the y-range and x-range to focus on jersey area
    height, width = roi.shape[:2]
    y_start = int(height * y_range[0])
    y_end = int(height * y_range[1])
    x_start = int(width * x_range[0])
    x_end = int(width * x_range[1])
    
    # Ensure valid crop dimensions
    if y_end <= y_start or x_end <= x_start:
        return (0, 0, 0)
    
    # Convert full ROI to RGB for visualization
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Crop to the region of interest (typically upper body for jersey)
    cropped = roi[y_start:y_end, x_start:x_end]
    
    if cropped.size == 0:
        return (0, 0, 0)
    
    # Convert BGR to RGB
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # Reshape the image for clustering
    pixels = cropped_rgb.reshape(-1, 3)
    
    # Make sure we have enough pixels for clustering
    if len(pixels) < 2:
        return (0, 0, 0)
    
    try:
        # Convert RGB to LAB color space for better perceptual clustering
        # Normalize RGB values to 0-1 range for skimage
        rgb_normalized = pixels.astype(np.float32) / 255.0
        lab_pixels = color.rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Use KMeans to find 2 clusters: pitch (green) and jersey
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(lab_pixels)
        
        # Get the cluster centers and their frequencies
        centers_lab = kmeans.cluster_centers_
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort clusters by count (descending)
        sorted_indices = np.argsort(-counts)
        sorted_counts = counts[sorted_indices]
        
        # Calculate the percentage of each cluster
        total_pixels = np.sum(counts)
        percentages = counts / total_pixels
        
        # Convert LAB centers back to RGB
        centers_lab_reshaped = centers_lab.reshape(1, -1, 3)
        centers_rgb = color.lab2rgb(centers_lab_reshaped).reshape(-1, 3)
        centers_rgb_255 = (centers_rgb * 255).astype(int)
        
        # Sort the centers according to the same order as the counts
        sorted_centers_rgb = centers_rgb[sorted_indices]
        sorted_centers_rgb_255 = centers_rgb_255[sorted_indices]
        sorted_centers_lab = centers_lab[sorted_indices]
        
        # Create visualization directory if it doesn't exist
        # os.makedirs(f'{base_dir}/bbox_color_detections', exist_ok=True)
        
        # # Get track_id from the calling context if available
        # track_id = None
        # import inspect
        # frame = inspect.currentframe()
        # try:
        #     if frame.f_back and 'detection' in frame.f_back.f_locals:
        #         if 'track_id' in frame.f_back.f_locals['detection']:
        #             track_id = frame.f_back.f_locals['detection']['track_id']
        # except:
        #     pass
        
        # # If track_id is not available, use a timestamp
        # if track_id is None:
        #     import time
        #     track_id = f"unknown_{int(time.time())}"
        
        # Create visualization
        # fig = plt.figure(figsize=(18, 12))
        # gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        
        # # Original cropped image with red rectangle showing analyzed region
        # ax0 = plt.subplot(gs[0, 0])
        # ax0.imshow(roi_rgb)
        # rect = plt.Rectangle((x_start-0.5, y_start-0.5), x_end - x_start, y_end - y_start, 
        #                    fill=False, edgecolor='red', linewidth=2)
        # ax0.add_patch(rect)
        # ax0.set_title('Full Player Region\n(Red rectangle shows analyzed area)')
        # ax0.axis('off')
        
        # # 3D scatter plot in LAB space
        # ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        # plot_colors = ['green', 'blue']  # Green for pitch, blue for jersey
        # cluster_names = ['Pitch (Green)', 'Jersey']
        
        # # Sample points to avoid overcrowding the plot
        # max_points = 1000
        # sample_indices = np.random.choice(len(lab_pixels), min(max_points, len(lab_pixels)), replace=False)
        
        # for i, cluster_idx in enumerate(sorted_indices):
        #     mask = labels == cluster_idx
        #     mask_sampled = np.zeros_like(mask)
        #     mask_sampled[sample_indices] = mask[sample_indices]
            
        #     ax1.scatter(
        #         lab_pixels[mask_sampled, 1],  # a* (green-red)
        #         lab_pixels[mask_sampled, 2],  # b* (blue-yellow)
        #         lab_pixels[mask_sampled, 0],  # L* (lightness)
        #         c=plot_colors[i],
        #         marker='o',
        #         s=30,
        #         label=f'{cluster_names[i]} ({percentages[cluster_idx]:.1%})'
        #     )
        
        # # Set the view angle to show a* and b* axes in front
        # ax1.view_init(elev=20, azim=-60)
        
        # ax1.set_xlabel('a* (Green-Red)')
        # ax1.set_ylabel('b* (Blue-Yellow)')
        # ax1.set_zlabel('L* (Lightness)')
        # ax1.set_title('LAB Color Space Clusters')
        # ax1.legend()
        
        # # Display cluster colors
        # for i, cluster_idx in enumerate(sorted_indices):
        #     ax = fig.add_subplot(gs[1, i])
        #     rgb_color = sorted_centers_rgb[i]
        #     rgb_color_255 = sorted_centers_rgb_255[i]
        #     lab_color = sorted_centers_lab[i]
            
        #     # Create a solid color patch using the RGB values
        #     solid_color = np.full((50, 50, 3), rgb_color)
        #     ax.imshow(solid_color)
        #     ax.set_title(f"{cluster_names[i]}\nRGB: {rgb_color_255}")
        #     ax.axis('off')
        
        # plt.tight_layout()
        # plt.savefig(f'{base_dir}/bbox_color_detections/{frame_id:04d}_{track_id}.png', dpi=150)
        # plt.close(fig)
        
        # Return the jersey color (second largest cluster)
        jersey_color = sorted_centers_rgb_255[1] if len(sorted_centers_rgb_255) > 1 else sorted_centers_rgb_255[0]
        return tuple(int(v) for v in jersey_color)
    
    except Exception as e:
        print(f"Error in color clustering: {e}")
        return (0, 0, 0)

def rgb_color_to_hex(rgb_color):
    """Convert RGB color (0-1 range) to hex string."""
    rgb_int = (np.clip(rgb_color, 0, 1) * 255).astype(int)
    return f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}"

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
            
        # Get RGB values from non-overlapping detections
        rgb_values = np.array([color['dominant_color_rgb'] for color in non_overlapping_colors])
        
        # Convert RGB to LAB color space using skimage's method (same as in create_color_clusters.py)
        # First normalize RGB values to 0-1 range
        rgb_values = rgb_values / 255.0 if np.any(rgb_values > 1.0) else rgb_values
        # Convert to LAB using skimage's color module
        from skimage import color
        lab_values = color.rgb2lab(rgb_values.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Use kmeans predict with LAB values
        cluster_assignments = kmeans.predict(lab_values)
        
        # Map cluster indices to team names
        team_assignments = []
        for cluster_idx in cluster_assignments:
            closest_cluster = None
            for i, idx in enumerate(cluster_size_order):
                if idx == cluster_idx:
                    # Make sure the index exists in cluster_mapping
                    if idx in cluster_mapping:
                        closest_cluster = cluster_mapping[idx]
                        break
            
            # Only add valid team assignments
            if closest_cluster is not None:
                team_assignments.append(closest_cluster)
        
        # Use majority voting to determine the final team assignment if we have assignments
        if team_assignments:
            most_common_team = Counter(team_assignments).most_common(1)[0][0]
            track_id_to_team[track_id] = most_common_team
        else:
            # Fallback if no valid team assignments could be made
            track_id_to_team[track_id] = 'Unknown'
    
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

    base_dir = 'intermediate_results3/' + os.path.basename(input_path).split('.')[0]
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
                    dominant_color = get_dominant_color(original_frame, (x1, y1, x2, y2), base_dir, frame_count)
                    
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
    kmeans, cluster_mapping, color_info = create_color_clusters_lab_dynamic(all_player_colors, output_plot_path)
    
    # Calculate cluster sizes and order them
    if hasattr(kmeans, 'labels_'):
        # Count elements in each cluster
        cluster_counts = np.bincount(kmeans.labels_)
        # Get indices sorted by cluster size (descending)
        cluster_size_order = np.argsort(-cluster_counts)
    else:
        # Fallback if labels_ not available
        cluster_size_order = list(range(len(cluster_mapping)))
    
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
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()
