import os
import json
import argparse
from create_minimap import transform_matrix
import cv2

def main(player_detections_path, homography_path):
    # Load player detections
    with open(player_detections_path, 'r') as f:
        player_detections = json.load(f)
    
    # Load homography data
    with open(homography_path, 'r') as f:
        homography_data = json.load(f)
    
    # Load pitch.jpg to get dimensions
    gt_img = cv2.imread('pitch.jpg')
    gt_h, gt_w, _ = gt_img.shape
    
    # Process each frame in player detections
    for frame_data in player_detections:
        frame_id = frame_data['frame_id']
        frame_id_str = str(frame_id)
        
        # Get homography matrix for this frame (or closest available)
        M = None
        if frame_id_str in homography_data:
            M = homography_data[frame_id_str]
        else:
            # Find closest frame with homography data
            frame_ids = [int(k) for k in homography_data.keys()]
            if frame_ids:
                closest_frame = min(frame_ids, key=lambda x: abs(x - frame_id))
                M = homography_data[str(closest_frame)]
        
        if M is None:
            print(f"Warning: No homography matrix found for frame {frame_id}")
            continue
        
        # Get video dimensions from first detection if available
        if 'detections' in frame_data and frame_data['detections']:
            # Assuming all frames have same dimensions
            detection = frame_data['detections'][0]
            x1, y1, x2, y2 = detection['bbox']
            # Estimate frame size based on bbox coordinates
            w = 398
            h = 224
            
            # Process each detection
            for detection in frame_data['detections']:
                x1, y1, x2, y2 = detection['bbox']
                # Use bottom center of bounding box
                center_x = x1 + (x2 - x1)/2
                center_y = y1 + (y2 - y1)
                
                # Transform coordinates to minimap
                coords = transform_matrix(M, (center_x, center_y), (h, w), (gt_h, gt_w))
                
                # Add minimap position to detection
                detection['minimap_position'] = coords
    
    # Save updated detections to new file
    print(player_detections)
    output_path = os.path.splitext(player_detections_path)[0] + '_with_minimap.json'
    with open(output_path, 'w') as f:
        json.dump(player_detections, f, indent=2)
    
    print(f"Updated detections saved to {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add minimap positions to player detections')
    parser.add_argument('--player_detections', required=True, help='Path to player detections JSON file')
    parser.add_argument('--homography', required=True, help='Path to homography JSON file')
    args = parser.parse_args()
    
    try:
        output_file = main(args.player_detections, args.homography)
        print(f"Successfully processed player detections and saved to {output_file}")
    except Exception as e:
        print(f"Error processing detections: {e}") 