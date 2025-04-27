from utils.perspective_transform import Perspective_Transform
import torch
import os
import cv2
import numpy as np
import sys
import json
import argparse


def transform_matrix(M, point, src_size, dst_size):
    """
    Transform a point using homography matrix and scale to target size
    """
    h, w = src_size
    dst_h, dst_w = dst_size
    
    # Apply homography to the point
    point_array = np.array([point[0], point[1], 1])
    warped_point = np.dot(M, point_array)
    warped_point = warped_point[:2] / warped_point[2]
    
    # Scale to target size (115x74 is the standard pitch dimensions)
    x_scaled = int(warped_point[0] * dst_w / 115)
    y_scaled = int(warped_point[1] * dst_h / 74)
    
    return (x_scaled, y_scaled)


def transform_image(M, image, src_size, dst_size):
    """
    Transform an image using homography matrix and scale to target size
    """
    h, w = src_size
    dst_h, dst_w = dst_size

    # Resize input image to match the dimensions used in homography calculation
    resized_image = cv2.resize(image, (1280, 720))
    
    # Create output image of desired size
    warped = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    
    # For each pixel in the output image
    for y_out in range(dst_h):
        for x_out in range(dst_w):
            # Convert output coordinates to pitch coordinates (115x74)
            x_pitch = x_out * 115 / dst_w
            y_pitch = y_out * 74 / dst_h
            
            # Apply inverse homography to get input image coordinates
            point = np.array([x_pitch, y_pitch, 1])
            inv_warped = np.dot(np.linalg.inv(M), point)
            inv_warped = inv_warped[:2] / inv_warped[2]
            
            # Scale back to input image coordinates
            x_in = int(inv_warped[0])
            y_in = int(inv_warped[1])
            
            # Copy pixel if within bounds
            if 0 <= x_in < 1280 and 0 <= y_in < 720:
                warped[y_out, x_out] = resized_image[y_in, x_in]
                
    return warped

def add_detections_to_frame(frame, center_x, center_y):
    cv2.circle(frame, (int(center_x), int(center_y)), 2, (0, 0, 255), -1)


def main(input_path, player_detections_path):
    # Load models
    perspective_transform = Perspective_Transform()

    # Video capture
    cap = cv2.VideoCapture(input_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_num = 0
    
    # Store homography matrices
    homography_data = {}
    
    # Create output directories in intermediate_results folder
    base_dir = 'intermediate_results/' + os.path.basename(input_path).split('.')[0]
    os.makedirs(base_dir, exist_ok=True)
    
    output_path = os.path.join(base_dir, 'homography')
    warped_images_path = os.path.join(base_dir, 'warped_images')
    minimap_dir = os.path.join(base_dir, 'minimap')
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(warped_images_path, exist_ok=True)
    os.makedirs(minimap_dir, exist_ok=True)

    with open(player_detections_path, 'r') as f:
        player_detections = json.load(f)
    
    # Load pitch.jpg as background without resizing
    gt_img = cv2.imread('pitch.jpg')
    gt_h, gt_w, _ = gt_img.shape
    
    # Calculate circle size based on image dimensions
    circle_radius = max(2, int(gt_w / 115))
    
    # Store the last calculated homography matrix
    last_M = None

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            main_frame = frame.copy()
            # Calculate homography matrix every 2 frames
            if frame_num % 2 == 0:
                M, warped_image = perspective_transform.homography_matrix(main_frame)
                last_M = M
                # Store the homography matrix for this frame
                homography_data[str(frame_num)] = M.tolist()
                # Save warped image
                image_filename = f"frame_{frame_num:06d}.jpg"
                warped_image_path = os.path.join(warped_images_path, image_filename)
                cv2.imwrite(warped_image_path, warped_image)

                # transformed_image = transform_image(M, main_frame, (h, w), (gt_h, gt_w))
                # cv2.imwrite(os.path.join(warped_images_path, f"frame_{frame_num:06d}_transformed.jpg"), transformed_image)
            else:
                # Use the last calculated homography matrix
                M = last_M if last_M is not None else perspective_transform.homography_matrix(main_frame)[0]
            
            # Get player detections for this frame
            player_detections_frame = next((det for det in player_detections if det['frame_id'] == frame_num), None)
            
            if player_detections_frame:
                # Create minimap with pitch.jpg as background
                bg_img = gt_img.copy()
                
                for detection in player_detections_frame['detections']:
                    x1, y1, x2, y2 = detection['bbox']
                    # Use bottom center of bounding box for better positioning
                    center_x = x1 + (x2 - x1)/2
                    center_y = y1 + (y2 - y1)
                    
                    # Transform coordinates to minimap
                    coords = transform_matrix(M, (center_x, center_y), (h, w), (gt_h, gt_w))
                    
                    # Only draw if coordinates are within bounds
                    if 0 <= coords[0] < gt_w and 0 <= coords[1] < gt_h:
                        # Draw player as circle, color based on team
                        color = (0, 255, 255) if detection.get('team') == 'Referee' else (0, 0, 0)
                        # Use calculated circle radius
                        cv2.circle(bg_img, coords, circle_radius, color, -1)
                
                # Save minimap
                cv2.imwrite(os.path.join(minimap_dir, f'frame_{frame_num:06d}.jpg'), bg_img)
                
                sys.stdout.write(
                    "\r[Input Video : %s] [%d/%d Frames Processed] [Homography calculated] [Minimap saved]"
                    % (
                        input_path,
                        frame_num,
                        frame_count,
                    )
                )
            else:
                sys.stdout.write(
                    "\r[Input Video : %s] [%d/%d Frames Processed]"
                    % (
                        input_path,
                        frame_num,
                        frame_count,
                    )
                )
            frame_num += 1
        else:
            break

    # Get output filename for homography data
    output_name = os.path.basename(input_path).split('.')[0] + '_homography.json'
    output_file = os.path.join(output_path, output_name)
    
    # Save homography matrices to file
    with open(output_file, 'w') as f:
        json.dump(homography_data, f)
    
    print(f'\n\nHomography matrices have been saved to {output_file}!')
    print(f'Warped images have been saved to {warped_images_path}/')
    print(f'Minimap images have been saved to {minimap_dir}/')
    
    cap.release()
    cv2.destroyAllWindows()
    
    return frame_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create minimap from video file')
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('--player_detections', help='Path to player detections file', required=True)
    args = parser.parse_args()
    
    try:
        with torch.no_grad():
            frame_count = main(args.input, args.player_detections)
        print(f"Processed {frame_count} frames successfully.")
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)
    
    sys.exit(0)

