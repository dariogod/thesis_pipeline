from utils.perspective_transform import Perspective_Transform
import torch
import os
import cv2
import numpy as np
import sys
import json
import argparse


def main(input_path):
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
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(warped_images_path, exist_ok=True)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            main_frame = frame.copy()
            
            # Calculate the homography matrix every 5 frames
            if frame_num % 5 == 0:
                M, warped_image = perspective_transform.homography_matrix(main_frame)
                # Store the homography matrix for this frame
                homography_data[str(frame_num)] = M.tolist()
                
                # Save warped image
                image_filename = f"frame_{frame_num:06d}.jpg"
                warped_image_path = os.path.join(warped_images_path, image_filename)
                cv2.imwrite(warped_image_path, warped_image)
                
                sys.stdout.write(
                    "\r[Input Video : %s] [%d/%d Frames Processed] [Homography calculated] [Warped image saved]"
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
    
    cap.release()
    cv2.destroyAllWindows()
    
    return frame_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create minimap from video file')
    parser.add_argument('input', help='Path to input video file')
    
    args = parser.parse_args()
    
    try:
        with torch.no_grad():
            frame_count = main(args.input)
        print(f"Processed {frame_count} frames successfully.")
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)
    
    sys.exit(0)

