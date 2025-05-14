import os
import cv2
import numpy as np
import json
import sys
from utils.perspective_transform import Perspective_Transform
from custom_types import FrameDetections, MinimapCoordinates

class CoordinateTransformer:
    def __init__(self):
        self.perspective_transform = Perspective_Transform()
        self.output_dir = 'coordinate_transformer_results'
    
    def transform_matrix(self, M, point, src_size, dst_size):
        """
        Transform a point using homography matrix and scale to target size
        """
        h, w = src_size
        dst_h, dst_w = dst_size
        
        # Apply homography to the point
        point_array = np.array([point[0] * 1280 / w, point[1] * 720 / h, 1])
        warped_point = np.dot(M, point_array)
        warped_point = warped_point[:2] / warped_point[2]
        
        # Scale to target size (115x74 is the standard pitch dimensions)
        x_scaled = int(warped_point[0] * dst_w / 115)
        y_scaled = int(warped_point[1] * dst_h / 74)
        
        return (x_scaled, y_scaled)
    
    def transform_image(self, M, image, dst_size):
        """
        Transform an image using homography matrix and scale to target size
        """
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
        
    def transform(self, input_path: str, detections: list[FrameDetections], store_results: bool = True) -> list[FrameDetections]:
        """
        Calculate minimap coordinates for all detections and update the detections object
        
        Args:
            input_path: Path to the input video file
            detections: List of detection objects for each frame
            store_results: Whether to store results (images, JSONs) to disk (default: True)
            
        Returns:
            Updated detections with minimap coordinates
        """
        # Setup output directories if storing results
        if store_results:
            base_dir = os.path.join(self.output_dir, os.path.basename(input_path).split('.')[0])
            os.makedirs(base_dir, exist_ok=True)
            
            homography_dir = os.path.join(base_dir, 'homography')
            warped_images_dir = os.path.join(base_dir, 'warped_images')
            minimap_dir = os.path.join(base_dir, 'minimap')
            
            os.makedirs(homography_dir, exist_ok=True)
            os.makedirs(warped_images_dir, exist_ok=True)
            os.makedirs(minimap_dir, exist_ok=True)
        
        # Load pitch.jpg as background
        pitch_img = cv2.imread('pitch3.png')
        gt_h, gt_w, _ = pitch_img.shape
        circle_radius = max(2, int(gt_w / 115))
        
        # Video capture
        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        homography_data = {}
        last_M = None
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate homography matrix every 2 frames to save processing time
            if processed_frames % 2 == 0:
                M, warped_image = self.perspective_transform.homography_matrix(frame)
                last_M = M
                homography_data[str(processed_frames)] = M.tolist()
                
                # Save warped image if storing results
                if store_results:
                #     image_filename = f"frame_{processed_frames:06d}.jpg"
                #     warped_image_path = os.path.join(warped_images_dir, image_filename)
                #     cv2.imwrite(warped_image_path, warped_image)
                    
                #     # Save high-res warped image
                    warped_image_high_res = self.transform_image(M, frame, (540, 960))
                    warped_image_high_res_path = os.path.join(warped_images_dir, f"frame_{processed_frames:06d}_high_res.jpg")
                    cv2.imwrite(warped_image_high_res_path, warped_image_high_res)
            else:
                # Use the last calculated homography matrix
                M = last_M if last_M is not None else self.perspective_transform.homography_matrix(frame)[0]
            
            # Find detection for current frame
            for frame_detections in detections:
                if frame_detections.frame_id == processed_frames:
                    # Create minimap visualization if storing results
                    if store_results:
                        bg_img = pitch_img.copy()

                    for detection in frame_detections.detections:
                        x1, y1, x2, y2 = detection.bbox.as_list()
                        # Use bottom center of bounding box for better positioning
                        center_x = x1 + (x2 - x1)/2
                        center_y = y1 + (y2 - y1)
                        
                        # Transform coordinates to minimap
                        minimap_coords = self.transform_matrix(M, (center_x, center_y), (h, w), (gt_h, gt_w))
                        
                        # Add minimap coordinates to detection
                        detection.minimap_coordinates = MinimapCoordinates(x=minimap_coords[0], y=minimap_coords[1])
                        
                        # Draw on visualization if storing results
                        if store_results and 0 <= minimap_coords[0] < gt_w and 0 <= minimap_coords[1] < gt_h:
                            # Draw player as circle, color based on team
                            color = (0, 0, 0)
                            cv2.circle(bg_img, minimap_coords, circle_radius, color, -1)
                    
                    # Save minimap if storing results
                    if store_results:
                        cv2.imwrite(os.path.join(minimap_dir, f'frame_{processed_frames:06d}.jpg'), bg_img)
                    break
            
            sys.stdout.write(
                "\r[Input Video: %s] [%d/%d Frames Processed]"
                % (
                    input_path,
                    processed_frames,
                    frame_count,
                )
            )
            sys.stdout.flush()
            
            processed_frames += 1
        
        cap.release()
        
        # Save files if storing results
        if store_results:
            # Save homography matrices
            output_file = os.path.join(homography_dir, "homography.json")
            with open(output_file, 'w') as f:
                json.dump(homography_data, f)
            
            # Save updated detections
            detections_file = os.path.join(base_dir, "detections_with_minimap.json")
            with open(detections_file, 'w') as f:
                raw_detections = [frame_detections.model_dump() for frame_detections in detections]
                json.dump(raw_detections, f, indent=4)
            
            print(f"\n\nProcessed {processed_frames} frames.")
            print(f"Updated detections saved to {detections_file}")
            print(f"Homography matrices saved to {output_file}")
            print(f"Warped images saved to {warped_images_dir}/")
            print(f"Minimap visualizations saved to {minimap_dir}/")
        else:
            print(f"\n\nProcessed {processed_frames} frames.")
            print("Results not saved to disk (store_results=False)")
        
        return detections