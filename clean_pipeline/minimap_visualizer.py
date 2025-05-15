from typing import List
import os
import cv2
import numpy as np
from custom_types import FrameDetections

class MinimapVisualizer:
    def __init__(self):
        self.output_dir = 'minimap_visualizer_results'
        # Define colors for different roles (BGR format for OpenCV)
        self.colors = {
            "TEAM A": (0, 0, 255),  # Red
            "TEAM B": (255, 0, 0),  # Blue
            "REF": (0, 255, 255),   # Yellow
            "GK": (0, 0, 0),        # Black
            "REF/GK": (128, 128, 128), # Gray (should not happen)
            "UNK": (128, 128, 128),   # Gray for unknown
            "OOB": (128, 128, 128)   # Gray for out of bounds (should not happen)
        }

    def visualize(self, input_path: str, detections: List[FrameDetections], save_frames: bool = False):
        """
        Create a minimap visualization video showing players with different colors based on roles
        
        Args:
            input_path: Path to the input video file
            detections: List of detection objects for each frame
            save_frames: Whether to save individual frames as images (default: False)
        """
        # Setup output directories
        base_dir = os.path.join(self.output_dir, os.path.basename(input_path).split('.')[0])
        os.makedirs(base_dir, exist_ok=True)
        
        # Create frames directory if saving individual frames
        frames_dir = None
        if save_frames:
            frames_dir = os.path.join(base_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
        
        # Load pitch.jpg as background
        pitch_img = cv2.imread('pitch3.png')
        if pitch_img is None:
            raise FileNotFoundError("Could not find pitch3.png, which is required for minimap visualization")
            
        gt_h, gt_w, _ = pitch_img.shape
        circle_radius = max(2, int(gt_w / 115))
        
        # Get video properties
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Create video writer
        output_dir = os.path.join(base_dir, os.path.basename(input_path).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, 'minimap.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (gt_w, gt_h))
        
        # Process each frame
        for frame_idx in range(frame_count):
            # Create fresh background for each frame
            bg_img = pitch_img.copy()
            
            # Find detection for current frame
            for frame_detections in detections:
                if frame_detections.frame_id == frame_idx:
                    for detection in frame_detections.detections:
                        # Skip if no minimap coordinates or not a person
                        if detection.minimap_coordinates is None or detection.class_name != "person":
                            continue
                            
                        # Get minimap coordinates
                        x = detection.minimap_coordinates.x
                        y = detection.minimap_coordinates.y
                        
                        # Check if coordinates are within bounds
                        if 0 <= x < gt_w and 0 <= y < gt_h:
                            # Determine color based on role
                            role = detection.role if detection.role is not None else "UNK"
                            
                            # Determine color based on role
                            if role == "GK":
                                color = self.colors["GK"]
                            elif role == "REF" or role == "REF/GK":
                                color = self.colors["REF"]
                            else:
                                color = self.colors.get(role, self.colors["UNK"])
                                
                            # Draw player as circle
                            cv2.circle(bg_img, (x, y), circle_radius, color, -1)
                    break
            
            # Write frame to video
            out.write(bg_img)
            
            # Save individual frame if requested
            if save_frames and frames_dir:
                frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(frame_path, bg_img)
            
            # Display progress
            if frame_idx % 10 == 0:
                print(f"\rCreating minimap video: {frame_idx}/{frame_count} frames processed", end="")
                
        out.release()
        print(f"\nMinimap video saved to {output_video_path}")
        
        return output_video_path
        
    def create_combined_video(self, input_path: str, minimap_path: str):
        """
        Create a combined video with original video on top and minimap video below, 
        both of equal size and synchronized
        
        Args:
            input_path: Path to the original input video
            minimap_path: Path to the minimap video
        
        Returns:
            Path to the combined video
        """
        # Setup output
        base_dir = os.path.join(self.output_dir, os.path.basename(input_path).split('.')[0])
        os.makedirs(base_dir, exist_ok=True)
        
        output_dir = os.path.join(base_dir, os.path.basename(input_path).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, 'combined_view.mp4')
        
        # Open both videos
        cap_original = cv2.VideoCapture(input_path)
        cap_minimap = cv2.VideoCapture(minimap_path)
        
        if not cap_original.isOpened() or not cap_minimap.isOpened():
            raise ValueError("Error opening video files")
        
        # Get properties
        fps = cap_original.get(cv2.CAP_PROP_FPS)
        frame_count = min(int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT)), 
                          int(cap_minimap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        # Read first frames to determine size
        ret1, frame_original = cap_original.read()
        ret2, frame_minimap = cap_minimap.read()
        
        if not ret1 or not ret2:
            raise ValueError("Error reading frames from video files")
        
        # Determine the final size (both videos will have the same width)
        target_width = max(frame_original.shape[1], frame_minimap.shape[1])
        original_height = int(frame_original.shape[0] * target_width / frame_original.shape[1])
        minimap_height = int(frame_minimap.shape[0] * target_width / frame_minimap.shape[1])
        
        # Reset video positions
        cap_original.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_minimap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        combined_height = original_height + minimap_height
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, combined_height))
        
        # Process each frame
        for frame_idx in range(frame_count):
            # Read frames
            ret1, frame_original = cap_original.read()
            ret2, frame_minimap = cap_minimap.read()
            
            if not ret1 or not ret2:
                break
                
            # Resize frames to have the same width
            frame_original_resized = cv2.resize(frame_original, (target_width, original_height))
            frame_minimap_resized = cv2.resize(frame_minimap, (target_width, minimap_height))
            
            # Stack vertically
            combined_frame = np.vstack((frame_original_resized, frame_minimap_resized))
            
            # Write to video
            out.write(combined_frame)
            
            # Display progress
            if frame_idx % 10 == 0:
                print(f"\rCreating combined video: {frame_idx}/{frame_count} frames processed", end="")
        
        # Release resources
        cap_original.release()
        cap_minimap.release()
        out.release()
        
        print(f"\nCombined video saved to {output_video_path}")
        return output_video_path
