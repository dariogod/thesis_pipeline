import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional
from copy import deepcopy
import os
import json
import math
from custom_types import FrameDetections, Detection, BBox, DominantColors
from conversions import RGBColor255, LABColor, rgb_to_lab, lab_to_rgb_255

class ColorAssigner:
    def __init__(self, max_roi_size: int = 64, kmeans_n_init: int = 3):
        """        
        Args:
            max_roi_size: Maximum size for ROI dimensions (will be downsampled if larger)
            kmeans_n_init: Number of times KMeans will be run with different initializations
        """
        self.max_roi_size = max_roi_size
        self.kmeans_n_init = kmeans_n_init

    def _get_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame 
    
    def _load_video(self, input_path: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        return cap

    def _crop_roi(self, frame: np.ndarray, bbox: BBox, x_range: Tuple[float, float] = (0.0, 1.0), y_range: Tuple[float, float] = (0.0, 0.5)) -> Tuple[np.ndarray, BBox]:
        x1, y1, x2, y2 = bbox.as_list()

        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return np.array([]), None
        
        height, width = roi.shape[:2]
        y_start = int(height * y_range[0])
        y_end = int(height * y_range[1])
        x_start = int(width * x_range[0])
        x_end = int(width * x_range[1])
        
        if y_end <= y_start or x_end <= x_start:
            return np.array([]), None
        
        # Calculate absolute coordinates for ROI BBox
        roi_x1 = x1 + x_start
        roi_y1 = y1 + y_start
        roi_x2 = x1 + x_end
        roi_y2 = y1 + y_end
        
        # Create ROI BBox object
        roi_bbox = BBox(x1=roi_x1, y1=roi_y1, x2=roi_x2, y2=roi_y2)
        
        # Crop the ROI
        roi = roi[y_start:y_end, x_start:x_end]
        
        # Downsample large ROIs for faster processing
        if roi.size > 0:
            h, w = roi.shape[:2]
            if h > self.max_roi_size or w > self.max_roi_size:
                ratio = min(self.max_roi_size / h, self.max_roi_size / w)
                new_size = (int(w * ratio), int(h * ratio))
                roi = cv2.resize(roi, new_size, interpolation=cv2.INTER_AREA)
        
        return roi, roi_bbox
    
    def _get_dominant_colors(self, roi: np.ndarray) -> DominantColors:
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            return DominantColors(
                background=None,
                jersey=None
            )
            
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Get the distance of each pixel from the bottom center (anchor_point) of the ROI. 
        # Pixels near this point are more likely to be the jersey.
        height, width = roi.shape[:2]
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        anchor_y, anchor_x = height - 1, width / 2
        
        squared_distances = (y_coords - anchor_y)**2 + (x_coords - anchor_x)**2
        
        # Cluster pixels
        pixels = roi_rgb.reshape(-1, 3)
        lab_pixels = rgb_to_lab(pixels)
        
        # Use KMeans with reduced number of initializations
        kmeans = KMeans(n_clusters=2, n_init=self.kmeans_n_init, random_state=42)
        labels = kmeans.fit_predict(lab_pixels)
        centers_lab: list[LABColor] = [LABColor.from_array(center) for center in kmeans.cluster_centers_]
        
        # Convert centers back to RGB
        centers_rgb_255 = []
        for center in centers_lab:
            rgb_color = lab_to_rgb_255(center)
            centers_rgb_255.append([rgb_color.r, rgb_color.g, rgb_color.b])
        centers_rgb_255 = np.array(centers_rgb_255)
        
        # Calculate average squared distance to center for each cluster
        avg_distances = []
        for cluster_idx in range(2):
            cluster_mask = labels == cluster_idx
            if np.sum(cluster_mask) > 0:  # Avoid division by zero
                avg_dist = np.mean(squared_distances.reshape(-1)[cluster_mask])
                avg_distances.append((cluster_idx, avg_dist))
        
        # Sort clusters by average distance to center (ascending)
        avg_distances.sort(key=lambda x: x[1])
        
        # Closest cluster to center is jersey, other is background
        jersey_idx = avg_distances[0][0]
        background_idx = avg_distances[1][0]
        
        # Get RGB values
        jersey_rgb = centers_rgb_255[jersey_idx]
        background_rgb = centers_rgb_255[background_idx]
        
        # Create DominantColors object
        return DominantColors(
            background=RGBColor255.from_array(background_rgb),
            jersey=RGBColor255.from_array(jersey_rgb)
        )

    def process_frame(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        detections_copy = deepcopy(detections)

        for detection in detections_copy:
            roi, roi_bbox = self._crop_roi(frame, detection.bbox)
            dominant_colors = self._get_dominant_colors(roi)
            detection.roi_bbox = roi_bbox
            detection.jersey_color = dominant_colors.jersey

        return detections_copy

    def process_video(self, input_path: str, detections: list[FrameDetections], output_dir: Optional[str] = None, store_results: bool = True):
        cap = self._load_video(input_path)
        
        if store_results:
            if output_dir is None:
                base_name = os.path.basename(input_path).split('.')[0]
                output_dir = os.path.join('color_assigner_results', base_name)
            
            os.makedirs(output_dir, exist_ok=True)
            output_json_path = os.path.join(output_dir, 'player_colors.json')
            
            # Get video properties for output video
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set up font parameters based on frame height
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = height / 1000
            font_thickness = math.floor(font_face)
            text_height = cv2.getTextSize("1234567890", font_face, font_scale, font_thickness)[0][1]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_output_path = os.path.join(output_dir, 'color_visualization.mp4')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        # Process each frame's detections
        for frame_detections in detections:
            frame = self._get_frame(cap, frame_detections.frame_id)
            if frame is None:
                raise ValueError(f"Frame {frame_detections.frame_id} not found in video")
            
            updated_detections = self.process_frame(frame, frame_detections.detections)
            frame_detections.detections = updated_detections
            
            # Save color visualizations if store_results is True
            if store_results:
                self._visualize_and_write_frame(
                    frame, 
                    updated_detections, 
                    video_writer,
                    font_face=font_face,
                    font_scale=font_scale,
                    font_thickness=font_thickness,
                    text_height=text_height
                )

        # Release resources
        cap.release()
        if store_results:
            video_writer.release()

            detections_json = [frame_data.model_dump() for frame_data in detections]
            with open(output_json_path, 'w') as f:
                json.dump(detections_json, f, indent=4)
        
        return detections
    
    def _visualize_and_write_frame(
        self, 
        frame: np.ndarray, 
        detections: list[Detection], 
        video_writer: cv2.VideoWriter,
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        text_height: int = 0
    ):
        """Helper method to visualize detections and write frame to video"""
        frame_copy = frame.copy()
        for detection in detections:
            if detection.jersey_color:
                # Draw jersey color as a rectangle at the top of the bounding box
                x1, y1, x2, y2 = detection.bbox.as_list()
                color_rect_height = min(30, (y2 - y1) // 4)
                
                # Draw jersey color rectangle
                jersey_color = (
                    detection.jersey_color.b,  # BGR format for OpenCV
                    detection.jersey_color.g,
                    detection.jersey_color.r
                )
                cv2.rectangle(
                    frame_copy, 
                    (x1, y2), 
                    (x2, y2 + color_rect_height), 
                    jersey_color, 
                    -1  # Fill the rectangle
                )
                
                # Add track ID if available
                if detection.track_id is not None:
                    cv2.putText(
                        frame_copy,
                        f"ID: {detection.track_id}",
                        (x1, y1 - int(text_height * 0.5)),  # Adjust position based on text height
                        font_face,
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                        cv2.LINE_AA
                    )
        
        # Write frame to video
        video_writer.write(frame_copy)
            
            
    
