import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from skimage import color
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


class RoleAssigner:
    def __init__(self):
        self.cluster_centers = {}

    def _load_video(self, video_path: str) -> cv2.VideoCapture:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return cv2.VideoCapture(video_path)
    
    def _get_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame 

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> np.ndarray:
        bgr = np.uint8([[list(reversed(rgb))]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab[0, 0].astype(np.float32)  # Convert to float32 for calculations

    def _lab_to_rgb(self, lab: np.ndarray) -> Tuple[int, int, int]:
        # Ensure lab is in proper format for conversion
        lab_reshaped = np.uint8([[lab]])
        bgr = cv2.cvtColor(lab_reshaped, cv2.COLOR_LAB2BGR)
        # Convert from BGR to RGB and return as tuple
        rgb = tuple(reversed(bgr[0, 0].tolist()))
        return rgb  # Returns (R, G, B) tuple with values 0-255

    def _extract_roi(self, frame: np.ndarray, bbox: List[int], y_range: Tuple[float, float], x_range: Tuple[float, float]) -> np.ndarray:
        """
        Returns:
            Cropped region
        """
        x1, y1, x2, y2 = bbox
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return np.array([])
        
        height, width = roi.shape[:2]
        y_start = int(height * y_range[0])
        y_end = int(height * y_range[1])
        x_start = int(width * x_range[0])
        x_end = int(width * x_range[1])
        
        if y_end <= y_start or x_end <= x_start:
            return np.array([])
        
        return roi[y_start:y_end, x_start:x_end]
    
    def _cluster_colors(self, pixels: np.ndarray) -> Tuple:
        rgb_normalized = pixels.astype(np.float32) / 255.0
        lab_pixels = color.rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(lab_pixels)
        
        centers_lab = kmeans.cluster_centers_
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Convert centers to RGB
        centers_lab_reshaped = centers_lab.reshape(1, -1, 3)
        centers_rgb = color.lab2rgb(centers_lab_reshaped).reshape(-1, 3)
        centers_rgb_255 = (centers_rgb * 255).astype(int)
        
        return centers_lab, centers_rgb, centers_rgb_255, labels, counts
    
    def _identify_background_jersey(self, centers_rgb_255: np.ndarray) -> Tuple[int, int]:
        # Define a reference dark green color for football field (in RGB)
        reference_green_rgb = (0, 100, 0)  # Dark green
        reference_green_lab = self._rgb_to_lab(reference_green_rgb)
        
        # Convert RGB to LAB for better perceptual color comparison
        centers_lab = np.array([self._rgb_to_lab(tuple(map(int, rgb))) for rgb in centers_rgb_255])
        
        # Calculate distance to reference green in LAB space
        distances = []
        for lab in centers_lab:
            # Euclidean distance in LAB space
            distance = np.sqrt(
                (lab[0] - reference_green_lab[0])**2 +  # L* difference
                (lab[1] - reference_green_lab[1])**2 +  # a* difference
                (lab[2] - reference_green_lab[2])**2    # b* difference
            )
            distances.append(distance)
        
        # Get index of the cluster closest to reference green (background)
        background_idx = np.argmin(distances)
        jersey_idx = 1 - background_idx  # The other cluster is the jersey
        
        return background_idx, jersey_idx

    def get_dominant_color(
            self, 
            frame: np.ndarray, 
            bbox: List[int], 
            output_dir: Optional[str] = None, 
            frame_id: Optional[int] = None, 
            track_id: Optional[int] = None, 
            y_range: Tuple[float, float] = (0.0, 0.5), 
            x_range: Tuple[float, float] = (0.0, 1.0),
            visualize: bool = False
        ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        # Extract region of interest
        x1, y1, x2, y2 = bbox
        full_cropped = self._extract_roi(frame, bbox, (0.0, 1.0), (0.0, 1.0))
        cropped = self._extract_roi(frame, bbox, y_range, x_range)
        
        if cropped.size == 0:
            return (0, 0, 0), (0, 0, 0)
        
        # Convert to RGB and reshape for clustering
        full_cropped_rgb = cv2.cvtColor(full_cropped, cv2.COLOR_BGR2RGB)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pixels = cropped_rgb.reshape(-1, 3)
        
        if len(pixels) < 2:
            return (0, 0, 0), (0, 0, 0)
        
        try:
            # Cluster colors
            centers_lab, centers_rgb, centers_rgb_255, labels, counts = self._cluster_colors(pixels)
            
            # Identify background and jersey clusters
            background_idx, jersey_idx = self._identify_background_jersey(centers_rgb_255)
            
            # Reorder clusters so background is first, jersey is second
            cluster_order = [background_idx, jersey_idx]
            sorted_centers_rgb = centers_rgb[cluster_order]
            sorted_centers_rgb_255 = centers_rgb_255[cluster_order]
            sorted_centers_lab = centers_lab[cluster_order]
            
            total_pixels = np.sum(counts)
            percentages = counts / total_pixels
            sorted_percentages = percentages[cluster_order]
            
            # Optionally visualize the results
            if visualize and output_dir and frame_id is not None:
                try:
                    self._visualize_color_clusters(
                        frame, bbox, full_cropped_rgb, 
                        output_dir, frame_id, track_id,
                        x_range, y_range,
                        labels, sorted_centers_rgb, sorted_centers_rgb_255,
                        cluster_order, percentages, centers_lab
                    )
                except Exception as e:
                    print(f"Visualization error (continuing): {e}")
            
            # Return the background and jersey colors
            background_color = tuple(int(v) for v in sorted_centers_rgb_255[0])
            jersey_color = tuple(int(v) for v in sorted_centers_rgb_255[1])
            return background_color, jersey_color
        
        except Exception as e:
            msg = f"Error in color clustering: {e}"
            print(msg)
            raise Exception(msg)
    
    def _visualize_color_clusters(
            self,
            frame: np.ndarray,
            bbox: List[int],
            roi_rgb: np.ndarray,
            output_dir: str,
            frame_id: int,
            track_id: Optional[int],
            x_range: Tuple[float, float],
            y_range: Tuple[float, float],
            labels: np.ndarray,
            sorted_centers_rgb: np.ndarray,
            sorted_centers_rgb_255: np.ndarray,
            cluster_order: List[int],
            percentages: np.ndarray,
            centers_lab: np.ndarray
        ) -> None:
        vis_dir = os.path.join(output_dir, 'player_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        track_str = f"track_{track_id}" if track_id is not None else "unknown_track"
        filename = f"{frame_id:04d}_{track_str}.png"
        
        # Convert full frame to RGB for visualization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        x1, y1, x2, y2 = bbox
        height, width = roi_rgb.shape[:2]
        y_start = int(height * y_range[0])
        y_end = int(height * y_range[1])
        x_start = int(width * x_range[0])
        x_end = int(width * x_range[1])
        
        # Get LAB pixel values for visualization
        cropped_rgb = roi_rgb[y_start:y_end, x_start:x_end]
        pixels = cropped_rgb.reshape(-1, 3)
        rgb_normalized = pixels.astype(np.float32) / 255.0
        lab_pixels = color.rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Create visualization
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 1])
        
        # Display full frame in top left
        ax_frame = plt.subplot(gs[0, 0])
        ax_frame.imshow(frame_rgb)
        rect = plt.Rectangle((x1-0.5, y1-0.5), x2-x1, y2-y1, 
                           fill=False, edgecolor='yellow', linewidth=2)
        ax_frame.add_patch(rect)
        ax_frame.set_title(f'Frame {frame_id} - Full View')
        ax_frame.axis('off')
        
        # Display player ROI
        ax0 = plt.subplot(gs[1, 0])
        ax0.imshow(roi_rgb)
        rect = plt.Rectangle((x_start-0.5, y_start-0.5), x_end - x_start, y_end - y_start, 
                          fill=False, edgecolor='red', linewidth=2)
        ax0.add_patch(rect)
        ax0.set_title('Player Region\n(Red rectangle shows analyzed area)')
        ax0.axis('off')
        
        # Display color clusters
        ax1 = fig.add_subplot(gs[0:2, 1], projection='3d')
        plot_colors = ['green', 'blue']
        cluster_names = ['Background', 'Jersey']
        
        max_points = 1000
        sample_indices = np.random.choice(len(lab_pixels), min(max_points, len(lab_pixels)), replace=False)
        
        for i, cluster_idx in enumerate(cluster_order):
            mask = labels == cluster_idx
            mask_sampled = np.zeros_like(mask)
            mask_sampled[sample_indices] = mask[sample_indices]
            
            ax1.scatter(
                lab_pixels[mask_sampled, 1],
                lab_pixels[mask_sampled, 2],
                lab_pixels[mask_sampled, 0],
                c=plot_colors[i],
                marker='o',
                s=30,
                label=f'{cluster_names[i]} ({percentages[cluster_idx]:.1%})'
            )
        
        ax1.view_init(elev=20, azim=-60)
        
        ax1.set_xlabel('a* (Green-Red)')
        ax1.set_ylabel('b* (Blue-Yellow)')
        ax1.set_zlabel('L* (Lightness)')
        ax1.set_title('LAB Color Space Clusters')
        ax1.legend()
        
        for i, cluster_idx in enumerate(cluster_order):
            ax = fig.add_subplot(gs[2, i])
            rgb_color = sorted_centers_rgb[i]
            rgb_color_255 = sorted_centers_rgb_255[i]
            
            solid_color = np.full((50, 50, 3), rgb_color)
            ax.imshow(solid_color)
            ax.set_title(f"{cluster_names[i]}\nRGB: {rgb_color_255}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, filename), dpi=150)
        plt.close(fig) 

    def _calculate_detections_metadata(
            self, 
            frame: np.ndarray, 
            bbox: List[int], 
            all_background_colors_rgb: List[Tuple[int, int, int]], 
            background_color_rgb: Tuple[int, int, int], 
            jersey_color_rgb: Tuple[int, int, int], 
            detections: List[Dict]
        ) -> Dict:
        x1, y1, x2, y2 = bbox
        
        # Calculate bounding box size (normalized by frame size)
        frame_height, frame_width = frame.shape[:2]
        bbox_width, bbox_height = x2 - x1, y2 - y1
        bbox_size = (bbox_width * bbox_height) / (frame_width * frame_height)
        
        # Check for bbox overlap with other detections
        bbox_has_overlap = self._check_bbox_overlap(bbox, detections)
        
        # Determine if background color is an outlier using DBSCAN
        background_is_outlier = self._is_background_outlier(
            all_background_colors_rgb, background_color_rgb
        )
        
        return {
            "bbox_has_overlap": bbox_has_overlap,
            "bbox_size": bbox_size,
            "background_is_outlier": background_is_outlier
        }
    
    def _check_bbox_overlap(self, bbox: List[int], detections: List[Dict]) -> bool:
        x1, y1, x2, y2 = bbox
        
        for other_detection in detections:
            if "bbox" not in other_detection or other_detection.get("class") != "person":
                continue
                
            other_bbox = other_detection["bbox"]
            other_x1, other_y1, other_x2, other_y2 = other_bbox
            
            # Skip comparing with self
            if other_x1 == x1 and other_y1 == y1 and other_x2 == x2 and other_y2 == y2:
                continue
                
            # Check for overlap
            if not (other_x2 < x1 or other_x1 > x2 or other_y2 < y1 or other_y1 > y2):
                return True
        
        return False
    
    def _is_background_outlier(
            self, 
            all_background_colors_rgb: List[Tuple[int, int, int]], 
            background_color_rgb: Tuple[int, int, int]
        ) -> bool:
        if len(all_background_colors_rgb) < 3:  # Need at least 3 backgrounds to use DBSCAN effectively
            return False
        
        # Convert all background colors to LAB for better perceptual comparison
        background_labs = np.array([self._rgb_to_lab(bg) for bg in all_background_colors_rgb])
        
        # Find the index of the current background color
        current_bg_idx = -1
        for i, bg in enumerate(all_background_colors_rgb):
            if bg == background_color_rgb:
                current_bg_idx = i
                break
        
        if current_bg_idx == -1:
            return False
            
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=20.0, min_samples=2)
        clusters = dbscan.fit_predict(background_labs)
        
        # Points labeled as -1 are considered outliers by DBSCAN
        return bool(clusters[current_bg_idx] == -1)
    
    def _setup_output_paths(self, video_path: str, output_dir: Optional[str]) -> Tuple[str, str, str]:
        if output_dir is None:
            base_name = os.path.basename(video_path).split('.')[0]
            output_dir = os.path.join('role_assignment_results', base_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_json_path = os.path.join(output_dir, 'role_assignments.json')
        output_color_assignments_video_path = os.path.join(output_dir, 'color_assignments_video.mp4')
        output_role_assignments_video_path = os.path.join(output_dir, 'role_assignments_video.mp4')
        
        return output_dir, output_json_path, output_color_assignments_video_path, output_role_assignments_video_path
    
    def _setup_video_writer(self, cap: cv2.VideoCapture, output_video_path: str) -> Tuple[cv2.VideoWriter, int, int, float, int]:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        return out, width, height, fps, total_frames 

    def _process_frame_detections(
            self,
            frame: np.ndarray, 
            frame_data: Dict, 
            output_dir: Optional[str],
            store_results: bool,
            visualize_colors: bool,
            viz_frame: Optional[np.ndarray] = None,
            total_frames: Optional[int] = None
        ) -> None:

        frame_idx = frame_data['frame_id']
        
        if store_results and viz_frame is not None:
            # Add frame counter to top left
            frame_text = f"{frame_idx}/{total_frames}"
            cv2.putText(viz_frame, 
                      frame_text, 
                      (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, 
                      (0, 0, 0), 
                      1, 
                      cv2.LINE_AA)
        
        # Step 1: Extract ROIs for all valid detections
        extracted_rois = self._extract_all_rois(frame, frame_data)
        
        # Step 2: Find dominant colors for all ROIs
        dominant_colors = self._get_dominant_colors_for_rois(frame, extracted_rois, output_dir, frame_idx, visualize_colors)
        
        # Step 3: Calculate metadata including ROI overlaps
        detection_metadata = self._calculate_all_detection_metadata(frame, frame_data, extracted_rois, dominant_colors)
        
        # Step 4: Add color and metadata to detections
        self._add_color_metadata_to_detections(frame, frame_data, dominant_colors, detection_metadata, store_results, viz_frame)
    
    def _extract_all_rois(self, frame: np.ndarray, frame_data: Dict) -> Dict[int, Dict]:
        """Extract ROIs for all valid detections in a frame.
        
        Args:
            frame: Video frame
            frame_data: Frame data including detections
            
        Returns:
            Dictionary mapping track_ids to ROI data including bbox, full ROI, and cropped ROI
        """
        extracted_rois = {}
        
        # Process each detection in the frame
        for detection in frame_data['detections']:
            if "bbox" not in detection or detection.get("class") != "person":
                continue
                
            bbox = detection["bbox"]
            track_id = detection.get("track_id")
            
            if track_id is None:
                continue
            
            # Extract both full ROI and cropped upper body ROI
            x1, y1, x2, y2 = bbox
            full_roi = self._extract_roi(frame, bbox, (0.0, 1.0), (0.0, 1.0))
            
            if full_roi.size == 0:
                continue
                
            # Extract upper body ROI (for jersey detection)
            upper_body_roi = self._extract_roi(frame, bbox, (0.0, 0.5), (0.0, 1.0))
            
            if upper_body_roi.size == 0:
                continue
            
            # Convert to RGB
            full_roi_rgb = cv2.cvtColor(full_roi, cv2.COLOR_BGR2RGB)
            upper_roi_rgb = cv2.cvtColor(upper_body_roi, cv2.COLOR_BGR2RGB)
            
            # Store in dictionary
            extracted_rois[track_id] = {
                "bbox": bbox,
                "full_roi": full_roi,
                "full_roi_rgb": full_roi_rgb,
                "upper_roi": upper_body_roi,
                "upper_roi_rgb": upper_roi_rgb,
                "y_range": (0.0, 0.5),  # The range used for upper body
                "x_range": (0.0, 1.0)   # The range used for width
            }
        
        return extracted_rois
    
    def _get_dominant_colors_for_rois(
            self, 
            frame: np.ndarray, 
            extracted_rois: Dict[int, Dict], 
            output_dir: Optional[str], 
            frame_idx: int, 
            visualize_colors: bool
        ) -> Dict[int, Dict]:
        """Find dominant colors for all extracted ROIs.
        
        Args:
            frame: Video frame
            extracted_rois: Dictionary of extracted ROIs
            output_dir: Output directory for visualizations
            frame_idx: Frame index
            visualize_colors: Whether to create color visualizations
            
        Returns:
            Dictionary mapping track_ids to dominant colors
        """
        dominant_colors = {}
        
        for track_id, roi_data in extracted_rois.items():
            upper_roi_rgb = roi_data["upper_roi_rgb"]
            bbox = roi_data["bbox"]
            
            pixels = upper_roi_rgb.reshape(-1, 3)
            
            if len(pixels) < 2:
                continue
                
            try:
                # Cluster colors
                centers_lab, centers_rgb, centers_rgb_255, labels, counts = self._cluster_colors(pixels)
                
                # Identify background and jersey clusters
                background_idx, jersey_idx = self._identify_background_jersey(centers_rgb_255)
                
                # Reorder clusters so background is first, jersey is second
                cluster_order = [background_idx, jersey_idx]
                sorted_centers_rgb = centers_rgb[cluster_order]
                sorted_centers_rgb_255 = centers_rgb_255[cluster_order]
                sorted_centers_lab = centers_lab[cluster_order]
                
                total_pixels = np.sum(counts)
                percentages = counts / total_pixels
                sorted_percentages = percentages[cluster_order]
                
                # Optionally visualize the results
                if visualize_colors and output_dir and frame_idx is not None:
                    try:
                        self._visualize_color_clusters(
                            frame, bbox, roi_data["full_roi_rgb"], 
                            output_dir, frame_idx, track_id,
                            roi_data["x_range"], roi_data["y_range"],
                            labels, sorted_centers_rgb, sorted_centers_rgb_255,
                            cluster_order, percentages, centers_lab
                        )
                    except Exception as e:
                        print(f"Visualization error (continuing): {e}")
                
                # Store the dominant colors
                background_color = tuple(int(v) for v in sorted_centers_rgb_255[0])
                jersey_color = tuple(int(v) for v in sorted_centers_rgb_255[1])
                
                dominant_colors[track_id] = {
                    "background_color": background_color,
                    "jersey_color": jersey_color,
                    "percentages": sorted_percentages
                }
                
            except Exception as e:
                print(f"Error in color clustering for track {track_id}: {e}")
                continue
        
        return dominant_colors
    
    def _calculate_all_detection_metadata(
            self, 
            frame: np.ndarray, 
            frame_data: Dict,
            extracted_rois: Dict[int, Dict],
            dominant_colors: Dict[int, Dict]
        ) -> Dict[int, Dict]:
        """Calculate metadata for all detections including ROI overlap.
        
        Args:
            frame: Video frame
            frame_data: Frame data including detections
            extracted_rois: Dictionary of extracted ROIs
            dominant_colors: Dictionary of dominant colors
            
        Returns:
            Dictionary mapping track_ids to metadata
        """
        metadata_by_track = {}
        frame_height, frame_width = frame.shape[:2]
        
        # Get all background colors
        all_background_colors_rgb = [
            colors["background_color"] for colors in dominant_colors.values()
        ]
        
        # Calculate metadata for each detection
        for track_id, roi_data in extracted_rois.items():
            if track_id not in dominant_colors:
                continue
                
            bbox = roi_data["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Calculate bounding box size (normalized by frame size)
            bbox_width, bbox_height = x2 - x1, y2 - y1
            bbox_size = (bbox_width * bbox_height) / (frame_width * frame_height)
            
            # Check for ROI overlap with other detections (instead of bbox overlap)
            roi_has_overlap = self._check_roi_overlap(track_id, extracted_rois)
            
            # Determine if background color is an outlier
            background_color_rgb = dominant_colors[track_id]["background_color"]
            background_is_outlier = self._is_background_outlier(
                all_background_colors_rgb, background_color_rgb
            )
            
            # Store metadata
            metadata_by_track[track_id] = {
                "roi_has_overlap": roi_has_overlap,
                "bbox_size": bbox_size,
                "background_is_outlier": background_is_outlier
            }
        
        return metadata_by_track
    
    def _check_roi_overlap(self, track_id: int, extracted_rois: Dict[int, Dict]) -> bool:
        """Check if a ROI overlaps with any other ROIs.
        
        Args:
            track_id: Track ID to check
            extracted_rois: Dictionary of extracted ROIs
            
        Returns:
            True if the ROI overlaps with any other ROI
        """
        if track_id not in extracted_rois:
            return False
            
        current_bbox = extracted_rois[track_id]["bbox"]
        x1, y1, x2, y2 = current_bbox
        
        for other_id, other_roi_data in extracted_rois.items():
            # Skip comparing with self
            if other_id == track_id:
                continue
                
            other_bbox = other_roi_data["bbox"]
            other_x1, other_y1, other_x2, other_y2 = other_bbox
            
            # Check for overlap between ROIs
            # ROIs overlap if their bounding boxes overlap
            if not (other_x2 < x1 or other_x1 > x2 or other_y2 < y1 or other_y1 > y2):
                # Calculate overlap area
                overlap_width = min(x2, other_x2) - max(x1, other_x1)
                overlap_height = min(y2, other_y2) - max(y1, other_y1)
                overlap_area = overlap_width * overlap_height
                
                # Calculate minimum ROI area
                current_area = (x2 - x1) * (y2 - y1)
                other_area = (other_x2 - other_x1) * (other_y2 - other_y1)
                min_area = min(current_area, other_area)
                
                # If overlap is significant (more than 20% of the smaller ROI)
                if overlap_area > 0.2 * min_area:
                    return True
        
        return False
    
    def _add_color_metadata_to_detections(
            self,
            frame: np.ndarray,
            frame_data: Dict,
            dominant_colors: Dict[int, Dict],
            detection_metadata: Dict[int, Dict],
            store_results: bool,
            viz_frame: Optional[np.ndarray] = None
        ) -> None:
        """Add color info and metadata to each detection.
        
        Args:
            frame: Video frame
            frame_data: Frame data including detections
            dominant_colors: Dictionary of dominant colors by track_id
            detection_metadata: Dictionary of metadata by track_id
            store_results: Whether to store/visualize results
            viz_frame: Frame to draw visualization on
        """
        for detection in frame_data['detections']:
            if "bbox" not in detection or detection.get("class") != "person":
                continue
            
            track_id = detection.get("track_id")
            
            if track_id is None or track_id not in dominant_colors or track_id not in detection_metadata:
                continue

            background_color_rgb = dominant_colors[track_id]["background_color"]
            jersey_color_rgb = dominant_colors[track_id]["jersey_color"]

            if self.use_predefined:
                closest_color = self.find_closest_predefined_color(jersey_color_rgb)
                final_color = closest_color
            else:
                closest_color = None
                final_color = jersey_color_rgb

            color_info = {
                "raw_background_rgb": background_color_rgb,
                "raw_jersey_rgb": jersey_color_rgb,
                "closest_jersey_color": closest_color,
            }

            # Get metadata
            metadata = detection_metadata[track_id]
            
            # If background is an outlier, set final_color to None
            if metadata["background_is_outlier"]: 
                final_color = None
            color_info["final_color"] = final_color
            
            # Add to detection
            detection["color_info"] = color_info
            detection["metadata"] = metadata

            if store_results and viz_frame is not None:
                self._draw_enhanced_detection_visualization(viz_frame, detection["bbox"], track_id, final_color)
    
    def _draw_enhanced_detection_visualization(
            self,
            viz_frame: np.ndarray,
            bbox: List[int],
            track_id: int,
            final_color: Optional[Union[str, Tuple[int, int, int]]]
        ) -> None:
        """Draw enhanced visualization with black bbox and semi-transparent ROI.
        
        Args:
            viz_frame: Frame to draw on
            bbox: Bounding box coordinates
            track_id: Track ID to label the box
            final_color: Final assigned color (either color name or RGB tuple)
        """
        x1, y1, x2, y2 = bbox
        
        # 1. Draw black bounding box for all players
        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # 2. Draw a colored border around the upper body ROI (jersey region)
        if final_color is not None:
            # Convert string color names to RGB values if necessary
            if isinstance(final_color, str) and final_color in self.predefined_colors:
                color_value = self.predefined_colors[final_color]
            elif isinstance(final_color, (tuple, list)) and len(final_color) == 3:
                color_value = final_color
            else:
                # Skip if color format is unexpected
                color_value = None
                
            if color_value is not None:
                # Define upper body ROI (top half of bounding box)
                upper_roi_y2 = y1 + (y2 - y1) // 2
                
                # Draw a thicker colored border around the ROI (3px vs 2px for black bbox)
                cv2.rectangle(
                    viz_frame, 
                    (x1, y1), 
                    (x2, upper_roi_y2), 
                    (color_value[2], color_value[1], color_value[0]),  # BGR format for OpenCV
                    3  # Thicker border
                )
        
        # 3. Add track ID label
        if track_id is not None:
            label = f"{track_id}"
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_PLAIN
            
            cv2.putText(viz_frame, 
                      label, 
                      (x1, y2 + 7), 
                      font, font_scale, 
                      (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    def process_detections(
            self, 
            video_path: str, 
            detections: List[Dict], 
            output_dir: Optional[str] = None, 
            store_results: bool = True,
            visualize_colors: bool = False
        ) -> List[Dict]:

        # Setup outputs if storing results
        if store_results:
            output_dir, output_json_path, color_assignments_video_path, role_assignments_video_path = self._setup_output_paths(video_path, output_dir)
        
        # Load video
        cap = self._load_video(video_path)
        
        # Setup video writer if storing results
        if store_results:
            out, width, height, fps, total_frames = self._setup_video_writer(cap, color_assignments_video_path)
        else:
            out, total_frames = None, None
        
        # Process each frame
        for frame_data in detections:
            frame_idx = frame_data['frame_id']
            frame = self._get_frame(cap, int(frame_idx))
            
            if frame is None:
                continue
            
            viz_frame = frame.copy() if store_results else None
            
            self._process_frame_detections(
                frame, frame_data, output_dir, store_results, 
                visualize_colors, viz_frame, total_frames
            )
            
            if store_results and viz_frame is not None:
                out.write(viz_frame)
        
        # Clean up resources
        cap.release()
        
        if store_results:
            out.release()


        if store_results:
            with open(output_json_path, 'w') as f:
                json.dump(detections, f, indent=2)

        return detections