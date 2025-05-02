import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from skimage import color
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


class RoleAssigner:
    def __init__(self):
        self.predefined_colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 128, 0),
            "light_green": (144, 238, 144),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "sky_blue": (135, 206, 235),
            "grey_blue": [78, 94, 128],
            "navy_blue": (0, 0, 128),
            "brown": (160, 100, 80),
            "maroon": (128, 0, 0),
            "pink": (255, 192, 203),
            "cyan_blue": (0, 154, 255)
        }
        self.predefined_colors_lab = {name: self._rgb_to_lab(rgb) for name, rgb in self.predefined_colors.items()}

    def _load_video(self, video_path: str) -> cv2.VideoCapture:
        """Load a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoCapture object
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return cv2.VideoCapture(video_path)
    
    def _get_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame from the video.
        
        Args:
            cap: VideoCapture object
            frame_idx: Index of the frame to retrieve
            
        Returns:
            Frame as numpy array, or None if frame couldn't be read
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame 

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> np.ndarray:
        """Convert RGB to LAB color space using OpenCV.
        
        Args:
            rgb: RGB color tuple (R, G, B) with values 0-255
            
        Returns:
            LAB color as numpy array
        """
        bgr = np.uint8([[list(reversed(rgb))]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab[0, 0].astype(np.float32)  # Convert to float32 for calculations
    
    def _extract_roi(self, frame: np.ndarray, bbox: List[int], y_range: Tuple[float, float], x_range: Tuple[float, float]) -> np.ndarray:
        """Extract and crop region of interest from a frame.
        
        Args:
            frame: Video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            y_range: Range of y-coordinates to crop (normalized)
            x_range: Range of x-coordinates to crop (normalized)
            
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
        """Cluster pixel colors using KMeans.
        
        Args:
            pixels: RGB pixel values
            
        Returns:
            Tuple containing cluster centers, labels, and related data
        """
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
    
    def _identify_background_jersey(self, centers_rgb: np.ndarray) -> Tuple[int, int]:
        """Identify which cluster is background (green) vs jersey.
        
        Args:
            centers_rgb: RGB values of cluster centers
            
        Returns:
            Indices of background and jersey clusters
        """
        # Calculate how "green" each cluster is
        green_scores = []
        for rgb in centers_rgb:
            green_score = rgb[1] - (rgb[0] + rgb[2]) / 2
            green_scores.append(green_score)
        
        # Get index of the greenest cluster (background)
        background_idx = np.argmax(green_scores)
        jersey_idx = 1 - background_idx  # The other cluster is the jersey
        
        return background_idx, jersey_idx
    
    def find_closest_predefined_color(self, rgb_color: np.ndarray) -> str:
        """Find the closest predefined color to the given RGB color using LAB color space.
        
        Args:
            rgb_color: RGB color array [R, G, B] with values 0-255
            
        Returns:
            The name of the closest predefined color
        """
        rgb_tuple = tuple(map(int, rgb_color))
        lab_color = self._rgb_to_lab(rgb_tuple)
        
        min_distance = float('inf')
        closest_color_name = None
        
        for name, predefined_lab in self.predefined_colors_lab.items():
            # Calculate Euclidean distance in LAB space
            delta_l = float(lab_color[0]) - float(predefined_lab[0])
            delta_a = float(lab_color[1]) - float(predefined_lab[1])
            delta_b = float(lab_color[2]) - float(predefined_lab[2])
            
            # Standard Euclidean distance
            distance = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_color_name = name
        
        return closest_color_name 

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
        """Extract dominant background and jersey colors from player bounding box.
        
        Args:
            frame: Video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            output_dir: Directory to save visualizations (if visualize=True)
            frame_id: Frame index (for visualization filename)
            track_id: Track ID (for visualization filename)
            y_range: Range of y-coordinates to analyze (normalized)
            x_range: Range of x-coordinates to analyze (normalized)
            visualize: Whether to generate and save visualization
            
        Returns:
            Tuple of (background_color, jersey_color) as RGB tuples
        """
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
            background_idx, jersey_idx = self._identify_background_jersey(centers_rgb)
            
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
        """Create and save visualization of color clustering results.
        
        Args:
            frame: Original video frame
            bbox: Player bounding box coordinates
            roi_rgb: RGB player region
            output_dir: Directory to save visualization
            frame_id: Frame index
            track_id: Player track ID
            x_range: X cropping range used
            y_range: Y cropping range used
            labels: Cluster labels for each pixel
            sorted_centers_rgb: Sorted RGB cluster centers
            sorted_centers_rgb_255: Sorted RGB cluster centers (0-255 scale)
            cluster_order: Order of clusters (background first, jersey second)
            percentages: Percentage of pixels in each cluster
            centers_lab: Cluster centers in LAB color space
        """
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
        """Calculate metadata for a detection that helps with role assignment.
        
        Args:
            frame: Video frame
            bbox: Bounding box coordinates
            all_background_colors_rgb: List of all background colors from all detections
            background_color_rgb: Background color for this detection
            jersey_color_rgb: Jersey color for this detection
            detections: List of all detections in the frame
            
        Returns:
            Dictionary with metadata attributes
        """
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
        """Check if a bounding box overlaps with any other person detections.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            detections: List of all detections in the frame
            
        Returns:
            True if the bbox overlaps with any other person detection
        """
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
        """Determine if a background color is an outlier compared to other backgrounds.
        
        Args:
            all_background_colors_rgb: List of all background colors
            background_color_rgb: Background color to check
            
        Returns:
            True if the background color is an outlier
        """
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
        """Set up output directory and file paths.
        
        Args:
            video_path: Path to input video
            output_dir: Optional output directory
            
        Returns:
            Tuple of (output_dir, output_json_path, output_video_path)
        """
        if output_dir is None:
            base_name = os.path.basename(video_path).split('.')[0]
            output_dir = os.path.join('role_assignment_results', base_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_json_path = os.path.join(output_dir, 'role_assignments.json')
        output_video_path = os.path.join(output_dir, 'role_assignments_video.mp4')
        
        return output_dir, output_json_path, output_video_path
    
    def _setup_video_writer(self, cap: cv2.VideoCapture, output_video_path: str) -> Tuple[cv2.VideoWriter, int, int, float, int]:
        """Set up video writer for visualization.
        
        Args:
            cap: VideoCapture object
            output_video_path: Path to save output video
            
        Returns:
            Tuple of (video_writer, width, height, fps, total_frames)
        """
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
        """Process all detections in a single frame.
        
        Args:
            frame: Video frame
            frame_data: Frame data including detections
            output_dir: Output directory for visualizations
            store_results: Whether to store results
            visualize_colors: Whether to visualize color processing
            viz_frame: Copy of frame for visualization (if store_results=True)
            total_frames: Total number of frames in video (for visualization)
        """
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
        
        # First pass, get all dominant color info
        dominant_colors = self._extract_dominant_colors(
            frame, frame_data, output_dir, frame_idx, visualize_colors
        )
        
        # Second pass, add color info and metadata to detections
        self._add_color_metadata_to_detections(
            frame, frame_data, dominant_colors, store_results, viz_frame
        )
    
    def _extract_dominant_colors(
            self,
            frame: np.ndarray,
            frame_data: Dict,
            output_dir: Optional[str],
            frame_idx: int,
            visualize_colors: bool
        ) -> Dict[int, Dict]:
        """Extract dominant colors for all detections in a frame.
        
        Args:
            frame: Video frame
            frame_data: Frame data including detections
            output_dir: Output directory for visualizations
            frame_idx: Frame index
            visualize_colors: Whether to visualize color processing
            
        Returns:
            Dictionary mapping track_ids to dominant colors
        """
        dominant_colors = {}
        for detection in frame_data['detections']:
            if "bbox" not in detection or detection.get("class") != "person":
                continue
                
            bbox = detection["bbox"]
            track_id = detection.get("track_id")
            
            if track_id is None:
                continue
            
            background_color, jersey_color_rgb = self.get_dominant_color(
                frame, bbox, output_dir, frame_idx, track_id, visualize=visualize_colors
            )
            
            dominant_colors[track_id] = {
                "background_color": background_color,
                "jersey_color": jersey_color_rgb
            }
        
        return dominant_colors
    
    def _add_color_metadata_to_detections(
            self,
            frame: np.ndarray,
            frame_data: Dict,
            dominant_colors: Dict[int, Dict],
            store_results: bool,
            viz_frame: Optional[np.ndarray] = None
        ) -> None:
        """Add color info and metadata to each detection.
        
        Args:
            frame: Video frame
            frame_data: Frame data including detections
            dominant_colors: Dictionary of dominant colors by track_id
            store_results: Whether to store/visualize results
            viz_frame: Frame to draw visualization on
        """
        for detection in frame_data['detections']:
            if "bbox" not in detection or detection.get("class") != "person":
                continue
            
            bbox = detection["bbox"]
            track_id = detection.get("track_id")
            
            if track_id is None or track_id not in dominant_colors:
                continue

            background_color_rgb = dominant_colors[track_id]["background_color"]
            jersey_color_rgb = dominant_colors[track_id]["jersey_color"]
            closest_color = self.find_closest_predefined_color(jersey_color_rgb)

            color_info = {
                "raw_background_rgb": background_color_rgb,
                "raw_jersey_rgb": jersey_color_rgb,
                "closest_jersey_color": closest_color,
            }

            all_background_colors_rgb = [item["background_color"] for item in dominant_colors.values()]
            
            # Calculate and add jersey attributes
            metadata = self._calculate_detections_metadata(
                frame, bbox, all_background_colors_rgb, background_color_rgb, 
                jersey_color_rgb, frame_data['detections']
            )

            final_color = closest_color
            if metadata["background_is_outlier"]: 
                final_color = None
            color_info["final_color"] = final_color
            
            detection["color_info"] = color_info
            detection["metadata"] = metadata

            if store_results and viz_frame is not None:
                self._draw_detection_visualization(viz_frame, bbox, track_id, final_color)
    
    def _draw_detection_visualization(
            self,
            viz_frame: np.ndarray,
            bbox: List[int],
            track_id: int,
            final_color: Optional[str]
        ) -> None:
        """Draw visualization for a detection on the frame.
        
        Args:
            viz_frame: Frame to draw on
            bbox: Bounding box coordinates
            track_id: Track ID to label the box
            final_color: Final assigned color name
        """
        x1, y1, x2, y2 = bbox
        
        if final_color in self.predefined_colors:
            color_rgb = self.predefined_colors[final_color]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            border_thickness = 2
        else:
            color_bgr = (0, 0, 0)  # black
            border_thickness = 1

        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color_bgr, border_thickness)
        
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
        """Process detections to assign team roles based on jersey colors.
        
        Args:
            video_path: Path to the video file
            detections: List of detection data by frame
            output_dir: Directory to save results (if None, auto-generated)
            store_results: Whether to store JSON results and visualizations
            visualize_colors: Whether to create detailed color visualizations
            
        Returns:
            List of detection data with added color and role information
        """
        # Setup outputs if storing results
        if store_results:
            output_dir, output_json_path, output_video_path = self._setup_output_paths(video_path, output_dir)
        
        # Load video
        cap = self._load_video(video_path)
        
        # Setup video writer if storing results
        if store_results:
            out, width, height, fps, total_frames = self._setup_video_writer(cap, output_video_path)
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
            
            with open(output_json_path, 'w') as f:
                json.dump(detections, f, indent=2)

        # Aggregate colors for each track ID and print the mapping
        track_color_mapping = self.aggregate_track_colors(detections)
        
        # Visualize the track colors in LAB space
        if store_results:
            self.visualize_track_colors_lab(track_color_mapping, output_dir)
            self.cluster_tracks_and_assign_labels(track_color_mapping, output_dir)
            self.visualize_clustered_tracks(track_color_mapping, output_dir)

        return detections
    
    def aggregate_track_colors(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Aggregate color information for each track ID across all frames.
        
        Args:
            detections: List of detection data by frame
            
        Returns:
            Dictionary mapping track_ids to their majority-voted colors
        """
        # Dictionary to store raw jersey colors for each track ID
        track_raw_colors = {}
        
        # Go through all frames and collect raw jersey RGB values for each track
        for frame_data in detections:
            for detection in frame_data.get('detections', []):
                if detection.get("class") != "person":
                    continue
                
                track_id = detection.get("track_id")
                if track_id is None:
                    continue
                
                # Get the raw jersey color from the detection
                color_info = detection.get("color_info", {})
                raw_jersey_rgb = color_info.get("raw_jersey_rgb")
                
                if raw_jersey_rgb is not None:
                    # Initialize track entry if not exists
                    if track_id not in track_raw_colors:
                        track_raw_colors[track_id] = []
                    
                    # Add this color to the list
                    track_raw_colors[track_id].append(raw_jersey_rgb)
        
        # Average the colors in LAB space for each track
        track_avg_colors = {}
        for track_id, rgb_colors in track_raw_colors.items():
            if not rgb_colors:
                continue
            
            # Convert all RGB values to LAB
            lab_colors = [self._rgb_to_lab(rgb) for rgb in rgb_colors]
            
            # Calculate the average LAB color
            avg_lab = np.mean(lab_colors, axis=0)
            
            # Convert back to RGB for visualization and naming
            avg_lab_reshaped = avg_lab.reshape(1, 1, 3)
            bgr_color = cv2.cvtColor(np.uint8(avg_lab_reshaped), cv2.COLOR_Lab2BGR)
            avg_rgb = tuple(reversed(bgr_color[0, 0].tolist()))
            
            # Find the closest predefined color
            closest_color = self.find_closest_predefined_color(avg_rgb)
            
            # Store both the average RGB and the closest named color
            track_avg_colors[track_id] = {
                "avg_rgb": avg_rgb,
                "avg_lab": avg_lab,
                "closest_color": closest_color
            }
        
        return track_avg_colors
    
    def visualize_track_colors_lab(self, track_colors: Dict[int, Dict], output_dir: Optional[str] = None) -> None:
        """Create a scatter plot of track colors in LAB color space.
        
        Args:
            track_colors: Dictionary mapping track_ids to color information
            output_dir: Optional directory to save the visualization
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the predefined colors
        for name, lab_color in self.predefined_colors_lab.items():
            rgb_color = self.predefined_colors[name]
            normalized_rgb = tuple(c/255 for c in rgb_color)
            
            a, b, L = lab_color[1], lab_color[2], lab_color[0]
            
            ax.scatter(a, b, L, c=[normalized_rgb], s=100, alpha=0.5, 
                      marker='o', label=f"Ref: {name}")
        
        # Plot the track colors
        for track_id, color_info in track_colors.items():
            lab_color = color_info["avg_lab"]
            rgb_color = color_info["avg_rgb"]
            closest = color_info["closest_color"]
            
            normalized_rgb = tuple(c/255 for c in rgb_color)
            a, b, L = lab_color[1], lab_color[2], lab_color[0]
            
            ax.scatter(a, b, L, c=[normalized_rgb], s=150, 
                     marker='*', label=f"Track {track_id}: {closest}")
            
            # Add track ID label next to the point
            ax.text(a, b, L, f" {track_id}", fontsize=9)
        
        ax.set_xlabel('a* (Green-Red)')
        ax.set_ylabel('b* (Blue-Yellow)')
        ax.set_zlabel('L* (Lightness)')
        ax.set_title('Track Colors in LAB Space')
        
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'track_colors_lab.png'), dpi=150, bbox_inches='tight')
        
        # Remove plt.show() to prevent visualization display
        plt.close(fig)
    
    def cluster_tracks_and_assign_labels(self, track_colors: Dict[int, Dict], output_dir: Optional[str] = None) -> Dict[int, str]:
        """Cluster track colors using K-means and assign team labels.
        
        Args:
            track_colors: Dictionary mapping track_ids to color information
            output_dir: Optional directory to save visualization and labels
            
        Returns:
            Dictionary mapping track_ids to team labels
        """
        if not track_colors:
            return {}
        
        # Extract track IDs and LAB colors
        track_ids = list(track_colors.keys())
        lab_colors = np.array([track_colors[track_id]["avg_lab"] for track_id in track_ids])
        
        # Determine optimal number of clusters using elbow method with MCSS
        max_clusters = min(len(track_ids), 10)  # Cap at 10 clusters
        mcss_scores = []
        
        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(lab_colors)
            
            # Calculate Mean Cluster Sum of Squares (MCSS)
            # This is the mean of the sum of squared distances of points to their assigned cluster center
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Sum of squared distances for each point
            sum_squared_dists = 0
            for i, point in enumerate(lab_colors):
                center = cluster_centers[labels[i]]
                sum_squared_dists += np.sum((point - center) ** 2)
            
            # Mean of the sum of squared distances
            mcss = sum_squared_dists / len(lab_colors)
            mcss_scores.append(mcss)
        
        # Find elbow point
        optimal_clusters = self._find_elbow_point(range(1, max_clusters + 1), mcss_scores)
        
        # If we couldn't determine optimal clusters or it's 1, use at least 2
        if optimal_clusters is None or optimal_clusters < 2:
            optimal_clusters = min(2, len(track_ids))
        
        # Visualize elbow curve if output_dir is provided
        if output_dir:
            self._visualize_elbow_curve(range(1, max_clusters + 1), mcss_scores, optimal_clusters, output_dir)
        
        # Apply K-means with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(lab_colors)
        
        # Count samples in each cluster
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        
        # Sort clusters by size (largest first)
        sorted_clusters_with_idx = sorted(
            [(cluster_idx, count, i) for i, (cluster_idx, count) in enumerate(zip(unique_clusters, cluster_counts))], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Map cluster indices to team labels
        cluster_to_team = {}
        for i, (cluster_idx, _, orig_idx) in enumerate(sorted_clusters_with_idx):
            if i == 0:
                cluster_to_team[cluster_idx] = "TEAM A"
            elif i == 1:
                cluster_to_team[cluster_idx] = "TEAM B"
            else:
                cluster_to_team[cluster_idx] = f"REF/GK {i-1}"
        
        # Assign team labels to track IDs
        track_labels = {}
        for i, track_id in enumerate(track_ids):
            cluster_idx = cluster_labels[i]
            team_label = cluster_to_team[cluster_idx]
            track_labels[track_id] = team_label
            
            # Also store the cluster ID in the track_colors dict for visualization
            track_colors[track_id]["cluster"] = int(cluster_idx)
            track_colors[track_id]["team_label"] = team_label
        
        # Save labels to JSON file if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
                json.dump(track_labels, f, indent=2)
        
        return track_labels
    
    def _find_elbow_point(self, k_values, mcss_scores):
        """Find the elbow point in the MCSS curve using the maximum curvature method.
        
        Args:
            k_values: List of k values (number of clusters)
            mcss_scores: Corresponding MCSS scores
            
        Returns:
            Optimal number of clusters
        """
        if len(k_values) < 3:
            return k_values[0]
        
        # Normalize the data for better calculation
        x = np.array(k_values)
        y = np.array(mcss_scores)
        
        # Normalize coordinates
        x_norm = (x - min(x)) / (max(x) - min(x))
        y_norm = (y - min(y)) / (max(y) - min(y))
        
        # Calculate curvature
        dx_dt = np.gradient(x_norm)
        dy_dt = np.gradient(y_norm)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        
        # Find the point of maximum curvature
        elbow_idx = np.argmax(curvature[1:-1]) + 1  # Skip first and last points
        
        return k_values[elbow_idx]
    
    def _visualize_elbow_curve(self, k_values, mcss_scores, optimal_k, output_dir):
        """Visualize the elbow curve and the chosen optimal k.
        
        Args:
            k_values: List of k values
            mcss_scores: Corresponding MCSS scores
            optimal_k: The chosen optimal number of clusters
            output_dir: Directory to save the visualization
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, mcss_scores, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Mean Cluster Sum of Squares (MCSS)')
        plt.title('Elbow Method for Optimal k Selection')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'elbow_curve.png'), dpi=150)
        plt.close()
    
    def visualize_clustered_tracks(self, track_colors: Dict[int, Dict], output_dir: Optional[str] = None) -> None:
        """Create a 3D scatter plot showing track colors with cluster labels.
        
        Args:
            track_colors: Dictionary mapping track_ids to color information
            output_dir: Optional directory to save the visualization
        """
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up colors for different teams
        team_colors = {
            "TEAM A": "red",
            "TEAM B": "blue",
            "REF/GK": "black"
        }
        
        # Group tracks by cluster for the legend
        teams = {}
        for track_id, color_info in track_colors.items():
            team_label = color_info.get("team_label", "Unknown")
            if team_label not in teams:
                teams[team_label] = []
            teams[team_label].append(track_id)
        
        # Plot the track colors
        for track_id, color_info in track_colors.items():
            lab_color = color_info["avg_lab"]
            rgb_color = color_info["avg_rgb"]
            team_label = color_info.get("team_label", "Unknown")
            
            a, b, L = lab_color[1], lab_color[2], lab_color[0]
            
            # Use normalized RGB for the point color, but team color for marker edge
            normalized_rgb = tuple(c/255 for c in rgb_color)
            team_color = team_colors.get(team_label, "gray")
            
            ax.scatter(a, b, L, c=[normalized_rgb], s=150, 
                      marker='*', edgecolors=team_color, linewidths=2)
            
            # Add track ID and team label
            ax.text(a, b, L, f" {track_id} ({team_label})", fontsize=9)
        
        # Add team groups to legend
        for team_label, track_ids in teams.items():
            team_color = team_colors.get(team_label, "gray")
            track_str = ", ".join(map(str, sorted(track_ids)))
            ax.scatter([], [], [], c=team_color, marker='o', s=100, 
                      label=f"{team_label}: Tracks {track_str}")
        
        ax.set_xlabel('a* (Green-Red)')
        ax.set_ylabel('b* (Blue-Yellow)')
        ax.set_zlabel('L* (Lightness)')
        ax.set_title('Track Colors in LAB Space with Team Clustering')
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'label_clusters.png'), dpi=150, bbox_inches='tight')
        
        # Remove plt.show() to prevent visualization display
        plt.close(fig)
    
    def visualize_color_comparison(self, rgb_color: np.ndarray) -> None:
        """Visualize the input color and all predefined colors in 3D LAB space.
        
        Args:
            rgb_color: RGB color array [R, G, B] with values 0-255
        """
        # Convert input RGB to LAB
        rgb_tuple = tuple(map(int, rgb_color))
        input_lab = self._rgb_to_lab(rgb_tuple)
        
        # Find closest color
        closest_color_name = self.find_closest_predefined_color(rgb_color)
        closest_lab = self.predefined_colors_lab[closest_color_name]
        
        # Create figure for 3D plotting
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all predefined colors
        for name, lab_color in self.predefined_colors_lab.items():
            rgb_color = self.predefined_colors[name]
            normalized_rgb = tuple(c/255 for c in rgb_color)
            
            # Convert LAB coordinates for plotting
            a, b, L = lab_color[1], lab_color[2], lab_color[0]
            
            ax.scatter(a, b, L, c=[normalized_rgb], s=100, label=name)
        
        # Plot input color
        input_a, input_b, input_L = input_lab[1], input_lab[2], input_lab[0]
        normalized_input_rgb = tuple(c/255 for c in rgb_tuple)
        ax.scatter(input_a, input_b, input_L, c=[normalized_input_rgb], s=200, marker='*', label='Input color')
        
        # Draw line to closest color
        closest_a, closest_b, closest_L = closest_lab[1], closest_lab[2], closest_lab[0]
        ax.plot([input_a, closest_a], [input_b, closest_b], [input_L, closest_L], 'k--')
        
        # Set labels and title
        ax.set_xlabel('a* (Green-Red)')
        ax.set_ylabel('b* (Blue-Yellow)')
        ax.set_zlabel('L* (Lightness)')
        ax.set_title(f'Color Comparison in LAB Space\nClosest color: {closest_color_name}')
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        # Remove plt.show() to prevent visualization display
        plt.close(fig) 