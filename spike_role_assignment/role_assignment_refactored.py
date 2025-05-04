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
        # Dictionary to store cluster center colors after clustering
        self.cluster_centers = {}

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

    def _lab_to_rgb(self, lab: np.ndarray) -> Tuple[int, int, int]:
        """Convert LAB to RGB color space using OpenCV.
        
        Args:
            lab: LAB color as numpy array
            
        Returns:
            RGB color tuple (R, G, B) with values 0-255
        """ 
        # Ensure lab is in proper format for conversion
        lab_reshaped = np.uint8([[lab]])
        bgr = cv2.cvtColor(lab_reshaped, cv2.COLOR_LAB2BGR)
        # Convert from BGR to RGB and return as tuple
        rgb = tuple(reversed(bgr[0, 0].tolist()))
        return rgb  # Returns (R, G, B) tuple with values 0-255

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
    
    def _identify_background_jersey(self, centers_rgb_255: np.ndarray) -> Tuple[int, int]:
        """Identify which cluster is background (green) vs jersey.
        
        Args:
            centers_rgb_255: RGB values of cluster centers
            
        Returns:
            Indices of background and jersey clusters
        """
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
        output_color_assignments_video_path = os.path.join(output_dir, 'color_assignments_video.mp4')
        output_role_assignments_video_path = os.path.join(output_dir, 'role_assignments_video.mp4')
        
        return output_dir, output_json_path, output_color_assignments_video_path, output_role_assignments_video_path
    
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

        # Aggregate colors for each track ID and print the mapping
        avg_track_colors = self.aggregate_track_colors(detections)
        
        # Visualize the track colors in LAB space
        if store_results:
            self.visualize_track_colors_lab(avg_track_colors, output_dir)
    
        track_labels = self.cluster_tracks_and_assign_labels(avg_track_colors, output_dir, store_results)

        # Add labels and cluster information to detections
        for frame_data in detections:
            for detection in frame_data['detections']:
                track_id = detection.get("track_id")
                if track_id is not None:
                    detection["output_label"] = track_labels.get(track_id, "Unknown")
                    
                    # Add cluster information to metadata
                    if track_id in avg_track_colors and "cluster" in avg_track_colors[track_id]:
                        if "metadata" not in detection:
                            detection["metadata"] = {}
                        detection["metadata"]["cluster"] = avg_track_colors[track_id]["cluster"]

        # Create the role assignments video
        if store_results:
            self._create_role_assignments_video(video_path, detections, track_labels, role_assignments_video_path)

        if store_results:
            with open(output_json_path, 'w') as f:
                json.dump(detections, f, indent=2)

        return detections
    
    def _create_role_assignments_video(self, video_path: str, detections: List[Dict], track_labels: Dict[int, str], output_video_path: str) -> None:
        """Create a video with team role indicators around players' feet.
        
        Args:
            video_path: Path to the input video
            detections: List of detection data by frame
            track_labels: Dictionary mapping track_ids to team labels
            output_video_path: Path to save the output video
        """
        # Load video
        cap = self._load_video(video_path)
        
        # Setup video writer using the existing method
        out, width, height, fps, total_frames = self._setup_video_writer(cap, output_video_path)
        
        # Default colors for teams in case cluster centers aren't available
        default_team_colors = {
            "TEAM A": (255, 0, 0),  # Red
            "TEAM B": (0, 0, 255),  # Blue
            "Unknown": (128, 128, 128)  # Gray
        }
        
        # Process each frame
        for frame_data in detections:
            frame_idx = frame_data['frame_id']
            frame = self._get_frame(cap, int(frame_idx))
            
            if frame is None:
                continue
            
            # Create a copy for visualization
            viz_frame = frame.copy()
            
            # Add frame counter to top left (keeping this minimal indicator)
            frame_text = f"{frame_idx}/{total_frames}"
            cv2.putText(viz_frame, 
                     frame_text, 
                     (10, 30), 
                     cv2.FONT_HERSHEY_SIMPLEX, 
                     0.5, 
                     (0, 0, 0), 
                     1, 
                     cv2.LINE_AA)
            
            # Draw oval indicators around players' feet
            for detection in frame_data['detections']:
                if "bbox" not in detection or detection.get("class") != "person":
                    continue
                
                bbox = detection["bbox"]
                track_id = detection.get("track_id")
                
                if track_id is None:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = bbox
                
                # Calculate the position for the oval around feet
                # The oval should be at the bottom of the bounding box
                feet_y = y2  # Bottom of the bounding box
                center_x = (x1 + x2) // 2  # Center of the bounding box
                
                # Calculate oval dimensions based on the width of the person
                # The width of the oval is proportional to the width of the person
                # but not too large or too small
                box_width = x2 - x1
                oval_width = int(box_width)  # 100% of person width
                oval_height = int(oval_width * 0.4)  # Oval height is 40% of oval width
                
                # Get cluster index directly from detection metadata
                cluster_idx = detection.get("metadata", {}).get("cluster")
                
                # Use the cluster center color if available
                if cluster_idx is not None and cluster_idx in self.cluster_centers:
                    rgb_color = self.cluster_centers[cluster_idx]["rgb"]
                else:
                    # Fallback to default color only if necessary
                    team_label = track_labels.get(track_id, "Unknown")
                    rgb_color = default_team_colors.get(team_label, default_team_colors["Unknown"])
                
                # Convert RGB to BGR for OpenCV
                if not isinstance(rgb_color, tuple):
                    rgb_color = tuple(rgb_color)
                color_bgr = (rgb_color[2], rgb_color[1], rgb_color[0])
                
                # Calculate oval parameters
                angle_start = -10  # Starting angle in degrees
                angle_end = 190  # Ending angle in degrees - 180 means half oval (open at the back)
                
                # Draw the oval (half ellipse)
                cv2.ellipse(
                    viz_frame,
                    (center_x, feet_y),  # Center position
                    (oval_width // 2, oval_height // 2),  # Half width and height
                    0,  # Angle of rotation
                    angle_start,  # Starting angle
                    angle_end,  # Ending angle
                    color_bgr,  # Color
                    2  # Line thickness
                )
                
                # Add REF/GK text label underneath the oval for these special classes
                team_label = track_labels.get(track_id, "Unknown")
                if team_label.startswith("REF/GK"):
                    text = "REF/GK"
                    # Calculate text position (centered below the oval)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = feet_y + oval_height // 2 + 15  # Position below the oval
                    
                    # Add black text without background
                    cv2.putText(
                        viz_frame, 
                        text, 
                        (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,  # Font scale
                        (0, 0, 0),  # Black text
                        1,  # Line thickness
                        cv2.LINE_AA
                    )
            
            # Write frame to output video
            out.write(viz_frame)
        
        # Clean up resources
        cap.release()
        out.release()
    
    def aggregate_track_colors(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Aggregate color information for each track ID across all frames.
        
        Args:
            detections: List of detection data by frame
            
        Returns:
            Dictionary mapping track_ids to their majority-voted colors
        """
        # Dictionary to store raw jersey colors for each track ID
        track_final_colors = {}
        track_raw_colors = {}
        
        # Go through all frames and collect raw jersey RGB values for each track
        for frame_data in detections:
            for detection in frame_data.get('detections', []):
                if detection.get("class") != "person":
                    continue
                
                track_id = detection.get("track_id")
                if track_id is None:
                    continue
                if track_id not in track_final_colors:
                    track_final_colors[track_id] = []
                if track_id not in track_raw_colors:
                    track_raw_colors[track_id] = []

                final_color = detection["color_info"]["final_color"]
                if final_color:
                    final_color_lab = self.predefined_colors_lab[final_color]
                    track_final_colors[track_id].append(final_color_lab)
                raw_color = detection["color_info"]["raw_jersey_rgb"]
                track_raw_colors[track_id].append(raw_color)

        avg_track_colors = {}
        for track_id, final_colors in track_final_colors.items():
            if not final_colors:
                raw_colors_rgb = track_raw_colors[track_id]
                if not raw_colors_rgb:
                    print("WARNING NO RAW COLORS FOUND FOR TRACK ID: ", track_id)
                    continue
                raw_colors_lab = [self._rgb_to_lab(color) for color in raw_colors_rgb]

                    
                avg_lab = np.mean(raw_colors_lab, axis=0)
                avg_rgb = self._lab_to_rgb(avg_lab)
                avg_track_colors[track_id] = {
                    "avg_lab": avg_lab,
                    "avg_rgb": avg_rgb,
                }
                continue
            
            # Calculate the average LAB color
            avg_lab = np.mean(final_colors, axis=0)
            avg_rgb = self._lab_to_rgb(avg_lab)
            avg_track_colors[track_id] = {
                "avg_lab": avg_lab,
                "avg_rgb": avg_rgb,
            }
        
        return avg_track_colors
    
    def visualize_track_colors_lab(self, track_colors: Dict[int, Dict], output_dir: Optional[str] = None) -> None:
        """Create a scatter plot of track colors in LAB color space.
        
        Args:
            track_colors: Dictionary mapping track_ids to color information
            output_dir: Optional directory to save the visualization
        """
        if not track_colors:
            print("No track colors to visualize")
            return
            
        try:
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
                
                # Skip if NaN values are present
                if np.isnan(lab_color).any():
                    continue
                    
                normalized_rgb = tuple(c/255 for c in rgb_color)
                a, b, L = lab_color[1], lab_color[2], lab_color[0]
                
                ax.scatter(a, b, L, c=[normalized_rgb], s=150, 
                         marker='*', label=f"Track {track_id}")
                
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
        except Exception as e:
            print(f"Error visualizing track colors: {e}")
            # Ensure figure is closed even if error occurs
            try:
                plt.close(fig)
            except:
                pass
    
    def cluster_tracks_and_assign_labels(self, avg_track_colors: Dict[int, Dict], output_dir: Optional[str] = None, store_results: bool = True) -> Dict[int, str]:
        """Cluster track colors using DBSCAN with dynamic eps adjustment to find teams and outliers.
        
        Args:
            avg_track_colors: Dictionary mapping track_ids to color information
            output_dir: Optional directory to save visualization and labels
            store_results: Whether to save results to disk
            
        Returns:
            Dictionary mapping track_ids to team labels
        """
        if not avg_track_colors:
            return {}
        
        # Extract track IDs and LAB colors
        track_ids = list(avg_track_colors.keys())
        lab_colors = np.array([avg_track_colors[track_id]["avg_lab"] for track_id in track_ids])
        
        # Check if we have enough valid data for clustering
        if len(track_ids) < 2:
            # Not enough data for meaningful clustering
            track_labels = {track_id: "Unknown" for track_id in track_ids}
            return track_labels
        
        # Calculate min_samples as 25% of the total tracks, with minimum of 2
        min_samples = max(2, int(len(track_ids) * 0.25))
        
        # Start with a conservative eps value
        # Initial eps should be large enough to put most points in one cluster
        eps = 50.0  # Initial large value in LAB space
        min_eps = 5.0  # Don't go below this to avoid excessive fragmentation
        step = 5.0  # Decrement step
        
        main_clusters = 0
        best_eps = None
        best_labels = None
        
        try:
            # Dynamically adjust eps until we get exactly 2 main clusters
            while eps >= min_eps:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(lab_colors)
                
                # Count non-outlier clusters (clusters not labeled as -1)
                unique_clusters = np.unique(labels)
                non_outlier_clusters = [c for c in unique_clusters if c != -1]
                num_clusters = len(non_outlier_clusters)
                
                # If we have exactly 2 clusters, we've found our teams
                if num_clusters == 2:
                    best_eps = eps
                    best_labels = labels
                    break
                # If we have more than 2 main clusters, we need to increase eps
                elif num_clusters > 2:
                    eps += step
                    break  # We'll use this result with >2 clusters rather than having just 1
                # Otherwise we have 0 or 1 clusters, so decrease eps to try to split them
                else:
                    eps -= step
            
            # If we couldn't find exactly 2 clusters, use the last result or fallback to K-means
            if best_labels is None:
                # Use the last DBSCAN result if we have multiple clusters
                if num_clusters >= 2:
                    best_labels = labels
                else:
                    # Fallback to K-means if DBSCAN couldn't find multiple clusters
                    kmeans = KMeans(n_clusters=min(2, len(lab_colors)), n_init=10, random_state=42)
                    best_labels = kmeans.fit_predict(lab_colors)
                    # All points assigned to clusters in K-means (no -1 labels)
        
        except Exception as e:
            msg = f"Clustering error: {e}"
            print(msg)
            raise Exception(msg)
            
        # Count samples in each non-outlier cluster (-1 is outliers)
        # We only count non-outlier clusters, as we'll handle outliers separately
        cluster_counts = {}
        for label in best_labels:
            if label != -1:  # Skip outliers
                if label not in cluster_counts:
                    cluster_counts[label] = 0
                cluster_counts[label] += 1
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Map cluster indices to team labels
        cluster_to_team = {}        
        # Assign the two largest clusters to TEAM A and TEAM B
        ref_number = 1
        for i, (cluster_idx, _) in enumerate(sorted_clusters):
            if i == 0:
                cluster_to_team[cluster_idx] = "TEAM A"
            elif i == 1:
                cluster_to_team[cluster_idx] = "TEAM B"
            else:
                # Any additional non-outlier clusters beyond the top 2
                # Number them REF/GK 1, 2, 3, ...
                cluster_to_team[cluster_idx] = f"REF/GK {ref_number}"
                ref_number += 1
                
        # Handle outliers (-1 label)
        # We'll group outliers by assigning a new cluster ID to each outlier
        # Create a mapping of outlier tracks to new cluster IDs
        outlier_tracks = []
        for i, label in enumerate(best_labels):
            if label == -1:
                outlier_tracks.append(track_ids[i])
        
        # Assign new cluster IDs to outliers (continue from max_cluster_id + 1)
        max_cluster_id = max(np.max(best_labels), -1)  # At least 0 or higher
        outlier_cluster_map = {}
        for i, track_id in enumerate(outlier_tracks):
            new_cluster_id = max_cluster_id + 1 + i
            outlier_cluster_map[track_id] = new_cluster_id
            # Assign a unique REF/GK label
            cluster_to_team[new_cluster_id] = f"REF/GK {ref_number}"
            ref_number += 1

        # Calculate the center color for each cluster
        self.cluster_centers = {}
        # First, handle the non-outlier clusters
        for cluster_idx in np.unique(best_labels):
            if cluster_idx == -1:
                # Skip outliers, we'll handle them separately
                continue
                
            # Get all points in this cluster
            mask = best_labels == cluster_idx
            cluster_lab_points = lab_colors[mask]
            
            # Calculate the mean LAB color
            center_lab = np.mean(cluster_lab_points, axis=0)
            
            # Convert to RGB
            center_rgb = self._lab_to_rgb(center_lab)
            
            # Store center colors
            self.cluster_centers[cluster_idx] = {
                "lab": center_lab,
                "rgb": center_rgb,
                "team_label": cluster_to_team.get(cluster_idx, "Unknown")
            }
        
        # Now handle the outliers (each as its own cluster)
        for i, track_id in enumerate(outlier_tracks):
            # Get the assigned cluster ID for this outlier
            cluster_idx = outlier_cluster_map[track_id]
            
            # Get the color for this outlier
            lab_color = avg_track_colors[track_id]["avg_lab"]
            rgb_color = avg_track_colors[track_id]["avg_rgb"]
            
            # Store it as a cluster center
            self.cluster_centers[cluster_idx] = {
                "lab": lab_color,
                "rgb": rgb_color,
                "team_label": cluster_to_team.get(cluster_idx, "Unknown")
            }
        
        # Assign team labels to track IDs
        track_labels = {}
        track_clusters = {}  # Store cluster ID for each track
        
        for i, track_id in enumerate(track_ids):
            original_cluster_idx = best_labels[i]
            
            # For non-outliers, use the original cluster ID
            if original_cluster_idx != -1:
                cluster_idx = original_cluster_idx
                team_label = cluster_to_team.get(cluster_idx, "Unknown")
            else:
                # For outliers, use the new cluster ID
                cluster_idx = outlier_cluster_map[track_id]
                team_label = cluster_to_team.get(cluster_idx, "Unknown")
            
            track_labels[track_id] = team_label
            track_clusters[track_id] = int(cluster_idx)
            
            # Also store the cluster ID in the track_colors dict for visualization
            avg_track_colors[track_id]["cluster"] = int(cluster_idx)
            avg_track_colors[track_id]["team_label"] = team_label
            
            # Add cluster center color reference
            if cluster_idx in self.cluster_centers:
                avg_track_colors[track_id]["cluster_center_lab"] = self.cluster_centers[cluster_idx]["lab"]
                avg_track_colors[track_id]["cluster_center_rgb"] = self.cluster_centers[cluster_idx]["rgb"]
        
        # Visualize the clusters if requested
        if output_dir and store_results:
            try:
                self._visualize_dbscan_clusters(lab_colors, best_labels, best_eps, min_samples, output_dir, outlier_cluster_map)
            except Exception as e:
                print(f"Visualization error: {e}")
        
        # Save labels to JSON file if output_dir is provided
        if output_dir and store_results:
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
                    json.dump(track_labels, f, indent=2)
            except Exception as e:
                print(f"Error saving labels: {e}")
        
        return track_labels
    
    def _visualize_dbscan_clusters(self, lab_colors, labels, eps, min_samples, output_dir, outlier_cluster_map):
        """Visualize DBSCAN clustering results.
        
        Args:
            lab_colors: Array of LAB color values
            labels: Cluster labels from DBSCAN
            eps: Epsilon value used for DBSCAN
            min_samples: Min samples value used for DBSCAN
            output_dir: Directory to save the visualization
            outlier_cluster_map: Mapping of outlier tracks to new cluster IDs
        """
        plt.figure(figsize=(12, 10))
        
        # Create a 3D plot
        ax = plt.subplot(111, projection='3d')
        
        # Get unique non-outlier clusters
        unique_clusters = np.unique(labels)
        
        # Create a track_id to index mapping for outliers
        track_to_idx = {track_id: i for i, track_id in enumerate(outlier_cluster_map.keys())}
        
        # Plot each original (non-outlier) cluster
        for cluster_idx in unique_clusters:
            if cluster_idx == -1:
                # Skip outliers, we'll handle them separately
                continue
                
            # Get points in this cluster
            mask = labels == cluster_idx
            cluster_points = lab_colors[mask]
            
            # Get cluster information
            if cluster_idx in self.cluster_centers:
                cluster_info = self.cluster_centers[cluster_idx]
                team_label = cluster_info["team_label"]
                center_rgb = cluster_info["rgb"]
                # Normalize RGB for matplotlib
                color = tuple(c/255 for c in center_rgb)
            else:
                team_label = "Unknown Cluster"
                color = (0.5, 0.5, 0.5)  # Gray
            
            # Plot the points
            ax.scatter(
                cluster_points[:, 1],  # a* channel
                cluster_points[:, 2],  # b* channel
                cluster_points[:, 0],  # L* channel
                color=color,
                marker='o',
                s=100,
                label=team_label
            )
            
            # Plot the cluster center with a star marker
            center = self.cluster_centers[cluster_idx]["lab"]
            ax.scatter(
                center[1],  # a* channel
                center[2],  # b* channel
                center[0],  # L* channel
                color=color,
                marker='*',
                s=300,
                edgecolors='black'
            )
            
            # Add text label at the center
            ax.text(
                center[1],
                center[2],
                center[0],
                team_label,
                fontsize=12,
                weight='bold'
            )
        
        # Now plot each outlier as its own cluster
        outlier_mask = labels == -1
        if np.any(outlier_mask):
            # Get indices of outliers
            outlier_indices = np.where(outlier_mask)[0]
            
            # Get the track IDs that correspond to these indices
            outlier_track_ids = [list(outlier_cluster_map.keys())[i] for i in range(len(outlier_cluster_map))]
            
            # For each outlier, plot it with its assigned color and label
            for i, track_id in enumerate(outlier_track_ids):
                new_cluster_id = outlier_cluster_map[track_id]
                
                if new_cluster_id not in self.cluster_centers:
                    continue
                    
                cluster_info = self.cluster_centers[new_cluster_id]
                team_label = cluster_info["team_label"]
                center_rgb = cluster_info["rgb"]
                center_lab = cluster_info["lab"]
                
                # Normalize RGB for matplotlib
                color = tuple(c/255 for c in center_rgb)
                
                # Plot the outlier point
                # Find the index of this track_id in the original data
                idx = outlier_indices[i] if i < len(outlier_indices) else None
                if idx is not None:
                    point = lab_colors[idx]
                    
                    ax.scatter(
                        point[1],  # a* channel
                        point[2],  # b* channel
                        point[0],  # L* channel
                        color=color,
                        marker='x',
                        s=100,
                        label=team_label
                    )
                
                # Plot the center (which is the same as the point for outliers)
                ax.scatter(
                    center_lab[1],  # a* channel
                    center_lab[2],  # b* channel
                    center_lab[0],  # L* channel
                    color=color,
                    marker='*',
                    s=300,
                    edgecolors='black'
                )
                
                # Add text label
                ax.text(
                    center_lab[1],
                    center_lab[2],
                    center_lab[0],
                    team_label,
                    fontsize=12,
                    weight='bold'
                )
        
        # Add labels and title
        ax.set_xlabel('a* (Green-Red)')
        ax.set_ylabel('b* (Blue-Yellow)')
        ax.set_zlabel('L* (Lightness)')
        ax.set_title(f'Team Color Clustering: eps={eps:.1f}, min_samples={min_samples}')
        
        # Add legend with unique labels only
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'dbscan_clusters.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
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
        plt.show()
        plt.close(fig) 