import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from skimage import color
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

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
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return cv2.VideoCapture(video_path)
    
    def _get_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    
    def get_dominant_color(
            self, 
            frame: np.ndarray, 
            bbox: List[int], 
            output_dir: str, 
            frame_id: int, 
            track_id: Optional[int] = None, 
            y_range: Tuple[float, float] = (0.0, 0.5), 
            x_range: Tuple[float, float] = (0.0, 1.0),
            visualize: bool = False
        ) -> List[Tuple[int, int, int]]:
        x1, y1, x2, y2 = bbox
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return (0, 0, 0)
        
        height, width = roi.shape[:2]
        y_start = int(height * y_range[0])
        y_end = int(height * y_range[1])
        x_start = int(width * x_range[0])
        x_end = int(width * x_range[1])
        
        if y_end <= y_start or x_end <= x_start:
            return (0, 0, 0)
        
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if visualize else None
        cropped = roi[y_start:y_end, x_start:x_end]
        
        if cropped.size == 0:
            return (0, 0, 0)
        
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pixels = cropped_rgb.reshape(-1, 3)
        
        if len(pixels) < 2:
            return (0, 0, 0)
        
        try:
            rgb_normalized = pixels.astype(np.float32) / 255.0
            lab_pixels = color.rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
            
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            kmeans.fit(lab_pixels)
            
            centers_lab = kmeans.cluster_centers_
            labels = kmeans.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Convert centers to RGB for green comparison
            centers_lab_reshaped = centers_lab.reshape(1, -1, 3)
            centers_rgb = color.lab2rgb(centers_lab_reshaped).reshape(-1, 3)
            centers_rgb_255 = (centers_rgb * 255).astype(int)
            
            # Determine which cluster is more green (higher G value relative to R and B)
            green_scores = []
            for rgb in centers_rgb:
                # Calculate how "green" the color is (G value relative to R and B)
                green_score = rgb[1] - (rgb[0] + rgb[2]) / 2
                green_scores.append(green_score)
            
            # Get index of the greenest cluster (background)
            background_idx = np.argmax(green_scores)
            jersey_idx = 1 - background_idx  # The other cluster is the jersey
            
            # Reorder clusters so background is first, jersey is second
            cluster_order = [background_idx, jersey_idx]
            sorted_centers_rgb = centers_rgb[cluster_order]
            sorted_centers_rgb_255 = centers_rgb_255[cluster_order]
            sorted_centers_lab = centers_lab[cluster_order]
            sorted_counts = counts[cluster_order]
            
            total_pixels = np.sum(counts)
            percentages = counts / total_pixels
            sorted_percentages = percentages[cluster_order]
            
            if visualize:
                vis_dir = os.path.join(output_dir, 'player_visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                track_str = f"track_{track_id}" if track_id is not None else "unknown_track"
                filename = f"{frame_id:04d}_{track_str}.png"
                
                fig = plt.figure(figsize=(18, 12))
                gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 1])
                
                # Display full frame in top left
                ax_frame = plt.subplot(gs[0, 0])
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            
            # Return the jersey color (second cluster after reordering)
            background_color = sorted_centers_rgb_255[0]
            jersey_color = sorted_centers_rgb_255[1]
            return tuple(int(v) for v in background_color), tuple(int(v) for v in jersey_color)
        
        except Exception as e:
            print(f"Error in color clustering: {e}")
            return (0, 0, 0)
    
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
    
    def find_closest_predefined_color(self, rgb_color: np.ndarray) -> str:
        """Find the closest predefined color to the given RGB color using LAB color space.
        
        Args:
            rgb_color: RGB color array [R, G, B] with values 0-255
            
        Returns:
            The name of the closest predefined color
        """
        # Convert RGB to LAB for better perceptual color comparison
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
        bbox_has_overlap = False
        for other_detection in detections:
            if "bbox" not in other_detection or other_detection.get("class") != "person":
                continue
                
            other_bbox = other_detection["bbox"]
            other_x1, other_y1, other_x2, other_y2 = other_bbox
            
            # Skip comparing with self
            if other_x1 == x1 and other_y1 == y1 and other_x2 == x2 and other_y2 == y2:
                continue
                
            # Check for overlap
            # If there's any overlap between boxes
            if not (other_x2 < x1 or other_x1 > x2 or other_y2 < y1 or other_y1 > y2):
                bbox_has_overlap = True
                break
        
        # Determine if background color is an outlier using DBSCAN
        background_is_outlier = False
        if len(all_background_colors_rgb) >= 3:  # Need at least 3 backgrounds to use DBSCAN effectively
            # Convert all background colors to LAB for better perceptual comparison
            background_labs = np.array([self._rgb_to_lab(bg) for bg in all_background_colors_rgb])
            
            # Find the index of the current background color
            current_bg_idx = -1
            for i, bg in enumerate(all_background_colors_rgb):
                if bg == background_color_rgb:
                    current_bg_idx = i
                    break
            
            if current_bg_idx != -1:
                # Apply DBSCAN clustering
                # eps: maximum distance between samples to be considered in the same cluster
                # min_samples: minimum number of samples to form a dense region
                dbscan = DBSCAN(eps=20.0, min_samples=2)
                clusters = dbscan.fit_predict(background_labs)
                
                # Points labeled as -1 are considered outliers by DBSCAN
                background_is_outlier = bool(clusters[current_bg_idx] == -1)
        
        return {
            "bbox_has_overlap": bbox_has_overlap,
            "bbox_size": bbox_size,
            "background_is_outlier": background_is_outlier
        }
    
    def process_detections(self, video_path: str, detections: List[Dict], output_dir: Optional[str] = None, store_results: bool = True) -> List[Dict]:
        if store_results:
            if output_dir is None:
                base_name = os.path.basename(video_path).split('.')[0]
                output_dir = os.path.join('role_assignment_results', base_name)
            
            os.makedirs(output_dir, exist_ok=True)
            
            output_json_path = os.path.join(output_dir, 'role_assignments.json')
            output_video_path = os.path.join(output_dir, 'role_assignments_video.mp4')
        
        cap = self._load_video(video_path)
        
        if store_results:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        for frame_data in detections:
            frame_idx = frame_data['frame_id']
            frame = self._get_frame(cap, int(frame_idx))
            if frame is None:
                continue
            
            if store_results:
                viz_frame = frame.copy()
                
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
            dominant_colors = {}
            for detection in frame_data['detections']:
                if "bbox" not in detection or detection.get("class") != "person":
                    continue
                    
                bbox = detection["bbox"]
                track_id = detection.get("track_id")
                
                background_color, jersey_color_rgb = self.get_dominant_color(
                    frame, bbox, output_dir, frame_idx, track_id, visualize=False
                )
                
                dominant_colors[track_id] = {
                    "background_color": background_color,
                    "jersey_color": jersey_color_rgb
                }
            
            # Second pass, add color info and metadata to detections
            for detection in frame_data['detections']:
                if "bbox" not in detection or detection.get("class") != "person":
                    continue
                
                bbox = detection["bbox"]
                track_id = detection.get("track_id")

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
                    frame, bbox, all_background_colors_rgb, background_color_rgb,jersey_color_rgb, frame_data['detections']
                )

                final_color = closest_color
                if metadata["background_is_outlier"]: 
                    final_color = None
                color_info["final_color"] = final_color
                
                detection["color_info"] = color_info
                detection["metadata"] = metadata

                if store_results:
                    x1, y1, x2, y2 = bbox
                    
                    if final_color in self.predefined_colors:
                        color_rgb = self.predefined_colors[final_color]
                        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                        border_thickness = 2
                    else:
                        color_bgr = (0, 0, 0) # black
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

            
            if store_results:
                out.write(viz_frame)
        
        cap.release()
        
        if store_results:
            out.release()
            
            with open(output_json_path, 'w') as f:
                json.dump(detections, f, indent=2)
        
        return detections
    
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
