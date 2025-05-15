from custom_types import FrameDetections, BBox, TrackRole
from conversions import RGBColor255, LABColor, rgb_to_lab, lab_to_rgb_255
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Set, Any, Mapping
import json
from collections import defaultdict, Counter


class RoleAssigner:
    def __init__(self) -> None:
        self.cluster_centers: Dict[int, Dict[str, Any]] = {}

    def _roi_is_overlapping(self, roi_bbox: BBox, other_roi_bboxes: List[BBox]) -> bool:
        """Check if a region of interest overlaps with any other regions."""
        return any(
            roi_bbox.x1 < other_roi.x2 and roi_bbox.x2 > other_roi.x1 and 
            roi_bbox.y1 < other_roi.y2 and roi_bbox.y2 > other_roi.y1
            for other_roi in other_roi_bboxes
        )
    
    def _get_all_track_ids(self, detections: List[FrameDetections]) -> List[int]:
        """Extract all unique track IDs from detections."""
        track_ids: Set[int] = set()
        for frame_detections in detections:
            for detection in frame_detections.detections:
                track_ids.add(detection.track_id)
        return list(track_ids)
    
    def _list_all_colors(self, track_id: int, detections: List[FrameDetections], allow_overlapping: bool) -> List[RGBColor255]:
        """List all jersey colors for a specific track ID."""
        track_colors: List[RGBColor255] = []
        
        for frame_detections in detections:
            for detection in frame_detections.detections:
                if detection.track_id == track_id:
                    if not allow_overlapping:
                        other_roi_bboxes = [d.roi_bbox for d in frame_detections.detections if d.track_id != track_id]
                        if self._roi_is_overlapping(detection.roi_bbox, other_roi_bboxes):
                            continue
                    track_colors.append(detection.jersey_color)
        
        return track_colors

    def _get_track_colors(self, detections: List[FrameDetections]) -> Dict[int, List[RGBColor255]]:
        """Get jersey colors for all tracks, falling back to overlapping ROIs if needed."""
        track_ids = self._get_all_track_ids(detections)
        track_colors: Dict[int, List[RGBColor255]] = {}
        
        # First pass: get colors without overlapping ROIs
        for track_id in track_ids:
            track_colors[track_id] = self._list_all_colors(
                track_id, detections, allow_overlapping=False
            )
        
        # Second pass: for tracks with no colors, allow overlapping ROIs
        for track_id in track_ids:
            if not track_colors[track_id]:
                track_colors[track_id] = self._list_all_colors(
                    track_id, detections, allow_overlapping=True
                )

        return track_colors
    
    def _get_avg_lab_color(self, colors: List[RGBColor255]) -> LABColor:
        """Calculate the average LAB color from a list of RGB colors."""
        # Convert all colors to LAB and store as arrays
        lab_arrays = [rgb_to_lab(color).to_array() for color in colors]
        
        # Calculate the mean LAB color
        return LABColor.from_array(np.mean(lab_arrays, axis=0))

    def cluster_tracks_and_assign_labels(
        self, 
        avg_lab_colors: Dict[int, LABColor], 
        store_results: bool = True
    ) -> Dict[int, str]:
        """Cluster track colors using DBSCAN with dynamic eps adjustment to find teams and outliers.
        
        Args:
            avg_lab_colors: Dictionary mapping track_ids to LAB color information
            store_results: Whether to save results to disk
            
        Returns:
            Dictionary mapping track_ids to team labels
        """
        # Extract track IDs and LAB colors
        track_ids = list(avg_lab_colors.keys())
        lab_colors = np.array([avg_lab_colors[track_id].to_array() for track_id in track_ids])
        
        # Early exit if not enough data
        if len(track_ids) < 2:
            return {track_id: "Unknown" for track_id in track_ids}
        
        # Only use a* and b* channels for clustering (color without brightness)
        ab_colors = lab_colors[:, 1:]
        
        # Calculate min_samples as 25% of total tracks, minimum of 2
        min_samples = max(2, int(len(track_ids) * 0.25))
        
        # DBSCAN parameters
        eps = 50.0  # Start with large eps
        min_eps = 5.0  # Minimum eps threshold
        step = 5.0  # Step size for adjustment
        
        best_eps: Optional[float] = None
        best_labels: Optional[np.ndarray] = None
        num_clusters = 0  # Initialize for later use
        
        try:
            # Dynamically adjust eps to find exactly 2 main clusters
            while eps >= min_eps:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(ab_colors)
                
                # Count non-outlier clusters
                unique_clusters = np.unique(labels)
                non_outlier_clusters = [c for c in unique_clusters if c != -1]
                num_clusters = len(non_outlier_clusters)
                
                if num_clusters == 2:
                    # Perfect! Two team clusters
                    best_eps = eps
                    best_labels = labels
                    break
                elif num_clusters > 2:
                    # Too many clusters, use this but stop
                    eps += step
                    best_labels = labels
                    best_eps = eps
                    break
                else:
                    # Too few clusters, continue reducing eps
                    eps -= step
            
            # If we couldn't find two clusters, fallback to K-means
            if best_labels is None:
                kmeans = KMeans(n_clusters=min(2, len(ab_colors)), n_init=10, random_state=42)
                best_labels = kmeans.fit_predict(ab_colors)
                best_eps = eps  # Use the last eps value
        
        except Exception as e:
            print(f"Clustering error: {e}")
            return {track_id: "Unknown" for track_id in track_ids}
            
        # Count samples in each non-outlier cluster
        cluster_counts: Dict[int, int] = Counter(
            label for label in best_labels if label != -1
        )
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Map cluster indices to team labels
        cluster_to_team: Dict[int, str] = {}
        
        # Assign teams based on cluster size
        for i, (cluster_idx, _) in enumerate(sorted_clusters):
            if i == 0:
                cluster_to_team[cluster_idx] = "TEAM A"
            elif i == 1:
                cluster_to_team[cluster_idx] = "TEAM B"
            else:
                cluster_to_team[cluster_idx] = "REF/GK"
                
        # Handle outliers by assigning each to its own cluster
        outlier_cluster_map: Dict[int, int] = {}
        outlier_tracks: List[int] = []
        
        for i, label in enumerate(best_labels):
            if label == -1:
                outlier_tracks.append(track_ids[i])
        
        # Start new cluster IDs after the highest existing one
        max_cluster_id = max(np.max(best_labels), -1)
        
        for i, track_id in enumerate(outlier_tracks):
            new_cluster_id = max_cluster_id + 1 + i
            outlier_cluster_map[track_id] = new_cluster_id
            cluster_to_team[new_cluster_id] = "REF/GK"

        # Calculate and store cluster centers
        self.cluster_centers = {}
        
        # Handle non-outlier clusters
        for cluster_idx in np.unique(best_labels):
            if cluster_idx == -1:
                continue
                
            # Get points in this cluster
            mask = best_labels == cluster_idx
            cluster_lab_points = lab_colors[mask]
            
            # Calculate mean LAB color and convert to RGB
            center_lab = np.mean(cluster_lab_points, axis=0)
            center_rgb = lab_to_rgb_255(LABColor.from_array(center_lab))
            
            # Store center colors
            self.cluster_centers[cluster_idx] = {
                "lab": center_lab,
                "rgb": center_rgb,
                "team_label": cluster_to_team.get(cluster_idx, "Unknown")
            }
        
        # Handle outliers (each as its own cluster)
        for i, track_id in enumerate(outlier_tracks):
            cluster_idx = outlier_cluster_map[track_id]
            lab_color = avg_lab_colors[track_id]
            rgb_color = lab_to_rgb_255(lab_color)
            
            self.cluster_centers[cluster_idx] = {
                "lab": lab_color,
                "rgb": rgb_color,
                "team_label": cluster_to_team.get(cluster_idx, "REF/GK")
            }
        
        # Assign team labels to each track
        track_labels: Dict[int, str] = {}
        track_clusters: Dict[int, int] = {}
        
        for i, track_id in enumerate(track_ids):
            original_cluster_idx = best_labels[i]
            
            if original_cluster_idx != -1:
                cluster_idx = original_cluster_idx
            else:
                cluster_idx = outlier_cluster_map[track_id]
            
            team_label = cluster_to_team.get(cluster_idx, "Unknown")
            track_labels[track_id] = team_label
            track_clusters[track_id] = int(cluster_idx)
        
        # Visualize and save results if requested
        if store_results:
            output_dir = "role_assigner_results"
            os.makedirs(output_dir, exist_ok=True)

            try:
                self._visualize_dbscan_clusters(
                    lab_colors, 
                    best_labels, 
                    best_eps or eps, 
                    min_samples, 
                    output_dir, 
                    outlier_cluster_map
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Visualization error: {e}")
        
            try:
                with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
                    json.dump(track_labels, f, indent=2)
            except Exception as e:
                print(f"Error saving labels: {e}")
        
        return track_labels
    
    def _visualize_dbscan_clusters(
        self, 
        lab_colors: np.ndarray,
        labels: np.ndarray, 
        eps: float, 
        min_samples: int, 
        output_dir: str, 
        outlier_cluster_map: Dict[int, int]
    ) -> None:
        """Visualize DBSCAN clustering results."""
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        
        # Get unique clusters
        unique_clusters = np.unique(labels)
        
        # Plot non-outlier clusters
        for cluster_idx in unique_clusters:
            if cluster_idx == -1:
                continue
                
            # Get points in this cluster
            mask = labels == cluster_idx
            cluster_points = lab_colors[mask]
            
            # Get cluster information
            if cluster_idx in self.cluster_centers:
                cluster_info = self.cluster_centers[cluster_idx]
                team_label = cluster_info["team_label"]
                center_rgb: RGBColor255 = cluster_info["rgb"]
                # Normalize RGB for matplotlib
                color = tuple(c/255 for c in center_rgb.to_array())
            else:
                team_label = "Unknown Cluster"
                color = (0.5, 0.5, 0.5)  # Gray
            
            # Plot points (a* and b* channels)
            ax.scatter(
                cluster_points[:, 1],  # a* channel
                cluster_points[:, 2],  # b* channel
                color=color,
                marker='o',
                s=100,
                label=team_label
            )
            
            # Plot cluster center
            center = self.cluster_centers[cluster_idx]["lab"]
            ax.scatter(
                center[1],  # a* channel
                center[2],  # b* channel
                color=color,
                marker='*',
                s=300,
                edgecolors='black'
            )
            
            # Add text label
            ax.text(
                center[1],
                center[2],
                team_label,
                fontsize=12,
                weight='bold'
            )
        
        # Plot outliers
        outlier_mask = labels == -1
        if np.any(outlier_mask):
            outlier_indices = np.where(outlier_mask)[0]
            outlier_track_ids = list(outlier_cluster_map.keys())
            
            for i, track_id in enumerate(outlier_track_ids):
                if i >= len(outlier_indices):
                    continue
                    
                new_cluster_id = outlier_cluster_map[track_id]
                
                if new_cluster_id not in self.cluster_centers:
                    continue
                    
                cluster_info = self.cluster_centers[new_cluster_id]
                team_label = cluster_info["team_label"]
                center_rgb: RGBColor255 = cluster_info["rgb"]
                center_lab = cluster_info["lab"]
                
                color = tuple(c/255 for c in center_rgb.to_array())
                
                # Plot outlier point
                idx = outlier_indices[i]
                point = LABColor.from_array(lab_colors[idx])
                
                ax.scatter(
                    point.a,  # a* channel
                    point.b,  # b* channel
                    color=color,
                    marker='x',
                    s=100,
                    label=f"{team_label} (Outlier)"
                )
                
                # Plot center
                if hasattr(center_lab, 'a') and hasattr(center_lab, 'b'):
                    # If center_lab is a LABColor instance
                    a, b = center_lab.a, center_lab.b
                else:
                    # If center_lab is a numpy array
                    a, b = center_lab[1], center_lab[2]
                    
                ax.scatter(
                    a,  # a* channel
                    b,  # b* channel
                    color=color,
                    marker='*',
                    s=300,
                    edgecolors='black'
                )
                
                # Add text label
                ax.text(
                    a,
                    b,
                    team_label,
                    fontsize=12,
                    weight='bold'
                )
        
        # Add labels and title
        ax.set_xlabel('a* (Green-Red)')
        ax.set_ylabel('b* (Blue-Yellow)')
        ax.set_title(f'Team Color Clustering: eps={eps:.1f}, min_samples={min_samples}')
        
        # Add legend with unique labels only
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:  # Only add legend if there are labels
            ax.legend(by_label.values(), by_label.keys())
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'dbscan_clusters.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def assign_roles(self, detections: List[FrameDetections], store_results: bool = True) -> List[TrackRole]:
        """Assign team roles to tracks based on jersey colors.
        
        Args:
            detections: List of frame detections
            
        Returns:
            List of track role assignments
        """
        # Get colors for each track
        track_colors = self._get_track_colors(detections)
        
        # Calculate average LAB color for each track
        avg_lab_colors: Dict[int, LABColor] = {}
        role_assignments: List[TrackRole] = []
        
        # Process colors and handle tracks with no colors
        for track_id, colors in track_colors.items():
            if not colors:
                role_assignments.append(TrackRole(track_id=track_id, role="UNK"))
                continue
            
            avg_lab_colors[track_id] = self._get_avg_lab_color(colors)
        
        # Skip clustering if no tracks with colors
        if not avg_lab_colors:
            return role_assignments
            
        # Cluster tracks and assign team labels
        track_labels = self.cluster_tracks_and_assign_labels(avg_lab_colors, store_results)
        
        # Create role assignments from labels
        for track_id, label in track_labels.items():
            role_assignments.append(TrackRole(track_id=track_id, role=label))
            
        # Make sure all tracks have a role assignment
        assigned_track_ids = {role.track_id for role in role_assignments}
        for track_id in track_colors:
            if track_id not in assigned_track_ids:
                role_assignments.append(TrackRole(track_id=track_id, role="UNK"))
                
        return role_assignments
        