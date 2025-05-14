from custom_types import FrameDetections, RGBColor, BBox, TrackRole
from skimage import color
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class RoleAssigner:
    def __init__(self):
        pass

    def _roi_is_overlapping(self, roi_bbox: BBox, other_roi_bboxes: list[BBox]) -> bool:
        for other_roi_bbox in other_roi_bboxes:
            if roi_bbox.x1 < other_roi_bbox.x2 and roi_bbox.x2 > other_roi_bbox.x1 and roi_bbox.y1 < other_roi_bbox.y2 and roi_bbox.y2 > other_roi_bbox.y1:
                return True
        return False
    
    def _get_all_track_ids(self, detections: list[FrameDetections]) -> list[int]:
        return list(set([detection.track_id for frame_detections in detections for detection in frame_detections.detections]))
    
    def _list_all_colors(self, track_id: int, detections: list[FrameDetections], allow_overlapping: bool) -> list[RGBColor]:
        track_colors = []
        for frame_detections in detections:
            for detection in frame_detections.detections:
                if detection.track_id == track_id:
                    if not allow_overlapping:
                        other_roi_bboxes = [d.roi_bbox for d in frame_detections.detections if d.track_id != track_id]
                        if self._roi_is_overlapping(detection.roi_bbox, other_roi_bboxes):
                            continue
                    track_colors.append(detection.jersey_color)
        return track_colors

    def _get_track_colors(self, detections: list[FrameDetections]) -> dict[int, list[RGBColor]]:
        track_ids = self._get_all_track_ids(detections)
        track_colors = {}
        for track_id in track_ids:
            track_colors[track_id] = self._list_all_colors(
                track_id, detections, allow_overlapping=False
            )
        
        # If any track has no colors, try again but allow overlapping ROIs
        if any(len(colors) == 0 for colors in track_colors.values()):
            for track_id in track_ids:
                if len(track_colors[track_id]) == 0:
                    track_colors[track_id] = self._list_all_colors(
                        track_id, detections, allow_overlapping=True
                    )

        return track_colors
    
    def _get_avg_lab_color(self, colors: list[RGBColor]) -> np.ndarray:
        if len(colors) == 0:
            raise ValueError("No colors provided")

        lab_colors = []
        for current_color in colors:
            rgb_normalized = np.array([[current_color.r, current_color.g, current_color.b]]).astype(np.float32) / 255.0
            lab_color = color.rgb2lab(rgb_normalized).reshape(3)
            lab_colors.append(lab_color)
        return np.mean(lab_colors, axis=0)

    def cluster_tracks_and_assign_labels(self, avg_lab_colors: dict[int, np.ndarray], store_results: bool = True) -> dict[int, str]:
        """Cluster track colors using DBSCAN with dynamic eps adjustment to find teams and outliers.
        
        Args:
            avg_track_colors: Dictionary mapping track_ids to color information
            store_results: Whether to save results to disk
            
        Returns:
            Dictionary mapping track_ids to team labels
        """
        if not avg_lab_colors:
            return {}
        
        # Extract track IDs and LAB colors
        track_ids = list(avg_lab_colors.keys())
        lab_colors = np.array([avg_lab_colors[track_id] for track_id in track_ids])
        
        # Only use a* and b* channels for clustering
        ab_colors = lab_colors[:, 1:]  # Take only a* and b* channels
        
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
                labels = dbscan.fit_predict(ab_colors)
                
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
                    kmeans = KMeans(n_clusters=min(2, len(ab_colors)), n_init=10, random_state=42)
                    best_labels = kmeans.fit_predict(ab_colors)
                    # All points assigned to clusters in K-means (no -1 labels)
        
        except Exception as e:
            print(f"Clustering error: {e}")
            track_labels = {track_id: "Unknown" for track_id in track_ids}
            return track_labels
            
        # Count samples in each non-outlier cluster (-1 is outliers)
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
        for i, (cluster_idx, _) in enumerate(sorted_clusters):
            if i == 0:
                cluster_to_team[cluster_idx] = "TEAM A"
            elif i == 1:
                cluster_to_team[cluster_idx] = "TEAM B"
            else:
                # Any additional non-outlier clusters beyond the top 2 are REF/GK
                cluster_to_team[cluster_idx] = "REF/GK"
                
        # Handle outliers (-1 label)
        # We'll group outliers by assigning a new cluster ID to each outlier
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
            # Assign REF/GK label to all outliers
            cluster_to_team[new_cluster_id] = "REF/GK"

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
            lab_color = avg_lab_colors[track_id]
            rgb_color = self._lab_to_rgb(lab_color)
            
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
        if store_results:
            output_dir = "role_assigner_results"
            os.makedirs(output_dir, exist_ok=True)

            try:
                self._visualize_dbscan_clusters(lab_colors, best_labels, best_eps or eps, min_samples, output_dir, outlier_cluster_map)
            except Exception as e:
                print(f"Visualization error: {e}")
        
            try:
                import json
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
        
        # Create a 2D plot
        ax = plt.subplot(111)
        
        # Get unique non-outlier clusters
        unique_clusters = np.unique(labels)
        
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
            
            # Plot the points (only a* and b* channels)
            ax.scatter(
                cluster_points[:, 1],  # a* channel
                cluster_points[:, 2],  # b* channel
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
                color=color,
                marker='*',
                s=300,
                edgecolors='black'
            )
            
            # Add text label at the center
            ax.text(
                center[1],
                center[2],
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
            outlier_track_ids = list(outlier_cluster_map.keys())
            
            # For each outlier, plot it with its assigned color and label
            for i, track_id in enumerate(outlier_track_ids):
                if i >= len(outlier_indices):
                    continue
                    
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
                idx = outlier_indices[i]
                point = lab_colors[idx]
                
                ax.scatter(
                    point[1],  # a* channel
                    point[2],  # b* channel
                    color=color,
                    marker='x',
                    s=100,
                    label=team_label
                )
                
                # Plot the center (which is the same as the point for outliers)
                ax.scatter(
                    center_lab[1],  # a* channel
                    center_lab[2],  # b* channel
                    color=color,
                    marker='*',
                    s=300,
                    edgecolors='black'
                )
                
                # Add text label
                ax.text(
                    center_lab[1],
                    center_lab[2],
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
        ax.legend(by_label.values(), by_label.keys())
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'dbscan_clusters.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _lab_to_rgb(self, lab: np.ndarray) -> Tuple[int, int, int]:
        lab_reshaped = lab.reshape(1, 1, 3)
        rgb = color.lab2rgb(lab_reshaped)
        rgb_255 = (rgb.reshape(3) * 255).astype(int)
        return tuple(rgb_255)

    def assign_roles(self, detections: list[FrameDetections], store_results: bool = True) -> list[TrackRole]:
        track_colors = self._get_track_colors(detections)

        avg_lab_colors: dict[int, np.ndarray] = {}
        role_assignments: list[TrackRole] = []
        
        for track_id, colors in track_colors.items():
            if len(colors) == 0:
                role_assignments.append(TrackRole(track_id=track_id, role="UNK"))
                continue

            avg_lab_colors[track_id] = self._get_avg_lab_color(colors)
            
        # Cluster the tracks and assign team labels
        track_labels = self.cluster_tracks_and_assign_labels(avg_lab_colors, store_results=store_results)
        
        # Create TrackRole objects based on the labels
        for track_id, label in track_labels.items():
            role_assignments.append(TrackRole(track_id=track_id, role=label))
            
        # Add any tracks that didn't get a label (should be handled by clustering already, but just in case)
        assigned_track_ids = {role.track_id for role in role_assignments}
        for track_id in avg_lab_colors.keys():
            if track_id not in assigned_track_ids:
                role_assignments.append(TrackRole(track_id=track_id, role="UNK"))
                
        return role_assignments
        