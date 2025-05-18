from custom_types import FrameDetections, BBox, TrackRole
from conversions import RGBColor255, LABColor, rgb_to_lab, lab_to_rgb_255
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Set, Any, Mapping
import json
from collections import defaultdict, Counter
from pydantic import BaseModel
import math
from sklearn.mixture import GaussianMixture


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
    
    class ClusteringResult(BaseModel):
        team_a_track_ids: List[int]
        team_b_track_ids: List[int]
        team_a_center: LABColor | None
        team_b_center: LABColor | None
        outlier_track_ids: List[int]
        params: Dict[str, Any]

    def _determine_optimal_clusters(self, data, max_clusters=7):
        """Determine optimal number of clusters using elbow method."""
        distortions = []
        K = range(1, max_clusters+1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        
        # Calculate rate of decrease (approximate second derivative)
        decreases = []
        for i in range(len(distortions)-1):
            decreases.append(distortions[i] - distortions[i+1])
        
        # Find elbow point - look for significant drop in improvement
        elbow_point = 1  # Default
        for i in range(len(decreases)-1):
            if decreases[i] / decreases[i+1] > 1.5:  # Significant drop-off
                elbow_point = i + 1
                break
        
        # Ensure result is between 2-5
        optimal_clusters = max(2, min(5, elbow_point + 1))
        return optimal_clusters

    def _cluster_tracks_dbscan(self, input_path: str, avg_lab_colors: Dict[int, LABColor], store_results: bool = True) -> ClusteringResult:
        """Cluster track colors using DBSCAN with dynamic eps adjustment to find teams and outliers."""
        track_ids = list(avg_lab_colors.keys())
        lab_colors = np.array([avg_lab_colors[track_id].to_array() for track_id in track_ids])

        # First determine optimal number of clusters using elbow method
        optimal_clusters = self._determine_optimal_clusters(lab_colors)
        
        # DBSCAN parameters
        min_samples = max([3, math.floor(0.25 * len(track_ids))])
        min_eps = 0.1
        max_eps = 1000.0
        
        # Binary search to find the largest eps that yields optimal_clusters
        best_labels = None
        
        while max_eps - min_eps > 0.1:  # Convergence threshold
            current_eps = (min_eps + max_eps) / 2
            dbscan = DBSCAN(eps=current_eps, min_samples=min_samples)
            labels = dbscan.fit_predict(lab_colors)
            
            unique_clusters = np.unique(labels)
            non_outlier_clusters = [c for c in unique_clusters if c != -1]
            num_clusters = len(non_outlier_clusters)
            
            if num_clusters == optimal_clusters:
                # Found target clusters, try larger eps (save current result)
                best_labels = labels
                min_eps = current_eps
                params = {"dbscan": {"eps": current_eps, "min_samples": min_samples, "target_clusters": optimal_clusters}}
            elif num_clusters < optimal_clusters:
                # Too few clusters, need smaller eps
                max_eps = current_eps
            else:
                # Too many clusters, need larger eps
                min_eps = current_eps
        
        # If binary search didn't find a solution, use k-means
        if best_labels is None:
            kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
            best_labels = kmeans.fit_predict(lab_colors)
            params = {"kmeans": {"n_clusters": optimal_clusters, "n_init": 10}}
        
        # Process clustering results - separate into teams and outliers
        # First, count members in each cluster
        cluster_sizes = Counter(best_labels)
        
        # Remove -1 (DBSCAN outliers) from consideration for largest clusters
        if -1 in cluster_sizes:
            del cluster_sizes[-1]
        
        # Find the two largest clusters
        largest_clusters = [cluster for cluster, _ in cluster_sizes.most_common(2)]
        
        # If fewer than 2 clusters found, handle edge case
        if len(largest_clusters) < 2:
            largest_clusters = largest_clusters + list(range(max(largest_clusters) + 1, max(largest_clusters) + 3 - len(largest_clusters)))
        
        team_a_tracks: List[int] = []
        team_b_tracks: List[int] = []
        outlier_tracks: List[int] = []
        
        # Assign tracks to teams or outliers
        for i, track_id in enumerate(track_ids):
            label = best_labels[i]
            if label == -1 or label not in largest_clusters:
                outlier_tracks.append(track_id)
            elif label == largest_clusters[0]:
                team_a_tracks.append(track_id)
            elif label == largest_clusters[1]:
                team_b_tracks.append(track_id)
        
        # Calculate team centers        
        if team_a_tracks:
            team_a_lab_points = np.array([avg_lab_colors[track_id].to_array() for track_id in team_a_tracks])
            team_a_center = LABColor.from_array(np.mean(team_a_lab_points, axis=0))
        else:
            team_a_center = None
            
        if team_b_tracks:
            team_b_lab_points = np.array([avg_lab_colors[track_id].to_array() for track_id in team_b_tracks])
            team_b_center = LABColor.from_array(np.mean(team_b_lab_points, axis=0))
        else:
            team_b_center = None
        
        # Return ClusteringResult and plot if store_results is True
        clustering_result = self.ClusteringResult(
            team_a_track_ids=team_a_tracks,
            team_b_track_ids=team_b_tracks,
            team_a_center=team_a_center,
            team_b_center=team_b_center,
            outlier_track_ids=outlier_tracks,
            params=params
        )
        
        if store_results:
            self._plot_clustering_results(input_path, clustering_result, avg_lab_colors)
            
        return clustering_result
    
    def _plot_clustering_results(self, input_path: str, clustering_result: ClusteringResult, avg_lab_colors: Dict[int, LABColor]) -> None:
        """Plot clustering results in LAB color space."""
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        
        # Get team colors
        team_a_color = None
        team_b_color = None
        
        # Plot Team A points and center
        if clustering_result.team_a_center:
            team_a_color = lab_to_rgb_255(clustering_result.team_a_center)
            team_a_norm_color = tuple(c/255 for c in team_a_color.to_array())
            
            # Plot Team A points if available
            if avg_lab_colors:
                for track_id in clustering_result.team_a_track_ids:
                    if track_id in avg_lab_colors:
                        lab_color = avg_lab_colors[track_id]
                        norm_color = tuple(c/255 for c in lab_to_rgb_255(lab_color).to_array())
                        ax.scatter(
                            lab_color.a,
                            lab_color.b,
                            color=norm_color,
                            marker='o',
                            s=100
                        )
            
            # Plot team A center
            ax.scatter(
                clustering_result.team_a_center.a,
                clustering_result.team_a_center.b,
                color=team_a_norm_color,
                marker='*',
                s=300,
                edgecolors='black'
            )
            
            # Add text label
            ax.text(
                clustering_result.team_a_center.a,
                clustering_result.team_a_center.b,
                "Team A",
                fontsize=12,
                weight='bold'
            )
        
        # Plot Team B points and center
        if clustering_result.team_b_center:
            team_b_color = lab_to_rgb_255(clustering_result.team_b_center)
            team_b_norm_color = tuple(c/255 for c in team_b_color.to_array())
            
            # Plot Team B points if available
            if avg_lab_colors:
                for track_id in clustering_result.team_b_track_ids:
                    if track_id in avg_lab_colors:
                        lab_color = avg_lab_colors[track_id]
                        norm_color = tuple(c/255 for c in lab_to_rgb_255(lab_color).to_array())
                        ax.scatter(
                            lab_color.a,
                            lab_color.b,
                            color=norm_color,
                            marker='s',
                            s=100
                        )
            
            # Plot team B center
            ax.scatter(
                clustering_result.team_b_center.a,
                clustering_result.team_b_center.b,
                color=team_b_norm_color,
                marker='*',
                s=300,
                edgecolors='black'
            )
            
            # Add text label
            ax.text(
                clustering_result.team_b_center.a,
                clustering_result.team_b_center.b,
                "Team B",
                fontsize=12,
                weight='bold'
            )
        
        # Plot outliers if available
        if avg_lab_colors:
            for track_id in clustering_result.outlier_track_ids:
                if track_id in avg_lab_colors:
                    lab_color = avg_lab_colors[track_id]
                    norm_color = tuple(c/255 for c in lab_to_rgb_255(lab_color).to_array())
                    ax.scatter(
                        lab_color.a,
                        lab_color.b,
                        color=norm_color,
                        marker='x',
                        s=100
                    )
        
        # Add labels and title
        ax.set_xlabel('a* (Green-Red)')
        ax.set_ylabel('b* (Blue-Yellow)')
        clustering_technique = list(clustering_result.params.keys())[0]
        params = [f"{key} = {str(value)}" for key, value in clustering_result.params[clustering_technique].items()]
        ax.set_title(f'Clustering Technique: {clustering_technique}. Params: {", ".join(params)}')
        
        # Create legend
        legend_elements = []
        if team_a_color:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color="w", markerfacecolor=team_a_norm_color, markersize=10, label='Team A'))
        if team_b_color:
            legend_elements.append(plt.Line2D([0], [0], marker='s', color="w", markerfacecolor=team_b_norm_color, markersize=10, label='Team B'))
        if clustering_result.outlier_track_ids:
            legend_elements.append(plt.Line2D([0], [0], marker='x', color='w', markeredgecolor='black', markersize=10, label='Outliers (Ref/GK)'))
        
        if legend_elements:
            ax.legend(handles=legend_elements)
        
        # Create output directory if it doesn't exist
        output_dir = 'role_assigner_results/' + os.path.basename(input_path).split('.')[0]
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot
        plt.savefig(os.path.join(output_dir, 'team_clusters.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _distinguish_outliers(self, frame_detections: FrameDetections, outliers: List[int]) -> Dict[int, str]:
        """Decide if an outlier is a goalkeeper or a referee."""
        track_ids_in_frame = [detection.track_id for detection in frame_detections.detections]
        outliers_in_frame = [track_id for track_id in outliers if track_id in track_ids_in_frame]

        # No REFs or GKs in frame
        if len(outliers_in_frame) == 0:
            return {}
        
        # 1 REF or GK in frame, use minimap position
        elif len(outliers_in_frame) == 1:
            track_id = outliers_in_frame[0]
            detection = next(detection for detection in frame_detections.detections if detection.track_id == track_id)
            if detection.minimap_coordinates is None:
                return {track_id: "OOB"}
            x, y, x_max, y_max = detection.minimap_coordinates.x, detection.minimap_coordinates.y, detection.minimap_coordinates.x_max, detection.minimap_coordinates.y_max

            scale_x = x_max / 105 # 105m is the length
            scale_y = y_max / 68 # 68m is the width
            scale = (scale_x + scale_y) / 2 # average scale

            goal_width = 7.32

            middle_y = x_max / 2
            lower_box_boundary_y = middle_y - (goal_width / 2) * scale - 16.5 * scale
            upper_box_boundary_y = middle_y + (goal_width / 2) * scale + 16.5 * scale

            left_box_boundary_x = 0 + 16.5 * scale
            right_box_boundary_x = x_max - 16.5 * scale

            # Check if the outlier is inside either of the two boxes
            if y > lower_box_boundary_y and y < upper_box_boundary_y:
                if x < left_box_boundary_x or x > right_box_boundary_x:
                    return {track_id: "GK"}
                return {track_id: "REF"}
            else:
                return {track_id: "REF"}
        
        # Multiple outliers, REF is closest to the center of the pitch
        elif len(outliers_in_frame) > 1:
            return_value = {}
            outlier_detections = [detection for detection in frame_detections.detections if detection.track_id in outliers_in_frame]
            outlier_coordinates = {detection.track_id: detection.minimap_coordinates for detection in outlier_detections}

            ref_track_id = None
            closest_squared_distance = float('inf')
            for track_id, coordinates in outlier_coordinates.items():
                if coordinates is None:
                    return_value[track_id] = "OOB"
                    continue
                squared_distance = (coordinates.x - coordinates.x_max / 2) ** 2 + (coordinates.y - coordinates.y_max / 2) ** 2
                if squared_distance < closest_squared_distance:
                    closest_squared_distance = squared_distance
                    ref_track_id = track_id
            
            if ref_track_id:
                return_value[ref_track_id] = "REF"
            for track_id in outliers_in_frame:
                if track_id not in return_value:
                    return_value[track_id] = "GK"

            return return_value
            

    def assign_roles(self, input_path: str, detections: List[FrameDetections], store_results: bool = True) -> List[TrackRole]:
        """Assign team roles to tracks based on jersey colors.
        
        Args:
            detections: List of frame detections
            
        Returns:
            List of track role assignments
        """
        role_assignments: List[TrackRole] = []

        # Get colors for each track
        track_colors = self._get_track_colors(detections)
        
        # Calculate average LAB color for each track
        avg_lab_colors: Dict[int, LABColor] = {}
        for track_id, colors in track_colors.items():
            if not colors:
                role_assignments.append(TrackRole(track_id=track_id, role="UNK"))
                continue
            
            avg_lab_colors[track_id] = self._get_avg_lab_color(colors)
        
        # Skip clustering if no tracks with colors
        if not avg_lab_colors:
            return role_assignments
            
        # Cluster tracks and assign team labels
        clustering_result = self._cluster_tracks_dbscan(input_path, avg_lab_colors, store_results)
        
        # Create role assignments from clustering results
        for track_id in clustering_result.team_a_track_ids:
            role_assignments.append(TrackRole(track_id=track_id, role="TEAM A"))
            
        for track_id in clustering_result.team_b_track_ids:
            role_assignments.append(TrackRole(track_id=track_id, role="TEAM B"))

        # HANDLE REF/GK OUTLIERS
        outlier_assignments = {track_id: [] for track_id in clustering_result.outlier_track_ids}
        for frame_detections in detections:
            outlier_results = self._distinguish_outliers(frame_detections, clustering_result.outlier_track_ids)
            for track_id, role in outlier_results.items():
                outlier_assignments[track_id].append(role)
        
        majority_outlier_assignments = {track_id: max(roles, key=roles.count) for track_id, roles in outlier_assignments.items()}
        for track_id, role in majority_outlier_assignments.items():
            role_assignments.append(TrackRole(track_id=track_id, role=role))
            
        # Make sure all tracks have a role assignment
        assigned_track_ids = {role.track_id for role in role_assignments}
        for track_id in track_colors:
            if track_id not in assigned_track_ids:
                role_assignments.append(TrackRole(track_id=track_id, role="UNK"))

        if store_results:
            output_dir = 'role_assigner_results/' + os.path.basename(input_path).split('.')[0]
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "role_assignments.json"), "w") as f:
                json.dump([track_role.model_dump() for track_role in role_assignments], f, indent=4)

        for frame_detections in detections:
            for detection in frame_detections.detections:
                role_assignment = next((role for role in role_assignments if role.track_id == detection.track_id), None)
                if role_assignment is None:
                    msg = f"No role assignment found for track ID {detection.track_id}"
                    raise ValueError(msg)
                detection.role = role_assignment.role

        if store_results:
            output_dir = 'role_assigner_results/' + os.path.basename(input_path).split('.')[0]
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "detections.json"), "w") as f:
                json.dump([frame_detections.model_dump() for frame_detections in detections], f, indent=4)
        
        return detections
        