import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
from skimage import color
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

def create_color_clusters_lab_dynamic(all_player_colors, output_plot_path):
    # Extract RGB colors and confidences
    rgb_colors = np.array([color['dominant_color_rgb'] for color in all_player_colors])
    confidences = np.array([color['confidence'] for color in all_player_colors])

    # Normalize RGB values to 0-1 range if needed
    rgb_colors = rgb_colors / 255.0 if np.any(rgb_colors > 1.0) else rgb_colors
    
    # Convert RGB to LAB color space for better perceptual clustering
    lab_colors = color.rgb2lab(rgb_colors.reshape(1, -1, 3)).reshape(-1, 3)
    
    # Use elbow method to determine optimal number of clusters (between 1 and 5)
    max_clusters = min(5, len(lab_colors) - 1)  # Can't have more clusters than samples
    wcss_values = []
    silhouette_scores = []
    
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(lab_colors, sample_weight=confidences)
        
        # Calculate WCSS (Within-Cluster Sum of Squares)
        wcss_values.append(kmeans.inertia_)
        
        # Also calculate silhouette score for reference (only valid for n_clusters >= 2)
        if n_clusters >= 2:
            silhouette_avg = silhouette_score(lab_colors, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {n_clusters}, WCSS = {kmeans.inertia_:.2f}, silhouette score = {silhouette_avg:.3f}")
        else:
            print(f"For n_clusters = {n_clusters}, WCSS = {kmeans.inertia_:.2f}")
    
    # Plot elbow method and silhouette analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow method plot
    ax1.plot(range(1, max_clusters + 1), wcss_values, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS')
    ax1.set_title('Elbow Method For Optimal k')
    ax1.grid(True)
    
    # Silhouette score plot (for reference)
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis For Optimal k')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.splitext(output_plot_path)[0] + '_elbow_analysis.png')
    plt.close()
    
    # Find the optimal number of clusters using multiple metrics
    
    # 1. Use the geometric method for elbow detection
    if len(wcss_values) >= 3:
        # Normalize the values to 0-1 range for better elbow detection
        x = np.array(range(1, max_clusters + 1))
        x_norm = (x - min(x)) / (max(x) - min(x)) if max(x) > min(x) else x
        y = np.array(wcss_values)
        y_norm = (y - min(y)) / (max(y) - min(y)) if max(y) > min(y) else y
        
        # Calculate the distance from each point to the line connecting first and last points
        points = np.column_stack((x_norm, y_norm))
        line_vec = points[-1] - points[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = points - points[0]
        scalar_projection = np.sum(vec_from_first * np.tile(line_vec_norm, (points.shape[0], 1)), axis=1)
        vec_on_line = np.outer(scalar_projection, line_vec_norm)
        vec_to_line = vec_from_first - vec_on_line
        dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
        
        geometric_n_clusters = np.argmax(dist_to_line) + 1
    else:
        geometric_n_clusters = 2 if max_clusters >= 2 else 1
    
    # 2. Consider consecutive drops in WCSS
    if len(wcss_values) > 1:
        wcss_diffs = np.diff(wcss_values)
        percentage_drops = np.abs(wcss_diffs / np.array(wcss_values[:-1]))
        
        # Look for significant drops (more than 40% from previous value)
        significant_drops = np.where(percentage_drops > 0.4)[0] + 1
        drops_n_clusters = significant_drops[-1] + 1 if len(significant_drops) > 0 else 2
    else:
        drops_n_clusters = 2
    
    # 3. Consider silhouette scores for clusters ≥ 2
    if len(silhouette_scores) > 0:
        sil_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because silhouette starts at 2
    else:
        sil_n_clusters = 2
    
    # 4. Special rule for football: prefer 3 clusters (2 teams + referee)
    # We'll give a bonus to 3 clusters if it has a good silhouette score
    football_preference = 3
    
    print(f"Metrics - Geometric: {geometric_n_clusters}, Drops: {drops_n_clusters}, Silhouette: {sil_n_clusters}")
    
    # Combine metrics with a bias toward 3 clusters
    if football_preference <= max_clusters and (
        # If silhouette for k=3 is within 90% of the best silhouette score and better than k=2
        (football_preference == 3 and len(silhouette_scores) >= 2 and 
         silhouette_scores[1] >= 0.9 * max(silhouette_scores) and
         silhouette_scores[1] > silhouette_scores[0]) or
        # Or if both geometric and drops suggest k=3
        (geometric_n_clusters == football_preference and drops_n_clusters >= football_preference) or
        # Or if silhouette suggests k=3 and the drop from 2 to 3 is significant
        (sil_n_clusters == football_preference and len(wcss_values) >= 3 and 
         (wcss_values[1] - wcss_values[2]) / wcss_values[1] > 0.3)
    ):
        optimal_n_clusters = football_preference
        print(f"Selected {optimal_n_clusters} clusters based on football context preference")
    # Otherwise go with the geometric method, but prefer at least 2 clusters
    elif geometric_n_clusters >= 2:
        optimal_n_clusters = geometric_n_clusters
        print(f"Selected {optimal_n_clusters} clusters based on geometric method")
    # If all else fails, use the best silhouette score
    elif sil_n_clusters >= 2:
        optimal_n_clusters = sil_n_clusters
        print(f"Selected {optimal_n_clusters} clusters based on silhouette score")
    # Last resort: use 3 as a reasonable default for football
    else:
        optimal_n_clusters = min(3, max_clusters)
        print(f"No clear optimal clusters, defaulting to {optimal_n_clusters}")
    
    print(f"Optimal number of clusters based on combined metrics: {optimal_n_clusters}")
    
    # Perform k-means with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(lab_colors, sample_weight=confidences)
    
    # Calculate silhouette score for the optimal clustering (for reference)
    if optimal_n_clusters >= 2:
        sil_score = silhouette_score(lab_colors, cluster_labels)
        print(f"Final clustering silhouette score: {sil_score:.3f}")
    
    # Get cluster sizes and sort indices by size (descending)
    cluster_sizes = np.bincount(cluster_labels)
    cluster_size_order = np.argsort(-cluster_sizes)
    
    # Map clusters to teams/referee based on size
    cluster_mapping = {}
    for i, cluster_idx in enumerate(cluster_size_order):
        if i == 0:
            cluster_mapping[cluster_idx] = 'Team 1'
        elif i == 1:
            cluster_mapping[cluster_idx] = 'Team 2'
        elif i == 2:
            cluster_mapping[cluster_idx] = 'Referee/GK 1'
        elif i == 3:
            cluster_mapping[cluster_idx] = 'Referee/GK 2'
        elif i == 4:
            cluster_mapping[cluster_idx] = 'Referee/GK 3'
        else:
            raise ValueError(f"Invalid cluster index: {cluster_idx}")
    
    # Convert cluster centers back to RGB for visualization
    centers_lab = kmeans.cluster_centers_
    # Reshape for skimage conversion
    centers_lab_reshaped = centers_lab.reshape(1, -1, 3)
    centers_rgb = color.lab2rgb(centers_lab_reshaped).reshape(-1, 3)
    
    # Create a grid for multiple visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, optimal_n_clusters, height_ratios=[3, 1])
    
    # 1. 3D scatter plot
    ax1 = fig.add_subplot(gs[0, 0:optimal_n_clusters-1] if optimal_n_clusters > 1 else gs[0, 0], projection='3d')
    plot_colors = ['blue', 'red', 'yellow', 'green', 'purple'][:optimal_n_clusters]
    
    for i, cluster_idx in enumerate(cluster_size_order):
        mask = cluster_labels == cluster_idx
        ax1.scatter(
            lab_colors[mask, 0],
            lab_colors[mask, 1],
            lab_colors[mask, 2],
            c=plot_colors[i],
            marker='o',
            s=100 * confidences[mask],
            label=f'{cluster_mapping[cluster_idx]} (n={cluster_sizes[cluster_idx]})'
        )
    
    ax1.set_xlabel('L* (Lightness)')
    ax1.set_ylabel('a* (Green-Red)')
    ax1.set_zlabel('b* (Blue-Yellow)')
    ax1.set_title(f'LAB Color Space Clusters (k={optimal_n_clusters})')
    ax1.legend()
    
    # 2. 2D scatter plot of a* vs b* (color channels)
    if optimal_n_clusters > 1:
        ax2 = fig.add_subplot(gs[0, optimal_n_clusters-1])
        for i, cluster_idx in enumerate(cluster_size_order):
            mask = cluster_labels == cluster_idx
            ax2.scatter(
                lab_colors[mask, 1],
                lab_colors[mask, 2],
                c=plot_colors[i],
                marker='o',
                s=100 * confidences[mask],
                alpha=0.7,
                label=f'{cluster_mapping[cluster_idx]}'
            )
        
        ax2.set_xlabel('a* (Green-Red)')
        ax2.set_ylabel('b* (Blue-Yellow)')
        ax2.set_title('Color Distribution (a* vs b*)')
        ax2.grid(True)
    
    # 3. Display team colors
    for i, cluster_idx in enumerate(cluster_size_order):
        if i < optimal_n_clusters:  # Only display up to the number of clusters
            ax = fig.add_subplot(gs[1, i])
            rgb_color = centers_rgb[cluster_idx]
            ax.imshow([[rgb_color]])
            ax.set_title(f"{cluster_mapping[cluster_idx]}\nRGB: {rgb_color_to_hex(rgb_color)}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    
    # Create a report of color information
    color_info = {}
    for i, cluster_idx in enumerate(cluster_size_order):
        if i >= len(cluster_mapping):
            continue
            
        team_name = cluster_mapping[cluster_idx]
        rgb_color = centers_rgb[cluster_idx]
        lab_color = centers_lab[cluster_idx]
        hsv_color = color.rgb2hsv(rgb_color.reshape(1, 1, 3)).reshape(3)
        
        color_info[team_name] = {
            'rgb': rgb_color,
            'rgb_hex': rgb_color_to_hex(rgb_color),
            'lab': lab_color,
            'hsv': hsv_color,
            'count': cluster_sizes[cluster_idx],
            'confidence_avg': np.mean(confidences[cluster_labels == cluster_idx])
        }
    
    return kmeans, cluster_mapping, color_info

def create_color_clusters_lab(all_player_colors, output_plot_path):
    # Extract RGB colors and confidences
    rgb_colors = np.array([color['dominant_color_rgb'] for color in all_player_colors])
    confidences = np.array([color['confidence'] for color in all_player_colors])

    # Normalize RGB values to 0-1 range if needed
    rgb_colors = rgb_colors / 255.0 if np.any(rgb_colors > 1.0) else rgb_colors
    
    # Convert RGB to LAB color space for better perceptual clustering
    lab_colors = color.rgb2lab(rgb_colors.reshape(1, -1, 3)).reshape(-1, 3)
    
    # Perform weighted k-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(lab_colors, sample_weight=confidences)
    
    # Calculate silhouette score to evaluate clustering quality
    sil_score = silhouette_score(lab_colors, cluster_labels)
    print(f"Clustering silhouette score: {sil_score:.3f}")
    
    # Get cluster sizes and sort indices by size (descending)
    cluster_sizes = np.bincount(cluster_labels)
    cluster_size_order = np.argsort(-cluster_sizes)
    
    # Map clusters to teams/referee based on size
    cluster_mapping = {
        cluster_size_order[0]: 'Team 1',
        cluster_size_order[1]: 'Team 2', 
        cluster_size_order[2]: 'Referee'
    }
    
    # Convert cluster centers back to RGB for visualization
    centers_lab = kmeans.cluster_centers_
    # Reshape for skimage conversion
    centers_lab_reshaped = centers_lab.reshape(1, -1, 3)
    centers_rgb = color.lab2rgb(centers_lab_reshaped).reshape(-1, 3)
    
    # Create a grid for multiple visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
    
    # 1. 3D scatter plot
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    plot_colors = ['blue', 'red', 'yellow']
    
    for i, cluster_idx in enumerate(cluster_size_order):
        mask = cluster_labels == cluster_idx
        ax1.scatter(
            lab_colors[mask, 0],
            lab_colors[mask, 1],
            lab_colors[mask, 2],
            c=plot_colors[i],
            marker='o',
            s=100 * confidences[mask],
            label=f'{cluster_mapping[cluster_idx]} (n={cluster_sizes[cluster_idx]})'
        )
    
    ax1.set_xlabel('L* (Lightness)')
    ax1.set_ylabel('a* (Green-Red)')
    ax1.set_zlabel('b* (Blue-Yellow)')
    ax1.set_title('LAB Color Space Clusters')
    ax1.legend()
    
    # 2. 2D scatter plot of a* vs b* (color channels)
    ax2 = fig.add_subplot(gs[0, 2])
    for i, cluster_idx in enumerate(cluster_size_order):
        mask = cluster_labels == cluster_idx
        ax2.scatter(
            lab_colors[mask, 1],
            lab_colors[mask, 2],
            c=plot_colors[i],
            marker='o',
            s=100 * confidences[mask],
            alpha=0.7,
            label=f'{cluster_mapping[cluster_idx]}'
        )
    
    ax2.set_xlabel('a* (Green-Red)')
    ax2.set_ylabel('b* (Blue-Yellow)')
    ax2.set_title('Color Distribution (a* vs b*)')
    ax2.grid(True)
    
    # 3. Display team colors
    for i, cluster_idx in enumerate(cluster_size_order):
        ax = fig.add_subplot(gs[1, i])
        rgb_color = centers_rgb[cluster_idx]
        ax.imshow([[rgb_color]])
        ax.set_title(f"{cluster_mapping[cluster_idx]}\nRGB: {rgb_color_to_hex(rgb_color)}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    
    # Create a report of color information
    color_info = {}
    for cluster_idx in cluster_size_order:
        team_name = cluster_mapping[cluster_idx]
        rgb_color = centers_rgb[cluster_idx]
        lab_color = centers_lab[cluster_idx]
        hsv_color = color.rgb2hsv(rgb_color.reshape(1, 1, 3)).reshape(3)
        
        color_info[team_name] = {
            'rgb': rgb_color,
            'rgb_hex': rgb_color_to_hex(rgb_color),
            'lab': lab_color,
            'hsv': hsv_color,
            'count': cluster_sizes[cluster_idx],
            'confidence_avg': np.mean(confidences[cluster_labels == cluster_idx])
        }
    
    return kmeans, cluster_mapping, color_info

def rgb_color_to_hex(rgb_color):
    """Convert RGB color (0-1 range) to hex string."""
    rgb_int = (np.clip(rgb_color, 0, 1) * 255).astype(int)
    return f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}"

def evaluate_clustering_quality(lab_colors, confidences):
    """Find optimal number of clusters using comprehensive analysis of WCSS and silhouette scores."""
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import silhouette_score
    
    max_clusters = min(8, len(lab_colors) - 1)  # Can't have more clusters than samples
    wcss_values = []
    silhouette_scores = []
    
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(lab_colors, sample_weight=confidences)
        
        # Calculate WCSS (Within-Cluster Sum of Squares)
        wcss = kmeans.inertia_
        wcss_values.append(wcss)
        
        # Also calculate silhouette score for reference (only valid for n_clusters >= 2)
        if n_clusters >= 2:
            silhouette_avg = silhouette_score(lab_colors, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {n_clusters}, WCSS = {wcss:.2f}, silhouette score = {silhouette_avg:.3f}")
        else:
            print(f"For n_clusters = {n_clusters}, WCSS = {wcss:.2f}")
    
    # Plot WCSS analysis
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss_values, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.savefig('cluster_evaluation.png')
    plt.close()
    
    # Find the optimal number of clusters using multiple metrics
    
    # 1. Use the geometric method for elbow detection
    if len(wcss_values) >= 3:
        # Normalize the values to 0-1 range for better elbow detection
        x = np.array(range(1, max_clusters + 1))
        x_norm = (x - min(x)) / (max(x) - min(x)) if max(x) > min(x) else x
        y = np.array(wcss_values)
        y_norm = (y - min(y)) / (max(y) - min(y)) if max(y) > min(y) else y
        
        # Calculate the distance from each point to the line connecting first and last points
        points = np.column_stack((x_norm, y_norm))
        line_vec = points[-1] - points[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = points - points[0]
        scalar_projection = np.sum(vec_from_first * np.tile(line_vec_norm, (points.shape[0], 1)), axis=1)
        vec_on_line = np.outer(scalar_projection, line_vec_norm)
        vec_to_line = vec_from_first - vec_on_line
        dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
        
        geometric_n_clusters = np.argmax(dist_to_line) + 1
    else:
        geometric_n_clusters = 2 if max_clusters >= 2 else 1
    
    # 2. Consider consecutive drops in WCSS
    if len(wcss_values) > 1:
        wcss_diffs = np.diff(wcss_values)
        percentage_drops = np.abs(wcss_diffs / np.array(wcss_values[:-1]))
        
        # Look for significant drops (more than 40% from previous value)
        significant_drops = np.where(percentage_drops > 0.4)[0] + 1
        drops_n_clusters = significant_drops[-1] + 1 if len(significant_drops) > 0 else 2
    else:
        drops_n_clusters = 2
    
    # 3. Consider silhouette scores for clusters ≥ 2
    if len(silhouette_scores) > 0:
        sil_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because silhouette starts at 2
    else:
        sil_n_clusters = 2
    
    # 4. Special rule for football: prefer 3 clusters (2 teams + referee)
    # We'll give a bonus to 3 clusters if it has a good silhouette score
    football_preference = 3
    
    print(f"Metrics - Geometric: {geometric_n_clusters}, Drops: {drops_n_clusters}, Silhouette: {sil_n_clusters}")
    
    # Combine metrics with a bias toward 3 clusters
    if football_preference <= max_clusters and (
        # If silhouette for k=3 is within 90% of the best silhouette score and better than k=2
        (football_preference == 3 and len(silhouette_scores) >= 2 and 
         silhouette_scores[1] >= 0.9 * max(silhouette_scores) and
         silhouette_scores[1] > silhouette_scores[0]) or
        # Or if both geometric and drops suggest k=3
        (geometric_n_clusters == football_preference and drops_n_clusters >= football_preference) or
        # Or if silhouette suggests k=3 and the drop from 2 to 3 is significant
        (sil_n_clusters == football_preference and len(wcss_values) >= 3 and 
         (wcss_values[1] - wcss_values[2]) / wcss_values[1] > 0.3)
    ):
        elbow_point = football_preference
        print(f"Selected {elbow_point} clusters based on football context preference")
    # Otherwise go with the geometric method, but prefer at least 2 clusters
    elif geometric_n_clusters >= 2:
        elbow_point = geometric_n_clusters
        print(f"Selected {elbow_point} clusters based on geometric method")
    # If all else fails, use the best silhouette score
    elif sil_n_clusters >= 2:
        elbow_point = sil_n_clusters
        print(f"Selected {elbow_point} clusters based on silhouette score")
    # Last resort: use 3 as a reasonable default for football
    else:
        elbow_point = min(3, max_clusters)
        print(f"No clear optimal clusters, defaulting to {elbow_point}")
    
    print(f"Optimal number of clusters based on combined metrics: {elbow_point}")
    
    return elbow_point