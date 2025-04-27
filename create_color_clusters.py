import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
from skimage import color
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def create_color_clusters_basic(all_player_colors, output_plot_path):
    # Extract RGB colors and confidences
    rgb_colors = np.array([color['dominant_color_rgb'] for color in all_player_colors])
    confidences = np.array([color['confidence'] for color in all_player_colors])

    # Normalize RGB values to 0-1 range if needed
    rgb_colors = rgb_colors / 255.0 if np.any(rgb_colors > 1.0) else rgb_colors

    # Perform k-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(rgb_colors)

    # Get cluster sizes and sort indices by size (descending)
    cluster_sizes = np.bincount(cluster_labels)
    cluster_size_order = np.argsort(-cluster_sizes)

    # Map clusters to teams/referee based on size
    cluster_mapping = {
        cluster_size_order[0]: 'Team 1',
        cluster_size_order[1]: 'Team 2', 
        cluster_size_order[2]: 'Referee'
    }

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Plot each cluster with different colors and labels
    colors = ['blue', 'red', 'yellow']
    for i, cluster_idx in enumerate(cluster_size_order):
        mask = cluster_labels == cluster_idx
        plt.scatter(
            rgb_colors[mask, 0], 
            rgb_colors[mask, 1],
            c=colors[i],
            marker='o',
            s=100 * confidences[mask],
            label=f'{cluster_mapping[cluster_idx]} (n={cluster_sizes[cluster_idx]})'
        )

    plt.xlabel('Red')
    plt.ylabel('Green')
    plt.title('RGB Color Clusters of Players')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_plot_path)
    plt.close()

    # Create a figure to display average colors
    plt.figure(figsize=(10, 2))

    # Plot each cluster center as a solid color patch
    for i, cluster_idx in enumerate(cluster_size_order):
        center = kmeans.cluster_centers_[cluster_idx]
        plt.subplot(1, 3, i+1)
        plt.axis('off')
        plt.title(cluster_mapping[cluster_idx])
        plt.imshow([[center]])

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(output_plot_path)[0]}_centers.png")
    plt.close()

    return kmeans, cluster_mapping, cluster_size_order

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
    """Find optimal number of clusters using silhouette analysis."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    
    max_clusters = min(8, len(lab_colors) - 1)  # Can't have more clusters than samples
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(lab_colors, sample_weight=confidences)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(lab_colors, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}")
    
    # Plot silhouette analysis
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.grid(True)
    plt.savefig('cluster_evaluation.png')
    plt.close()
    
    # Return optimal number of clusters
    return np.argmax(silhouette_scores) + 2  # +2 because we started from 2