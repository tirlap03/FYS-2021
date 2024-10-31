import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import permutations



data = np.loadtxt('frey-faces.csv')         # Every row is an image with 20x28 pixels and there is 1965 images
print(f"Data shape: {data.shape}")

"""
Displaying a single face image on a given matplotlib axis
Arguments:
    image_Array: 1D array of pixel values for a single face
    ax: matplotlib axis to plot on
    title: title of the subplot
"""
def face_display(ax, image_array, title):
    face_image = image_array.reshape(28, 20)
    ax.imshow(face_image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

"""
Display a grid of random faces from the dataset
Arguments:
    data: array of face data where each row is a flattened face image
    n_faces: number of faces to display grid
    n_rows: number of rows in the display grid
"""
def random_display(data, n_faces=9, n_rows=3):
    n_cols = n_faces // n_rows
    plt.figure(figsize=(12,12))

    for i in range(n_faces):
        ax = plt.subplot(n_rows, n_cols, i+1)
        random_index = np.random.randint(0, data.shape[0])
        face_display(data[random_index], ax, f"Face {random_index}")

    plt.tight_layout()
    plt.savefig("Results/Face_display2.png")
    plt.show()


"""
Analyze and visualize the distribution of pixel intensities in the dataset
Arguments: 
    data: array of face data where each row is a flattened face image
"""
def analyze_pixel_distribution(data):
    mean = np.mean(data)
    std = np.std(data)

    print(f"Average pixel intensity: {np.mean(data):.2f}")
    print(f"Standard deviation of pixel intensities: {np.std(data):.2f}")

    # Histogram of pixel intensities
    plt.figure(figsize=(10,6))
    plt.hist(data.flatten(), bins=50, edgecolor='black')
    plt.title("Histogram of Pixel Intensities")
    plt.ylabel("Frequency")
    plt.xlabel("Pixel Intensity")
    plt.savefig("Results/Histogram-Pixel-intensities.png")
    plt.show()

    return mean, std


###################################################################################
""" Problem 2D """

"""
Implementing K-means:

- Decide how many clusters (K) we want to create
- Randomly assign each data point to one of these K clusters. It should not be perfect at this point.
- For each cluster, we calculate its CENTROID. A centroid is the "average" position of all the data points in that cluster, essentially acting as the cluster's center.
- Each data point is then reassigned to the cluster whose centroid is the closest to it, based on distance (often using Euclidian distance).
- So, we check the distance from a data point to each centroid and move the point to the cluster with the neareat centroid.

- After reassigning the data points, we agian compute the centroids of the new clusters.
- We repeat the process of recalculating controids and reassigning data points until the cluster become stable. Stability means that the cluster assignments of data points no longer change, i.e., each point stays in its assigned cluster even after recalculating centroids. 
"""


"""
Randomly initializing centroids by selecting K data points
"""
def centroid_init(data, K):
    random_image = np.random.randint(0, data.shape[0], size=K)          # Choosing a random image/row (shape[0] is rows) from the data set to be the initial centroid. 
    centroids = data[random_image]                                      # Identyfying the centroid as the image at the given index (random_image)

    return np.array(centroids)                                          # Returning the centroids as an array for easy of use later on


"""
Calculating the euclidian distance between two data points
"""
def euclidian_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))        # Returning the distance between two data points
    

"""
Assigning data to the cluster with the closest centroid
"""
def assign_data_to_cluster(data, centroids):
    """
    Parameters:
        data: input data points
        centroids: array of shape (K, n_features) containing current centroid positions

    """
    # Get number of clusters from centroid array
    K = len(centroids)
    # Initializing an array to store cluster assingments for each data point
    clusters = np.zeros(data.shape[0], dtype=int)

    # Loop through each data point in the dataset
    for i in range(data.shape[0]):
        # Initializing a minimum distance as infinity so first real distance wil be smaller
        min_distance = float('inf') 

        # For each data point, loop through all centroids to find the closest one
        for j in range(K):
            # Calculating the euclidian distance between the current data point and the current centroid
            distance = euclidian_distance(data[i], centroids[j])

            # If this distance is smaller then the current minimum, the assignment is updated
            if distance < min_distance:
                min_distance = distance
                clusters[i] = j             # Assigning data point i to cluster j

    # for i in range(K):
    #     print(f"Number of images in cluster {i}; {np.sum(clusters==i)}")        # Overview of how many images are in each cluster
    return clusters


""" 
Calculate new centroid positions
"""
def update_centroids(data, clusters, K):
    """
    Parameters: 
        data: the input dataset (array with shape (n_samples, n_features))
        K: number of clusters
    """
    # Initializing an array to store new centroid positions
    new_centroids = np.zeros((K, data.shape[1]))                                # (shape[1] is columns)

    # Calculate new centroid posotion for each cluster
    for i in range(K):
        # Getting all data points currently assigned to cluster i
        data_points_in_cluster = data[clusters == i]

        # Calculating the mean position of all in a cluster to get a new centroid
        # The mean is calculated across all features for those points
        new_centroids[i] = np.mean(data_points_in_cluster)
    
    return new_centroids


"""
Orchestrate the whole process of the K-means clustering
"""
def k_means_algorithm(data, K, max_iters=100, threshold=0.001):
    """
    Parameters:
        data: input dataset
        K: number of clusters
        max_iters: maximum number of iterations allowed
        threshold: convergence threshold for centroid movement
    """
    # Initializing centroid positions randomly from the data
    old_centroids = centroid_init(data, K)

    # Initializing the convergence flag and an iteration counter
    converged = False
    iters = 0

    # Creating an array to store visually interpretable centroids (the average faces)
    visual_centroids = np.zeros_like(old_centroids)

    # Main K-means loop - continue until convergence or max iterations reached
    while not converged and iters < max_iters:
        # print(f"This is iteration number {iters}")

        # Assigning each data point to the nearest centriod
        old_clusters = assign_data_to_cluster(data, old_centroids)

        # Calculating the new centroid positions based on cluster assignments
        new_centroids = update_centroids(data, old_clusters, K)

        # Calculating the visual centroids based for each cluster
        for i in range(K):
            # Getting all the images assigned to the current cluster
            cluster_images = data[old_clusters == i]
            # Calculating the mean image (average face) for the cluster
            visual_centroids[i] = np.mean(cluster_images, axis=0)

        # Calculating the average movement of centroids between iterations
        distance = np.mean([euclidian_distance(old_centroids[i], new_centroids[i]) for i in range(K)])
        # print(f"  Distance = {distance}\n")

        # Checking if the algorithm has converged (centroid movement is below threshold)
        if distance < threshold:
            # print(f"It is now converged!!\n")
            converged = True

        # Updating the centroid position for the next iteration
        old_centroids = new_centroids.copy()

        iters += 1

    print(f"Done with K-means after {iters} iterations!\n")
    return new_centroids, old_clusters, visual_centroids


""" Visualize K-means clustering results with properly denormalized centroids """
def visualize_clusters(data, centroids, clusters, visual_centroids, K, data_mean=None, data_std=None):
    """
    Parameters:
        data: original (non-normalized) face data
        centroids: array of K centroids (normalized)
        clusters: array of cluster assignments for each data point
        visual_centroids: array of visual representations of centroids (normalized)
        K: number of clusters
        data_mean: mean used for normalization (optional)
        data_std: standard deviation used for normalization (optional)
    """
    fig, axes = plt.subplots(K, 6, figsize=(12, 2*K))
    
    # Denormalize visual centroids if normalization parameters are provided
    if data_mean is not None and data_std is not None:
        visual_centroids = (visual_centroids * data_std) + data_mean
    
    for i in range(K):
        # Get indices of images in current cluster
        image_indices = np.where(clusters == i)[0]
        cluster_images = data[image_indices]
        
        print(f"Processing cluster {i} with {len(cluster_images)} images")
        
        # Calculate distances using normalized data for consistency
        if data_mean is not None and data_std is not None:
            normalized_images = (cluster_images - data_mean) / data_std
            normalized_centroid = centroids[i]
            distances = np.sqrt(np.sum((normalized_images - normalized_centroid)**2, axis=1))
        else:
            distances = np.sqrt(np.sum((cluster_images - centroids[i])**2, axis=1))
        
        # Find 5 closest images to centroid
        closest_positions = np.argsort(distances)[:5]
        actual_indexes = image_indices[closest_positions]
        
        # Display denormalized visual centroid
        face_display(axes[i, 0], visual_centroids[i], f'Centroid {i}\n({len(cluster_images)} images)')
        
        # Display 5 closest images (these are already in original scale)
        for j in range(5):
            face_display(axes[i, j+1], cluster_images[closest_positions[j]], f"Image {actual_indexes[j]}")
    
    plt.tight_layout()
    plt.savefig(f"Results/cluster_results_{K}clusters_testingsomething.png")
    plt.show()

""" Find the best clustering result over multiple tries """
def find_best_clustering(data, normalized_data, K, n_tries=100):
    """
    Parameters:
        data: original (non-normalized) face data
        normalized_data: normalized face data for clustering
        K: number of clusters
        n_tries: number of clustering attempts
    """
    best_inertia = float('inf')
    best_centroids = None
    best_clusters = None
    best_visual_centroids = None
    
    for i in range(n_tries):
        print(f"\nTry {i+1} of {n_tries}")
        centroids, clusters, visual_centroids = k_means_algorithm(normalized_data, K)
        inertia = calculate_inertia(normalized_data, clusters, centroids)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_clusters = clusters
            best_visual_centroids = visual_centroids
            print(f"New best inertia: {best_inertia}")
    
    return best_centroids, best_clusters, best_visual_centroids


"""Calculate the within-cluster sum of squares"""
def calculate_inertia(data, clusters, centroids):
    total_distance = 0
    for cluster_idx in range(len(centroids)):
        cluster_points = data[clusters == cluster_idx]
        if len(cluster_points) > 0:  # Check if cluster is not empty
            centroid = centroids[cluster_idx]
            distances = np.sum((cluster_points - centroid)**2)
            total_distance += distances
    return total_distance


if __name__ == "__main__":
    # Calculate normalization parameters
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    
    # Normalize data for clustering
    normalized_data = (data - data_mean) / data_std
    
    K = 10  # or whatever number of clusters you want
    best_centroids, best_clusters, best_visual_centroids = find_best_clustering(data, normalized_data, K, n_tries=100)
    
    # Ensure clusters are integers
    best_clusters = best_clusters.astype(int)
    
    # Use the modified visualization function with normalization parameters
    visualize_clusters(data, best_centroids, best_clusters, best_visual_centroids, K, data_mean=data_mean, data_std=data_std)
    
    print("\nDONE!")