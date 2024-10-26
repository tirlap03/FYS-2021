import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import permutations



# Skal ikke trenge test and split da unsupervised learning ikke har labels

data = np.loadtxt('frey-faces.csv')         # Every row is an image with 20x28 pixels and there is 1965 images
print(f"Data shape: {data.shape}")

"""
Some things about the shape (1965, 560):
- 1965 rows = 1965 images
- 560 columns = flattened pixels (20x28 = 560)
"""

###################################################################################
""" Problem 2C """


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
    random_image = np.random.randint(0, data.shape[0], size=K)                  # Choosing a random image/row (shape[0] is rows) from the data set to be the initial centroid. 
    centroids = data[random_image]                                              # Identyfying the centroid as the image at the given index (random_image)
    # print(f"{K} random images {centroids}")

    return np.array(centroids)                                                  # Returning the centroids as an array for easy of use later on


"""
Calculating the euclidian distance between two data points
"""
def euclidian_distance(point1, point2):
    # total_d = 0
    
    # for i in range(len(point1)):                                                #
    #     d = np.sqrt(np.sum(point1[i] - point2[i])**2)
    #     total_d += d
    return np.sqrt(np.sum((point1 - point2)**2))                                                              # Returning the distance between two data points
    


"""
Assigning data to the cluster with the closest centroid
"""
def assign_data_to_cluster(data, centroids):
    K = len(centroids)
    clusters = np.zeros(data.shape[0])                                          # Creating an empty array, with the size of our data set.


    for i in range(data.shape[0]):                                              # Looping through the dataset for each data point, (i is the index of the current data point we are looking at)
        min_distance = float('inf')                                             # Start distance is set to infinty, then the first distance we set wil be the first minimum distance

        for j in range(K):                                                      # Locating the nearest centorid, (j is the index of a centroid from 0 to K-1)
            distance = euclidian_distance(data[i], centroids[j])                # Calculating the euclidian distance between the data point at index i and the data point set as a centroid at index j. They are both 560 long
            if distance < min_distance:                                         # If the calculated distance is less then the current minimum distance
                min_distance = distance                                         # The minimum distance is updated to be the distance
                clusters[i] = j                                                 # Assigning data point i to cluster j

    for i in range(K):
        print(f"Number of images in cluster {i}; {np.sum(clusters==i)}")        # Overview of how many images are in each cluster
    
    return clusters

""" 
Calculate new centroid positions
"""
def update_centroids(data, clusters, K):
    new_centroids = np.zeros((K, data.shape[1]))                                     # (shape[1] is columns)

    for i in range(K):
        data_points_in_cluster = data[clusters == i]
        new_centroids[i] = np.mean(data_points_in_cluster)
    
    return new_centroids

"""
Orchestrate the whole process of the K-means algorithm
"""
def k_means_algorithm(data, K, max_iters=100, threshold=0.001):
    old_centroids = centroid_init(data, K)
    converged = False
    iters = 0

    while not converged and iters < max_iters:
        print(f"This is iteration number {iters}  ")
        old_clusters = assign_data_to_cluster(data, old_centroids)
        new_centroids = update_centroids(data, old_clusters, K)

        distance = np.mean([euclidian_distance(old_centroids[i], new_centroids[i]) for i in range(K)])
        print(f"  Distance = {distance}\n")

        if distance < threshold:
            print(f"It is now converged!!\n")
            converged = True

        old_centroids = new_centroids.copy()
        iters += 1

    print(f"Done with K-means after {iters} iterations!\n")
    return new_centroids, old_clusters



def visualize_clusters(data, centroids, clusters, K):
    visual_centroids = np.zeros_like(centroids)
    for i in range(K):
        cluster_images = data[clusters == i]

        visual_centroids[i] = np.mean(cluster_images, axis=0)

    fig, axes = plt.subplots(K, 6, figsize=(12, 2*K))

    for i in range(K):
        image_index = np.where(clusters == i)[0]
        cluster_images = data[clusters == i]
        print(f"Processing cluster {i} with {len(cluster_images)} images")

        # Use normalized centroids for distance calculation
        distances = np.sqrt(np.sum((data[clusters == i] - centroids[i])**2, axis=1))

        closest_positions = np.argsort(distances)[:5]
        actual_indexes = image_index[closest_positions]
        
        # Use visual centroid (average face) for display
        face_display(axes[i, 0], visual_centroids[i], f'Centroid {i}\n({len(cluster_images)} images)')
        
        for j in range(5):
            face_display(axes[i, j+1], cluster_images[closest_positions[j]], f"Image {actual_indexes[j]}")
    
    plt.tight_layout()
    plt.savefig("Results/cluster_results_good.png")
    plt.show()


################################################################################################

def compare_with_sklearn(data, k, your_centroids, your_clusters):
    """Compare your KMeans with sklearn's"""
    # Initialize sklearn with verbose output
    sklearn_kmeans = KMeans(
        n_clusters=k, 
        n_init=10, 
        random_state=42, 
        verbose=True,  # This will show iteration info
        max_iter=100
    )
    
    # Fit and get results
    sklearn_clusters = sklearn_kmeans.fit_predict(data)
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    
    # Get iteration information
    print("\nSKLearn K-means Information:")
    print(f"Number of iterations: {sklearn_kmeans.n_iter_}")
    print("\nFinal cluster sizes:")
    for i in range(k):
        print(f"Number of images in cluster {i}: {np.sum(sklearn_clusters == i)}")
    
    return {
        'your_silhouette': silhouette_score(data, your_clusters),
        'sklearn_silhouette': silhouette_score(data, sklearn_clusters),
        'your_inertia': calculate_inertia(data, your_clusters, your_centroids),
        'sklearn_inertia': sklearn_kmeans.inertia_,
        'cluster_agreement': calculate_cluster_agreement(your_clusters, sklearn_clusters, k),
        'sklearn_centroids': sklearn_centroids,
        'sklearn_clusters': sklearn_clusters,
        'n_iterations': sklearn_kmeans.n_iter_
    }

def calculate_inertia(data, clusters, centroids):
    """Calculate the within-cluster sum of squares"""
    total_distance = 0
    for cluster_idx in range(len(centroids)):
        cluster_points = data[clusters == cluster_idx]
        if len(cluster_points) > 0:  # Check if cluster is not empty
            centroid = centroids[cluster_idx]
            distances = np.sum((cluster_points - centroid)**2)
            total_distance += distances
    return total_distance

def calculate_cluster_agreement(your_clusters, sklearn_clusters, k):
    """
    Calculate the percentage of points that are assigned to the same clusters,
    accounting for possible label permutations
    """
    max_agreement = 0
    
    # Try all possible label permutations
    for perm in permutations(range(k)):
        # Create mapping dictionary
        mapping = dict(zip(range(k), perm))
        # Map your cluster labels to new permutation
        mapped_clusters = np.array([mapping[label] for label in your_clusters])
        # Calculate agreement
        agreement = np.mean(mapped_clusters == sklearn_clusters)
        max_agreement = max(max_agreement, agreement)
    
    return max_agreement

def evaluate_clustering(data, k, your_centroids, your_clusters):
    """
    Evaluate your clustering results and print comparison metrics
    """
    results = compare_with_sklearn(data, k, your_centroids, your_clusters)
    
    print("\nClustering Evaluation Results:")
    print("-" * 30)
    print(f"Your Implementation:")
    print(f"Silhouette Score: {results['your_silhouette']:.4f}")
    print(f"Inertia: {results['your_inertia']:.4f}")
    print("\nSklearn Implementation:")
    print(f"Silhouette Score: {results['sklearn_silhouette']:.4f}")
    print(f"Inertia: {results['sklearn_inertia']:.4f}")
    print("\nCluster Agreement:")
    print(f"Percentage of matching assignments: {results['cluster_agreement']*100:.2f}%")
    
    return results


def visualize_sklearn_clusters(data, sklearn_kmeans, K):
    centroids = sklearn_kmeans.cluster_centers_
    clusters = sklearn_kmeans.labels_
    
    # Calculate actual average faces
    visual_centroids = np.zeros_like(centroids)
    for i in range(K):
        cluster_images = data[clusters == i]
        visual_centroids[i] = np.mean(cluster_images, axis=0)
    
    fig, axes = plt.subplots(K, 6, figsize=(12, 2*K))
    
    for i in range(K):
        image_index = np.where(clusters == i)[0]
        cluster_images = data[clusters == i]
        print(f"Processing SKLearn cluster {i} with {len(cluster_images)} images")
        
        distances = np.sqrt(np.sum((data[clusters == i] - centroids[i])**2, axis=1))
        
        closest_positions = np.argsort(distances)[:5]
        actual_indexes = image_index[closest_positions]
        
        face_display(axes[i, 0], visual_centroids[i], 
                    f'SKLearn Centroid {i}\n({len(cluster_images)} images)')
        
        for j in range(5):
            face_display(axes[i, j+1], cluster_images[closest_positions[j]], 
                        f"Image {actual_indexes[j]}")
    
    plt.tight_layout()
    plt.savefig("Results/sklearn_cluster_result_good.png")
    plt.show()


def print_cluster_comparison(your_clusters, sklearn_clusters, k):
    """
    Print detailed comparison between your clusters and sklearn's clusters
    """
    print("\nCluster Size Comparison:")
    print("-" * 30)
    print("Your Implementation:")
    for i in range(k):
        count = np.sum(your_clusters == i)
        percentage = (count / len(your_clusters)) * 100
        print(f"Cluster {i}: {count} images ({percentage:.1f}%)")
    
    print("\nSKLearn Implementation:")
    for i in range(k):
        count = np.sum(sklearn_clusters == i)
        percentage = (count / len(sklearn_clusters)) * 100
        print(f"Cluster {i}: {count} images ({percentage:.1f}%)")
    
    # Get agreement percentage from results
    agreement = calculate_cluster_agreement(your_clusters, sklearn_clusters, k)
    print(f"\nCluster Agreement: {agreement*100:.2f}%")

#################################################################################################################

def find_best_clustering(data, K, n_tries=10):
    best_inertia = float('inf')
    best_centroids = None
    best_clusters = None
    
    for i in range(n_tries):
        print(f"\nTry {i+1} of {n_tries}")
        centroids, clusters = k_means_algorithm(data, K)
        inertia = calculate_inertia(data, clusters, centroids)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_clusters = clusters
            print(f"New best inertia: {best_inertia}")
    
    return best_centroids, best_clusters

if __name__ == "__main__":
    # Load original data
    original_data = np.loadtxt('frey-faces.csv')
    
    # Use normalized data for clustering but keep original for display
    data = (original_data - np.mean(original_data, axis=0)) / np.std(original_data, axis=0)
    
    best_centroids, best_clusters = find_best_clustering(data, 3, n_tries=10)
    visualize_clusters(original_data, best_centroids, best_clusters, 3)
    
    sklearn_kmeans = KMeans(n_clusters=3, n_init=10, random_state=42, verbose=True)
    sklearn_kmeans.fit(data)
    visualize_sklearn_clusters(original_data, sklearn_kmeans, 3)

    # print_cluster_comparison(best_clusters, sklearn_kmeans.labels_, 3)

    evaluate_clustering(data, 3, best_centroids, best_clusters)
    print("\nDONE!")
