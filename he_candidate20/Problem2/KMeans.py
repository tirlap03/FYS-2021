import numpy as np

""" Problem 2D """


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

""" Calculate the within-cluster sum of squares (inertia) """
def calculate_inertia(data, clusters, centroids):
    """
    Parameters:
    data: array of data points
    clusters: array of cluster assignments for each point
    centroids: array of centroid positions
    """

    # Initialize total distance counter
    total_distance = 0
    
    # Loop through each cluster
    for cluster_idx in range(len(centroids)):
        # Get all points assigned to current cluster
        cluster_points = data[clusters == cluster_idx]
        
        # Only process cluster if it contains points
        if len(cluster_points) > 0:
            # Get centroid for current cluster
            centroid = centroids[cluster_idx]
            # Calculate squared distances from points to centroid
            distances = np.sum((cluster_points - centroid)**2)
            # Add to total distance
            total_distance += distances
            
    return total_distance

""" Find the best clustering result by running k-means multiple times """
def find_best_clustering(data, normalized_data, K, n_tries=10):
    """    
    Parameters:
    data: original (non-normalized) face data
    normalized_data: normalized face data for clustering
    K: number of clusters
    n_tries: number of clustering attempts
    """

    # Initialize best results with worst possible score
    best_inertia = float('inf')
    best_centroids = None
    best_clusters = None
    best_visual_centroids = None
    
    # Try k-means multiple times with different random initializations
    for i in range(n_tries):
        print(f"\nTry {i+1} of {n_tries}")
        
        # Run k-means clustering
        centroids, clusters, visual_centroids = k_means_algorithm(normalized_data, K)
        
        # Calculate quality score (inertia) for this attempt
        inertia = calculate_inertia(normalized_data, clusters, centroids)
        
        # If this attempt is better than previous best, update best results
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_clusters = clusters
            best_visual_centroids = visual_centroids
            print(f"New best inertia: {best_inertia}")
    
    # Return the results from the best clustering attempt
    return best_centroids, best_clusters, best_visual_centroids


if __name__ == "__main__":
    data = np.loadtxt('frey-faces.csv')         # Every row is an image with 20x28 pixels and there is 1965 images

    # Calculate normalization parameters
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    
    # Normalize data for clustering
    normalized_data = (data - data_mean) / data_std
    
    K = 10  # or whatever number of clusters you want
    best_centroids, best_clusters, best_visual_centroids = find_best_clustering(data, normalized_data, K, n_tries=100)