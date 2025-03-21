'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.decomposition import PCA


#%% read file containing point cloud data
pcd = np.load("dataset2.npy")

print (pcd.shape)
print(pcd)


#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd):
    ground_points=pcd[:,2]
    N=len(ground_points)
    n=int(1+3.3*np.log10(N))
    hist, bin_edges = np.histogram(ground_points, bins=n)
    max_bin_index = np.argmax(hist)
    ground_level = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    plt.figure(figsize=(10, 6))
    plt.hist(ground_points, bins=n, alpha=0.7, color='b', edgecolor='black')
    plt.axvline(ground_level, color='r', linestyle='dashed', label=f"Estimated Ground: {ground_level:.2f}")
    plt.xlabel("ground points")
    plt.ylabel("Frequency")
    plt.title("Histogram of Ground Elevation")
    plt.legend()
    plt.show()
    X=pcd[:,:2]
    y=ground_points
    ransac=RANSACRegressor()
    ransac.fit(X,y)
    inlier_points=ransac.inlier_mask_
    ground_points_ransac = pcd[inlier_points][:,2]
    ground_level_ransac = np.mean(ground_points_ransac)
    N=len(ground_points_ransac)
    n=int(1+3.3*np.log10(N))
    hist, bin_edges = np.histogram(ground_points_ransac, bins=n)
    max_bin_index = np.argmax(hist)
    ground_level = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    plt.figure(figsize=(10, 6))
    plt.hist(ground_points_ransac, bins=n, alpha=0.7, color='b', edgecolor='black')
    plt.axvline(ground_level, color='r', linestyle='dashed', label=f"Estimated Ground: {ground_level_ransac:.2f}")
    plt.xlabel("ground points")
    plt.ylabel("Frequency")
    plt.title("Histogram of Ground Elevation RANSAC approach")
    plt.legend()
    plt.show()
    return ground_level_ransac
 
#%% show downsampled data in external window
#%matplotlib qt
show_cloud(pcd)
#show_cloud(pcd[::10]) # keep every 10th point

#%% remove ground plane
"Task 1"
est_ground_level = get_ground_level(pcd)
print(est_ground_level)
pcd_above_ground = pcd[pcd[:,2] > est_ground_level] 
#%%
pcd_above_ground.shape

#%% side view
#%matplotlib qt
show_cloud(pcd_above_ground)
# %%
unoptimal_eps = 10 
clustering = DBSCAN(eps = unoptimal_eps, min_samples=4).fit(pcd_above_ground)

#%%
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]
# %%
# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)
plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()
#%%
"Task 2"
k=5
# find the elbow and optimal eps
def calculate_k_dist (pcd_above_ground,k=k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(pcd_above_ground)
    distances, _ = neighbors_fit.kneighbors(pcd_above_ground)
    k_dist = np.sort(distances[:, -1])
    return np.sort(k_dist)

def find_knee_point(k_dist_sorted):
    dy = np.diff(k_dist_sorted)
    ddy = np.diff(dy)
    max_curvature_idx = np.argmax(ddy)
    optimal_eps = k_dist_sorted[max_curvature_idx]
    return optimal_eps


def find_knee_point_2(k_dist_sorted):
    i=np.arange(len(k_dist_sorted))
    knee=KneeLocator(i,k_dist_sorted,curve='convex',direction='increasing',interp_method='polynomial')
    optimal_eps_2 = k_dist_sorted[knee.knee]
    return optimal_eps_2, knee

def avg_opt_eps(optimal_eps, optimal_eps_2):
    return np.mean([optimal_eps, optimal_eps_2])

def plot_elbow_chart_knee(k_dist_sorted, k ,opt_eps,knee):
    plt.figure(figsize=(10,6))
    plt.plot(k_dist_sorted,color='darkgreen',label='Distance vs Data could points')
    plt.axvline(knee.knee,color='darkblue',linestyle=':')
    plt.axhline(optimal_eps,color='darkblue',linestyle=':')
    plt.axhline(optimal_eps_avg,color='darkblue',linestyle=':')
    plt.scatter(max_curvature_idx,optimal_eps_avg,color='blue', s=100, label=f"Average Epsilon (eps={optimal_eps_avg:.2f})")
    plt.scatter(knee.knee,optimal_eps_2,color='purple', s=100, label=f"Knee locator Epsilon (eps={optimal_eps_2:.2f})")
    plt.scatter(max_curvature_idx, max_curvature_eps, color='red', s=100, label=f"Max Curvature (eps={optimal_eps:.2f})")
    plt.title("Elbow chart")
    plt.xlabel("Point cloud data")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()
k_dist_sorted=calculate_k_dist(pcd_above_ground,k=k)
print(k_dist_sorted)
print(k_dist_sorted.shape)
optimal_eps_2, knee = find_knee_point_2(k_dist_sorted)
print(f"Optimal eps from knee locator: {optimal_eps_2}")
optimal_eps=find_knee_point(k_dist_sorted)
print(f"Optimal Epsilon (from max curvature):{optimal_eps}")
max_curvature_idx = np.argmax(np.diff(np.diff(k_dist_sorted)))
optimal_eps_avg = avg_opt_eps(optimal_eps, optimal_eps_2)
print(f"Average Optimal Epsilon: {optimal_eps_avg}")
max_curvature_eps = k_dist_sorted[max_curvature_idx]
plot_elbow_chart_knee(k_dist_sorted, k=k, opt_eps=optimal_eps_avg,knee=knee)

#optimal cluster max curvature
clustering_optimal = DBSCAN(eps=optimal_eps_avg, min_samples=k).fit(pcd_above_ground)
num_clusters = len(set(clustering_optimal.labels_)) - (1 if -1 in clustering_optimal.labels_ else 0) 
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, num_clusters)]

# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering_optimal.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)
plt.title('DBSCAN Average epsilon: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()
# %%
"Task 3"
#DBSCAN for optimal epsilon and plot
# Plotting resulting clusters
clustering_optimal = DBSCAN(eps=optimal_eps_avg, min_samples=k).fit(pcd_above_ground)
num_clusters = len(set(clustering_optimal.labels_)) - (1 if -1 in clustering_optimal.labels_ else 0) 
labels = clustering_optimal.labels_
def plot_clusters(pcd_above_ground, labels, title="DBSCAN Clusters"):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    plt.figure(figsize=(10, 10))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0,1] 
        cluster_points = pcd_above_ground[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=2, label=f"Cluster {label}" if label != -1 else "Noise")
    plt.title(title, fontsize=20)
    plt.xlabel('x axis', fontsize=14)
    plt.ylabel('y axis', fontsize=14)
    plt.show()
plot_clusters(pcd_above_ground, clustering_optimal.labels_, title=f"DBSCAN: {num_clusters} clusters")
#Catenary cluster
def find_largest_cluster(pcd_above_ground, labels):

    clusters = [pcd_above_ground[labels == i] for i in set(labels) if i != -1]
    if not clusters:
        print("No valid clusters found!")
        return None
    
    largest_cluster = max(clusters, key=lambda c: (np.max(c[:, 0]) - np.min(c[:, 0])) * (np.max(c[:, 1]) - np.min(c[:, 1])))


    return largest_cluster
largest_cluster=find_largest_cluster(pcd_above_ground, labels)    

if largest_cluster is not None:
    min_x, min_y = np.min(largest_cluster[:, 0]), np.min(largest_cluster[:, 1])
    max_x, max_y = np.max(largest_cluster[:, 0]), np.max(largest_cluster[:, 1])

    print(f"Catenary Cluster Bounding Box:")
    print(f"Min X: {min_x}, Min Y: {min_y}")
    print(f"Max X: {max_x}, Max Y: {max_y}")

    
    plt.figure(figsize=(10, 6))
    plt.scatter(largest_cluster[:, 0], largest_cluster[:, 1], s=1, color='blue')
    plt.title("Largest Cluster (Catenary)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


# %%
#EXTRA FILTERING IF REQUIRED
#x_min, x_max = 7, 40  # Adjust as needed
#y_min, y_max = 0, 80  # Adjust as needed

# Filter points within bounds
#filtered_catenary = largest_cluster[
    #(largest_cluster[:, 0] >= x_min) & (largest_cluster[:, 0] <= x_max) &
    #(largest_cluster[:, 1] >= y_min) & (largest_cluster[:, 1] <= y_max)
#]

#plt.figure(figsize=(10, 6))
#plt.scatter(filtered_catenary[:, 0], filtered_catenary[:, 1], s=1, color='blue')
#plt.title("Largest Cluster filtered (Catenary)")
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.grid(True)
#plt.show()

#def find_largest_cluster_filtered(filtered_catenary, labels):

    #clusters = [filtered_catenary[labels == i] for i in set(labels) if i != -1]
    #if not clusters:
        #print("No valid clusters found!")
        #return None
    
    #largest_cluster = max(clusters, key=lambda c: (np.max(c[:, 0]) - np.min(c[:, 0])) * (np.max(c[:, 1]) - np.min(c[:, 1])))


    #return largest_cluster
#largest_cluster=find_largest_cluster(pcd_above_ground, labels)    

#if largest_cluster is not None:
    #min_x, min_y = np.min(filtered_catenary[:, 0]), np.min(filtered_catenary[:, 1])
    #max_x, max_y = np.max(filtered_catenary[:, 0]), np.max(filtered_catenary[:, 1])

    #print(f"Catenary Filtered Cluster Bounding Box:")
    #print(f"Filtered Min X: {min_x}, Filtered Min Y: {min_y}")
    #print(f"Filtered Max X: {max_x}, Filtered Max Y: {max_y}")

