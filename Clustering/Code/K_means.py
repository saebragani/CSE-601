import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Create the ndarray of the input matrix and Initialize the k centroids
def data_extraction(input_file):
    with open(input_file, 'r') as f:
        data = [line.strip().split('\t') for line in f]
        data_array = np.asarray(data)
        data_array = data_array.astype(np.float)
        feature_array = data_array[:,2:]
        gtruth_index = data_array[:, 1]
        clus_num_set = set(gtruth_index)
        if -1 in clus_num_set:
            clus_num_set.remove(-1)
        k = len(clus_num_set)
        centroids = feature_array[0:k, :]

    return k, data_array, centroids, feature_array, gtruth_index

def cluster(feature_array, centroids):

    cluster_index = []
    for i, lines in enumerate(feature_array):
        dist = []
        for cents in centroids:
            dist.append(sum([(a-b)**2 for a, b in zip(lines, cents)])**0.5)
        cluster_index.append(dist.index(min(dist)))

    return cluster_index

def centroid_update(cluster_index, data2, k, feature_length):

    centroids = np.zeros((k,feature_length))
    for i in range(k):
        cluster_points = []
        for j, lines in enumerate(data2):
            if (cluster_index[j] == i):
                cluster_points.append(lines)
        centroids[i,:] = np.mean(cluster_points, axis=0)

    return centroids

def incidence_matrix(cluster):

    N = len(cluster)
    incidence = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (cluster[i]==cluster[j]):
                incidence[i,j] = 1
            else:
                incidence[i,j] = 0
    return incidence

def count(x, y):
    M11 = 0
    M10 = 0
    M01 = 0
    M00 = 0
    N = len(x)
    for i in range(N):
        for j in range(N):
            if x[i,j] == 1 and y[i,j] == 1:
                M11 += 1
            elif x[i,j] == 1 and y[i,j] == 0:
                M10 += 1
            elif x[i,j] == 0 and y[i,j] == 1:
                M01 += 1
            elif x[i,j] == 0 and y[i,j] == 0:
                M00 += 1
    return M11, M10, M01, M00

if __name__ == '__main__':

    input_file='./iyer.txt'

    k, data_array, centroids, feature_array, gtruth_index= data_extraction(input_file)

    ################################## Update centroids and # of clusters here:
    # k = 3
    # centroids = feature_array[[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], :]

    feature_length = centroids.shape[1]

    max_iter = 100
    prev_centroid = np.zeros((k, feature_length))
    for i in range(max_iter):
        cluster_index = cluster(feature_array, centroids)
        centroids = centroid_update(cluster_index, feature_array, k, feature_length)
        if (np.all(centroids == prev_centroid)):
            break
        prev_centroid = centroids

    # print("bbb", cluster_index[321])
    cluster_incidence = incidence_matrix(cluster_index)
    gtruth_incidence = incidence_matrix(gtruth_index)

    M11, M10, M01, M00 = count(cluster_incidence, gtruth_incidence)

    Rand = (M11 + M00)/(M11 + M00 + M10 + M01)
    Jaccard = M11/(M11 + M10 + M01)

    print('Rand Index = ', Rand)
    print('Jaccard Coefficients = ', Jaccard)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(feature_array)
    # centroid_pca = pca.fit_transform(centroids)

    fig, ax = plt.subplots()
    scatter = ax.scatter(data_pca[:,0],data_pca[:,1], c=cluster_index)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Cluster number")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("k-Means PCA 2D Visualization")
    plt.show()