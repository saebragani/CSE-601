import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Create the ndarray of the input matrix and Initialize the k centroids
def data_extraction(input_file):
    with open(input_file, 'r') as f:
        data = [line.strip().split('\t') for line in f]
        data_array = np.asarray(data)
        data_array = data_array.astype(np.float)
        feature_array = data_array[:, 2:]
        N = len(feature_array)
        gtruth_index = data_array[:, 1]
        clus_num_set = set(gtruth_index)
        if -1 in clus_num_set:
            clus_num_set.remove(-1)
        k = len(clus_num_set)
        centroids = data_array[0:k, 2:]

    return N, k, data_array, centroids, feature_array, gtruth_index

# Euclidean distance matrix
def distance_matrix (feature_array):
    dist = np.zeros((N,N))
    for i, x in enumerate(feature_array):
        for j, y in enumerate(feature_array):
            dist[i,j] = (sum((x-y)**2))**0.5
    return dist

def DBSCAN(X, epsilon, MinPts):
    C = [0]*X.shape[0]
    c = 0
    for pt_id in range(X.shape[0]):
        if C[pt_id] == 0:
            nbr_ids = regionQuery(pt_id, epsilon)
            if len(nbr_ids) < MinPts:
                C[pt_id] = -1
            else:
                c += 1
                expandCluster(pt_id, nbr_ids, C, c, epsilon, MinPts)
    return C

def expandCluster(pt_id, nbr_ids, C, c, epsilon, MinPts):
    C[pt_id] = c
    for id in nbr_ids:
        if C[id] == -1:
            C[id] = c
        if C[id] == 0:
            C[id] = c
            nbr_ids2 = regionQuery(id, epsilon)
            if len(nbr_ids2) >= MinPts:
                nbr_ids += nbr_ids2

def regionQuery(P, epsilon):
    nbr_ids = []
    for i in range(X.shape[0]):
        if dist[P, i] <= epsilon:
            nbr_ids.append(i)
    return nbr_ids

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

    MinPts = 3
    epsilon = 1.0

    N, k, data_array, centroids, X, gtruth_index = data_extraction(input_file)
    dist = distance_matrix (X)
    C = DBSCAN(X, epsilon, MinPts)

    cluster_incidence = incidence_matrix(C)
    gtruth_incidence = incidence_matrix(gtruth_index)

    M11, M10, M01, M00 = count(cluster_incidence, gtruth_incidence)

    Rand = (M11 + M00) / (M11 + M00 + M10 + M01)
    Jaccard = M11 / (M11 + M10 + M01)

    print('Rand Index = ', Rand)
    print('Jaccard Coefficients = ', Jaccard)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=C)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Cluster number")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("DBSCAN PCA 2D Visualization")
    plt.show()

