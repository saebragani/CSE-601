import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import cluster

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
        centroids = data_array[0:k, 2:]

    return k, data_array, centroids, feature_array, gtruth_index

def gaussian_kernel(X, sigma):
    gaus_kern = np.zeros((X.shape[0], X.shape[0]))
    for i,x_i in enumerate(X):
        for j,x_j in enumerate(X):
            gaus_kern[i,j] = np.exp(-((np.linalg.norm(x_i-x_j))**2)/(sigma**2))
            if i == j:
                gaus_kern[i,j] = 0

    D = np.diag(np.sum(gaus_kern, axis=1))
    return gaus_kern, D

def euclidian_distance(X, sigma):
    dist = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            dist[i, j] = (sum((x - y) ** 2)) ** 0.5

    D = np.diag(np.sum(dist, axis=1))
    return dist, D

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

    input_file='./cho.txt'

    k, data_array, centroids, X, gtruth_index= data_extraction(input_file)

    sigma = 20 # 20
    # X = np.array([[0,0.1], [0,0.2], [0, 0.3], [0, 0.4], [0, 0.5], [15,0.1], [15,0.2], [15, 0.3], [15, 0.4], [15, 0.5]])
    W, D = gaussian_kernel(X, sigma)
    # W, D = euclidian_distance(X, sigma)
    L = D-W

    e, v = np.linalg.eig(L) # e:eigenvalue, 386x1; eigenvectors:v:386x386, each col is an eigenvector

    K = 5
    index = np.argpartition(e, K)
    U = np.array(v[:, index[:K]])

    KM = 5
    init_cent = np.array([U[0,:], U[15,:], U[100,:], U[150,:], U[250,:]])
    kmeans = cluster.KMeans(init=init_cent, n_clusters=KM).fit(U) #k-means++
    c = kmeans.labels_
    # print(c)

    cluster_incidence = incidence_matrix(c)
    gtruth_incidence = incidence_matrix(gtruth_index)

    M11, M10, M01, M00 = count(cluster_incidence, gtruth_incidence)

    Rand = (M11 + M00)/(M11 + M00 + M10 + M01)
    Jaccard = M11/(M11 + M10 + M01)

    print('Rand Index = ', Rand)
    print('Jaccard Coefficients = ', Jaccard)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=c)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Cluster number")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Spectral PCA 2D Visualization")
    plt.show()