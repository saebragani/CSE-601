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

# Euclidean distance dictionary
def distance_dictionary (feature_array):
    dist_dic = {}
    for i, x in enumerate(feature_array):
        dist_dic[str(i)] = {}
        for j, y in enumerate(feature_array):
            dist_dic[str(i)][str(j)] = (sum((x-y)**2))**0.5
    return dist_dic

def find_min (dist_dic):
    min_dis = float("inf")
    for key1 in dist_dic.keys():
        for key2 in dist_dic[key1].keys():
            distance = dist_dic[key1][key2]
            if distance != 0:
                if distance < min_dis:
                    min_dis = distance
                    index_out = key1
                    index_in = key2

    return index_out, index_in

def update_dist_dic (dist_dic, index_out, index_in):
    new_key = index_out+"+"+index_in
    keys = dist_dic.keys()
    dist_dic[new_key] = {}
    for key in keys:
        dist_dic[key][new_key] = min(dist_dic[key][index_out], dist_dic[key][index_in])
        dist_dic[new_key][key] = min(dist_dic[index_out][key], dist_dic[index_in][key])
    dist_dic[new_key][new_key] = 0

    del dist_dic[index_out]
    del dist_dic[index_in]
    for key_out in dist_dic.keys():
        del dist_dic[key_out][index_out]
        del dist_dic[key_out][index_in]

    return dist_dic

def cluster_index(dist_dic):
    cluster_index = []
    data_index = []
    for i, key in enumerate(dist_dic.keys()):
        keys = key.split('+')
        for Key in keys:
            cluster_index.append(i+1)
            data_index.append(int(Key))
    index_array = np.zeros((N,2))
    index_array[:, 0] = cluster_index
    index_array[:, 1] = data_index
    sorted_array = index_array[index_array[:, 1].argsort()]
    cluster_index = sorted_array[:,0].tolist()

    return cluster_index

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

    N, k, data_array, centroids, feature_array, gtruth_index = data_extraction(input_file)

    dist_dic = distance_dictionary(feature_array)

    max_iter = 100
    for i in range(max_iter):
        index_out, index_in = find_min(dist_dic)
        dist_dic = update_dist_dic(dist_dic, index_out, index_in)
        if len(dist_dic.keys()) == k:
            break
    # print(len(dist_dic.keys()))
    # print("Number of iterations = ", i)

    cluster_index = cluster_index(dist_dic)

    cluster_incidence = incidence_matrix(cluster_index)
    gtruth_incidence = incidence_matrix(gtruth_index)

    M11, M10, M01, M00 = count(cluster_incidence, gtruth_incidence)

    Rand = (M11 + M00) / (M11 + M00 + M10 + M01)
    Jaccard = M11 / (M11 + M10 + M01)

    print('Rand = ', Rand)
    print('Jaccard = ', Jaccard)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(feature_array)

    fig, ax = plt.subplots()
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_index)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Cluster number")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Hierarchical PCA 2D Visualization")
    plt.show()