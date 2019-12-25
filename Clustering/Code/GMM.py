import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
        K = len(clus_num_set)
        centroids = data_array[0:K, 2:]

    return K, data_array, centroids, feature_array, gtruth_index

def initialize(feature_array, K):
    cov1 = 2 * np.identity(feature_array.shape[1])
    sigma = []
    pi = []
    for i in range(K):
        sigma.append(cov1)
        pi.append(1/K)

    # mu = []
    # mu_k = np.zeros((feature_array.shape[1], K))
    # for k in range(K):
    #     for col in range(feature_array.shape[1]):
    #         mu_k[col, :] = np.random.randint(np.min(feature_array[:, col]), np.max(feature_array[:, col]), K)
    #         mu_k[col, :] =
    #     mu.append(mu_k[:,k])

    mu = []
    mu_k = np.zeros((feature_array.shape[1], K))
    for k in range(K):
        mu_k[:, k] = feature_array[k, :]
        mu.append(mu_k[:, k])
    return sigma, mu, pi

def E_step(feature_array, sigma, mu, pi, K, smooth):
    num = np.zeros((feature_array.shape[0], K))
    for k in range(K):
        for i in range(feature_array.shape[0]):
            num[i, k] = pi[k]*multivariate_normal.pdf(feature_array[i, :], mean=mu[k], cov=sigma[k]) #386xK

    den = np.zeros((feature_array.shape[0], 1))
    for i in range(feature_array.shape[0]):
        den[i] = np.sum(num[i, :]) #386x1
    tmp = np.repeat(den, K, axis=1) #386xK

    r_ik = num / (tmp+smooth) #386xK
    return r_ik

def M_step(feature_array, r_ik, K, smooth):
    pi = []
    for k in range(K):
        pi.append(np.sum(r_ik[:, k]) / (feature_array.shape[0]))

    tmp_mu = (np.transpose(r_ik)).dot(feature_array) #shape:Kx16
    tmp = np.sum(r_ik, axis=0).reshape((1, K))
    sigma_r_ik = np.repeat(tmp, feature_array.shape[1], axis=0) #shape:16*K
    tmp2 = np.transpose(tmp_mu) / sigma_r_ik #shape:16*K

    mu = []
    for k in range(K):
        mu.append(tmp2[:, k]) #list of K length; each member is a matrix of shape 1x16

    # mu = []
    # for k in range(K):
    #     num11 = [0]*16
    #     for i in range(feature_array.shape[0]):
    #         num11 += r_ik[i, k]*feature_array[i, :]
    #     den11 = np.sum(r_ik[:, k])
    #     mu.append(num11/den11)

    sigma = []
    for k in range(K):
        tmp_num = np.zeros((feature_array.shape[1], feature_array.shape[1]))
        for i in range(feature_array.shape[0]):
            tmp3 = feature_array[i, :] - mu[k]
            tmp4 = tmp3.reshape((feature_array.shape[1], 1))
            tmp_num += r_ik[i,k]*(tmp4.dot(np.transpose(tmp4)))
        sigma1 = tmp_num/sigma_r_ik[0, k]  #list of lenK; each matrix:16x16
        sigma.append(sigma1+smooth)
    return sigma, mu, pi

def log_likelihood(feature_array, mu, sigma, pi, r_ik, K):
    log_like = 0
    for i in range(feature_array.shape[0]):
        tmp = 0
        for k in range(K):
            tmp += pi[k]*multivariate_normal.pdf(feature_array[i, :], mean=mu[k], cov=sigma[k])
        tmp2 = np.log(tmp)
        log_like += tmp2

    # log_like = 0
    # for i in range(feature_array.shape[0]):
    #     for k in range(K):
    #         tmp = multivariate_normal.pdf(feature_array[i, :], mean=mu[k], cov=sigma[k])
    #         log_like += r_ik[i, k]*(np.log(pi[k])+np.log(tmp))

    return log_like

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

    input_file = './iyer.txt'

    K, data_array, centroids, feature_array, gtruth_index = data_extraction(input_file)

    K = 5
    sigma, mu, pi = initialize(feature_array, K)
    ############Reinitialize
    # sigma[1] = ; sigma[2] = , sigma[3] =
    # pi = [1/3, 1/3, 1/3]
    # mu[0] = feature_array[40, :]; mu[1] = feature_array[100, :]; mu[2] = feature_array[200, :]

    smooth = 1e-5
    max_iter = 40
    conv_thresh = 1e-9

    log_like_old = 0
    log_like = []
    for i in range(max_iter):
        print(i)
        r_ik = E_step(feature_array, sigma, mu, pi, K, smooth)
        sigma, mu, pi = M_step(feature_array, r_ik, K, smooth)
        log_like_current = log_likelihood(feature_array, mu, sigma, pi, r_ik, K)
        log_like.append(log_like_current)
        if abs(log_like_current - log_like_old) < conv_thresh:
            break
        log_like_old = log_like_current

    plt.figure()
    plt.scatter(range(len(log_like)), log_like)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title("Log Likelihood")
    plt.show()

    cluster_index = r_ik.argmax(axis=1)
    cluster_incidence = incidence_matrix(cluster_index)
    gtruth_incidence = incidence_matrix(gtruth_index)

    M11, M10, M01, M00 = count(cluster_incidence, gtruth_incidence)

    Rand = (M11 + M00) / (M11 + M00 + M10 + M01)
    Jaccard = M11 / (M11 + M10 + M01)

    print('Rand Index= ', Rand)
    print('Jaccard Coefficients= ', Jaccard)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(feature_array)

    fig, ax = plt.subplots()
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_index)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Cluster number")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("GMM PCA 2D Visualization")
    plt.show()