import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def SVD_imp(data_file):

    list_disease=[]
    labels=[]

    with open(data_file,'r') as f:

        lines=f.readlines()
        str=lines[0].split('\t')
        N=len(str)-1
        data_matrix = np.zeros((len(lines),N))

        for row,line in enumerate(lines): # row becomes the enumerator and line becomes the components of lines
            feature_str=line.split('\t')

            if not (feature_str[-1] in list_disease):
                list_disease.append(feature_str[-1])

            for col in range(N):
                data_matrix[row,col]=float(feature_str[col])

        for line1 in lines:
            tmp_label=line1.split('\t')[-1]
            labels.append(list_disease.index(tmp_label))

    print(data_matrix.shape)
    U, S, V = np.linalg.svd(data_matrix)

    print(U.shape, S.shape, V.shape)
    data_matrix_new = U[:,:2].dot(np.diag(S[:2]))
    print(data_matrix_new.shape)
    return labels, data_matrix_new

if __name__ == '__main__':
    tmp = 'c'
    data_file = './pca_' + tmp + '.txt'

    labels, data_new = SVD_imp(data_file)

    plt.figure()
    plt.scatter(data_new[:, 0], data_new[:, 1], c=labels)
    plt.show()

