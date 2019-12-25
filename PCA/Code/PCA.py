import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def PCA_1(data_file):

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

    tmp=np.mean(data_matrix, axis=0)
    X_bar=np.tile(tmp, (len(lines), 1))
    X_prime=data_matrix-X_bar
    S=1/(N-1)*((X_prime).T).dot((X_prime))
    w,v=LA.eig(S) # w contains the eigenvalues and v contains the corresponding eigenvectors

    pcs=v[:,[0,1]] # selected eigenvectors

    data_matrix_new=data_matrix.dot(pcs) # principal components

    return labels, data_matrix_new

if __name__ == '__main__':

    tmp='c'
    data_file='./pca_'+tmp+'.txt'

    labels, data_new=PCA_1(data_file)

    plt.figure()
    plt.scatter(data_new[:,0],data_new[:,1], c=labels )
    plt.show()

