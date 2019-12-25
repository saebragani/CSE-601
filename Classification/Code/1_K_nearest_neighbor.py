import numpy as np
import pdb

# Create the ndarray of the input data
def data_extraction(input_file):
    with open(input_file, 'r') as f:
        data = [line.strip().split('\t') for line in f]
        data_array_tmp = np.asarray(data)

        string_index = []
        for i in range(data_array_tmp.shape[1]):
            if is_number(data_array_tmp[0, i]) == False:
                string_index.append(i)

        # replace the string clolumn with values 0, 1, 2, ...
        # unique_strings = []
        for i in range(len(string_index)):
            str_array = np.unique(data_array_tmp[:, string_index[i]])
            unique_strings = str_array.tolist()
            # unique_strings.append(np.unique(data_array_tmp[:, string_index[i]]))
            # unique_strings = unique_strings[0].tolist() # In data2 unique_string[0] gives ['Absent', 'Present']
            string_dict = dict(zip(unique_strings, range(len(unique_strings))))
            for j in range(data_array_tmp.shape[0]):
                data_array_tmp[j, string_index[i]] = string_dict[data_array_tmp[j, string_index[i]]]

        # print(data_array_tmp[0:5,:])
        # pdb.set_trace()
        data_array = data_array_tmp.astype(np.float)
        # data_array = np.delete(data_array, string_index, axis=1)

        # # string = data_array_tmp[:, string_index]
        # data_array = np.delete(data_array_tmp, string_index, 1)
        # data_array = data_array.astype(np.float)

    return data_array


def normalize_features(train, test):
    mean_train = np.mean(train, 0)
    mean_train_rep = np.repeat(mean_train.reshape(1,train.shape[1]), train.shape[0], axis=0)
    train_std = np.std(train, 0)
    train_std_rep = np.repeat(train_std.reshape(1, train.shape[1]), train.shape[0], axis=0)
    norm_train = (train - mean_train_rep) / train_std_rep

    mean_test = np.mean(test, 0)
    mean_test_rep = np.repeat(mean_test.reshape(1, test.shape[1]), test.shape[0], axis=0)
    test_std = np.std(test, 0)
    test_std_rep = np.repeat(test_std.reshape(1, test.shape[1]), test.shape[0], axis=0)
    norm_test = (test - mean_test_rep) / test_std_rep
    return norm_train, norm_test


def euc_dist(x,y): # x=test
    dist = np.zeros((x.shape[0], y.shape[0]))
    for i, x_point in enumerate(x):
        for j, y_point in enumerate(y):
            dist[i, j] = (sum((x_point - y_point) ** 2)) ** 0.5
    return dist


def find_test_class_pred(dist, K, train_class, test):
    test_class_pred = np.zeros((test.shape[0]))
    for i in range(test.shape[0]):
        distances = dist[i,]
        idx = np.argpartition(distances, K)
        zero_count = 0
        one_count = 0
        for j in idx[:K]:
            if (train_class[j] == 0):
                zero_count += 1/(dist[i,j]**2+1e-9)
            else:
                one_count += 1/(dist[i,j]**2+1e-9)

        if (zero_count > one_count):
            test_class_pred[i] = 0
        else:
            test_class_pred[i] = 1
    return test_class_pred

# Compute the accuracy, precision, recall, and f-1 measure on the training set
def evaluation(test_class_actual, test_class_pred):
    a = 0 # TP
    b = 0 # FN
    c = 0 # FP
    d = 0 # TN

    for i in range(len(test_class_actual)):
        if test_class_actual[i] == 1 and test_class_pred[i] == 1:
            a += 1
        elif test_class_actual[i] == 1 and test_class_pred[i] == 0:
            b += 1
        elif test_class_actual[i] == 0 and test_class_pred[i] == 1:
            c += 1
        elif test_class_actual[i] == 0 and test_class_pred[i] == 0:
            d += 1

    accuracy = (a+d)/(a+b+c+d+1e-9)
    precision = a/(a+c+1e-9)
    recall = a/(a+b+1e-9)
    f1_measure = (2*a)/(2*a+b+c+1e-9)

    return accuracy, precision, recall, f1_measure

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':

    input_file='./project3_dataset2.txt'
    K = 9 # Odd number

    data_array = data_extraction(input_file)

    cross_valid = np.array_split(data_array, 10) # creates a list. each member is a ndarray. starts from the top rows and splits the matrix so that the final list has length of second arg

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(len(cross_valid)):
        test_class_actual = cross_valid[i][:, -1]
        test = cross_valid[i][:,:-1]

        train = np.array([]).reshape(0, cross_valid[0].shape[1])
        for j in range(len(cross_valid)):
            if j != i:
                train = np.concatenate((train, cross_valid[j]), axis=0)
        train_class = train[:, -1]
        train = train[:, :-1]

        norm_train, norm_test = normalize_features(train, test)

        dist = euc_dist(norm_test, norm_train)
        test_class_pred = find_test_class_pred(dist, K, train_class, test)

        accuracy, precision, recall, f1_measure = evaluation(test_class_actual, test_class_pred)

        print("cross_val iteration = ", i+1)
        print("accuracy = ", accuracy)
        print("precision = ", precision)
        print("recall = ", recall)
        print("f1 measure = ", f1_measure)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_measure)

    print("average accuracy = ", sum(accuracy_list) / len(accuracy_list))
    print("average precision = ", sum(precision_list) / len(precision_list))
    print("average recall = ", sum(recall_list) / len(recall_list))
    print("average f1 measure= ", sum(f1_list) / len(f1_list))



















    # input_file_train = './project3_dataset3_train.txt'
    # input_file_test = './project3_dataset3_test.txt'
    # K = 9  # Odd number
    #
    # data_array_train = data_extraction(input_file_train)
    # data_array_test = data_extraction(input_file_test)
    #
    # train_class = data_array_train[:, -1]
    # train = data_array_train[:, :-1]
    #
    # test_class_actual = data_array_test[:, -1]
    # test = data_array_test[:, :-1]
    #
    # dist = euc_dist(test, train)
    # test_class_pred = find_test_class_pred(dist, K, train_class, test)
    # print(test_class_pred)
    #
    # accuracy, precision, recall, f1_measure = evaluation(test_class_actual, test_class_pred)
    #
    # print("accuracy = ", accuracy)
    # print("precision = ", precision)
    # print("recall = ", recall)
    # print("f1 measure = ", f1_measure)






