import numpy as np
import pdb
import time

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
        for i in range(len(string_index)):
            str_array = np.unique(data_array_tmp[:, string_index[i]])
            unique_strings = str_array.tolist()
            string_dict = dict(zip(unique_strings, range(len(unique_strings))))
            for j in range(data_array_tmp.shape[0]):
                data_array_tmp[j, string_index[i]] = string_dict[data_array_tmp[j, string_index[i]]]

        data_array = data_array_tmp.astype(np.float)
    return data_array

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def binary_divide(input_array, feature_col_ind, feature_divide_value):
    left_ind_list = []
    right_ind_list = []
    for i in range(input_array.shape[0]):
        if input_array[i, feature_col_ind] <= feature_divide_value:
            left_ind_list.append(i)
        else:
            right_ind_list.append(i)

    left = input_array[left_ind_list,:]
    right = input_array[right_ind_list,:]
    return left, right

def gain_calc(left, right):
    C0_parent = list(left[:, -1]).count(0) + list(right[:, -1]).count(0)
    C1_parent = list(left[:, -1]).count(1) + list(right[:, -1]).count(1)
    gini_parent = 1 - (C0_parent / (C0_parent + C1_parent))**2 - (C1_parent / (C0_parent + C1_parent))**2

    C0_left = list(left[:, -1]).count(0)
    C1_left = list(left[:, -1]).count(1)
    if (C0_left + C1_left) != 0:
        gini_left = 1 - (C0_left / (C0_left + C1_left))**2 - (C1_left / (C0_left + C1_left))**2
    else:
        gini_left = 0

    C0_right = list(right[:, -1]).count(0)
    C1_right = list(right[:, -1]).count(1)
    if (C0_right + C1_right) != 0:
        gini_right = 1 - (C0_right / (C0_right + C1_right))**2 - (C1_right / (C0_right + C1_right))**2
    else:
        gini_right = 0

    gini_children = ((C0_left + C1_left) / (C0_parent + C1_parent)) * gini_left + ((C0_right + C1_right) / (C0_parent + C1_parent)) * gini_right
    gain = gini_parent - gini_children
    return gain

def find_best_divide(X_array):
    old_gain = 0
    dict = {}
    for row in range(X_array.shape[0]):
        for col in range(X_array.shape[1] - 1):
            left, right = binary_divide(X_array, col, X_array[row, col])
            gain = gain_calc(left, right)
            if gain > old_gain:
                feature_col_ind = col
                feature_divide_value = X_array[row, col]
                old_gain = gain
                dict = {"feature_col_ind": feature_col_ind, "feature_divide_value": feature_divide_value, "left": left, "right": right}
    print("Gain = ", old_gain)
    return dict

def single_class(X):
    if len(np.unique(X[:, -1])) == 1:
        return True
    else:
        return False

def terminal_class(X):
    if list(X[:, -1]).count(0) >= list(X[:, -1]).count(1):
        return 0
    else:
        return 1

def continue_branching(root_dict, max_depth, depth, min_size):

    left = root_dict["left"]
    right = root_dict["right"]

    del(root_dict["left"])
    del(root_dict["right"])

    if left.shape[0] == 0:
        root_dict["left"] = root_dict["right"] = terminal_class(right)

    if right.shape[0] == 0:
        root_dict["left"] = root_dict["right"] = terminal_class(left)

    if depth >= max_depth:
        root_dict["left"] = terminal_class(left)
        root_dict["right"] = terminal_class(right)

    if left.shape[0] <= min_size:
        root_dict["left"] = terminal_class(left)
    else:
        if single_class(left):
            root_dict["left"] = left[0, -1]
        else:
            root_dict["left"] = continue_branching(find_best_divide(left), max_depth, depth+1, min_size)

    if right.shape[0] <= min_size:
        root_dict["right"] = terminal_class(right)
    else:
        if single_class(right):
            root_dict["right"] = right[0, -1]
        else:
            root_dict["right"] = continue_branching(find_best_divide(right), max_depth, depth+1, min_size)

    return root_dict

def find_test_class(test_data_point, root_dict):
    if test_data_point[root_dict["feature_col_ind"]] < root_dict["feature_divide_value"]:
        if type(root_dict["left"]) == dict:
            return find_test_class(test_data_point, root_dict["left"])
        else:
            return root_dict["left"]
    else:
        if type(root_dict["right"]) == dict:
            return find_test_class(test_data_point, root_dict["right"])
        else:
            return root_dict["right"]

def evaluation(test_pred_class, test):
    a = 0 # TP
    b = 0 # FN
    c = 0 # FP
    d = 0 # TN

    for i in range(len(test_pred_class)):
        if test[i, -1] == 1 and test_pred_class[i] == 1:
            a += 1
        elif test[i, -1] == 1 and test_pred_class[i] == 0:
            b += 1
        elif test[i, -1] == 0 and test_pred_class[i] == 1:
            c += 1
        elif test[i, -1] == 0 and test_pred_class[i] == 0:
            d += 1

    accuracy = (a+d)/(a+b+c+d+1e-9)
    precision = a/(a+c+1e-9)
    recall = a/(a+b+1e-9)
    f1_measure = (2*a)/(2*a+b+c+1e-9)

    return accuracy, precision, recall, f1_measure


if __name__ == '__main__':

    # input_file='./project3_dataset1.txt'
    #
    # data_array = data_extraction(input_file)
    #
    # cross_valid = np.array_split(data_array, 10)
    #
    # accuracy_list = []
    # precision_list = []
    # recall_list = []
    # f1_list = []
    # for i in range(len(cross_valid)):
    #     test = cross_valid[i]
    #     train = np.array([]).reshape(0, cross_valid[0].shape[1])
    #     for j in range(len(cross_valid)):
    #         if j != i:
    #             train = np.concatenate((train, cross_valid[j]), axis=0)
    #
    #     root_dict = find_best_divide(train)
    #     max_depth, min_size = 100, 10
    #     root_dict = continue_branching(root_dict, max_depth, 1, min_size)
    #
    #     test_pred_class = []
    #     for test_data_point in test:
    #         test_pred_class.append(find_test_class(test_data_point, root_dict))
    #
    #     accuracy, precision, recall, f1_measure = evaluation(test_pred_class, test)
    #
    #     print("cross_val iteration = ", i+1)
    #     print("accuracy = ", accuracy)
    #     print("precision = ", precision)
    #     print("recall = ", recall)
    #     print("f1_measure = ", f1_measure)
    #
    #     accuracy_list.append(accuracy)
    #     precision_list.append(precision)
    #     recall_list.append(recall)
    #     f1_list.append(f1_measure)
    #
    # print("average accuracy = ", sum(accuracy_list) / len(accuracy_list))
    # print("average precision = ", sum(precision_list) / len(precision_list))
    # print("average recall = ", sum(recall_list) / len(recall_list))
    # print("average f1 measure= ", sum(f1_list) / len(f1_list))










    input_file = './project3_dataset4.txt'
    data_array= data_extraction(input_file)
    train = data_array

    root_dict = find_best_divide(train)
    max_depth, min_size = 100, 1
    root_dict = continue_branching(root_dict, max_depth, 1, min_size)

    # print(root_dict)
    # print(root_dict["feature_col_ind"], root_dict["left"], root_dict["right"])
    print(root_dict["feature_col_ind"])
    print(root_dict["left"], root_dict["right"])
