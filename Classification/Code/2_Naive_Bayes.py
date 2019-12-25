import numpy as np
import random
import pdb

def data_extraction(input_file):
    with open(input_file, 'r') as f:
        data = [line.strip().split('\t') for line in f]
        data_array_tmp = np.asarray(data)

        string_index = []
        for i in range(data_array_tmp.shape[1]):
            if is_number(data_array_tmp[0, i]) == False:
                string_index.append(i)

        string_dict_list = []
        # replace the string clolumn with values 0, 1, 2, ...
        for i in range(len(string_index)):
            str_array = np.unique(data_array_tmp[:, string_index[i]])
            unique_strings = str_array.tolist()
            string_dict = dict(zip(unique_strings, range(len(unique_strings))))
            string_dict_list.append(string_dict)
            for j in range(data_array_tmp.shape[0]):
                data_array_tmp[j, string_index[i]] = string_dict[data_array_tmp[j, string_index[i]]]
        data_array = data_array_tmp.astype(np.float)
    return data_array, string_index, string_dict_list

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def pX(x,feature):
    vec=np.array(feature)
    mu = np.mean(vec)
    sig= np.std(vec)
    pdf_x = (1/(sig*(2*(np.pi))**0.5))*np.exp(-.5*((x - mu)**2)/(sig ** 2))
    return pdf_x


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


if __name__ == '__main__':
    input_file = './project3_dataset2.txt'
    data_array, string_index, string_dict_list = data_extraction(input_file)

    cross_valid = np.array_split(data_array, 10)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(len(cross_valid)):
        test_class_actual = cross_valid[i][:, -1]
        test = cross_valid[i]
        test = test.tolist()

        train = np.array([]).reshape(0, cross_valid[0].shape[1])
        for j in range(len(cross_valid)):
            if j != i:
                train = np.concatenate((train, cross_valid[j]), axis=0)
        train_class = train[:, -1]
        labels = data_array[:,-1].tolist()


        P_H={} #probability of each label can cover for multi labels
        X_given_H={}
        possible_H=list(set(labels))
        for label in possible_H:
            P_H[label]=labels.count(label)/len(labels)
            X_given_H[label]=[item.tolist() for item in train if item[-1]==label]

        list_P_Hgiven_test = []
        prediction=[]
        for point in test:

            P_H_given_X={}
            for label in possible_H:

                P_X_given_H =1
                count=0
                for feature in point[:-1]:
                    vec = [item[count] for item in X_given_H[label]]
                    if is_number(feature):
                        P_X_given_H*=pX(feature,vec)
                    else:
                        P_X_given_H *= (vec.count(feature)/len(vec))
                    count+=1

                P_H_given_X[label] = P_H[point[-1]] * P_X_given_H
            list_P_Hgiven_test.append(P_H_given_X)

            tmp_list=zip(P_H_given_X.values() , P_H_given_X.keys())
            prediction.append(sorted(tmp_list)[-1][-1])

        # print("P_H_given_X = ", list_P_Hgiven_test)


        test_class_pred = np.array(prediction)
        accuracy, precision, recall, f1_measure = evaluation(test_class_actual, test_class_pred)

        print("cross_val iteration = ", i + 1)
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






















    # input_file = './project3_dataset4.txt'
    # data_array, string_index, string_dict_list = data_extraction(input_file)
    #
    # train = data_array
    # train_class = train[:, -1]
    # labels = data_array[:,-1].tolist()
    #
    # X = ["sunny", "cool", "high", "weak"]
    # test = X
    # j = 0
    # for i in range(len(X)):
    #     if is_number(test[i]) == False:
    #         test[i] = string_dict_list[j][X[i]]
    #         j += 1
    #
    # P_H={} #probability of each label can cover for multi labels
    # X_given_H={}
    # possible_H=list(set(labels))
    # for label in possible_H:
    #     P_H[label]=labels.count(label)/len(labels)
    #     X_given_H[label]=[item.tolist() for item in train if item[-1]==label]
    #
    # list_P_Hgiven_test = []
    # prediction=[]
    # # for point in test:
    # point=test
    # P_H_given_X={}
    # for label in possible_H:
    #
    #     P_X_given_H =1
    #     count=0
    #     for feature in point[:-1]:
    #         vec = [item[count] for item in X_given_H[label]]
    #         if is_number(feature):
    #             P_X_given_H*=pX(feature,vec)
    #         else:
    #             P_X_given_H *= (vec.count(feature)/len(vec))
    #         count+=1
    #
    #     P_H_given_X[label] = P_H[point[-1]] * P_X_given_H
    # list_P_Hgiven_test.append(P_H_given_X)
    #
    # tmp_list=zip(P_H_given_X.values() , P_H_given_X.keys())
    # prediction.append(sorted(tmp_list)[-1][-1])
    #
    # print("P_H_given_X = ", list_P_Hgiven_test)
    #
    # # test_class_pred = np.array(prediction)



