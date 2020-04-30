import numpy as np
import os
from getFile import *

train_images = "./data/train-images.idx3-ubyte"
train_labels = "./data/train-labels.idx1-ubyte"
test_images = "./data/t10k-images.idx3-ubyte"
test_labels = "./data/t10k-labels.idx1-ubyte"
K_value = 3
train_size = 10000
test_size = 500

def KNN(testInput,trainSet,labels,k):
    samples_num = trainSet.shape[0]
    #将测试数据复制samples_num份
    diff = np.tile(testInput,(samples_num,1)) - trainSet
    #计算欧式距离
    square_diff = diff ** 2
    square_dist = np.sum(square_diff,axis = 1)
    distance = square_dist ** 0.5
    sorted_dist_indices = np.argsort(distance)

    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indices[i]]
        class_count[vote_label] = class_count.get(vote_label,0)+1
    max_count = 0
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            max_index = key
    return max_index

if __name__ == '__main__':
    print ("load data...")
    train_x,train_y = readImageLabelVector(train_images,train_labels,0,train_size)
    test_x,test_y = readImageLabelVector(test_images,test_labels,0,test_size)
    num_test_samples =test_x.shape[0]
 
    match_count = 0
    for i in range(num_test_samples):
        predict = KNN(test_x[i],train_x,train_y,K_value)
        if predict == test_y[i]:
            match_count += 1

    accuracy = float(match_count)/ num_test_samples
    print("准确率为 %.2f%%" % (accuracy*100))