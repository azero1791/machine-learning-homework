#!/usr/bin/env python

# For convenience of direct executation, use english instead of placing Chinese .ttf file into current directory.
# title: logic regression
# programmer : Zhixuan Qin

import numpy as np
import matplotlib.pyplot as plt

# TODO: open file and get data of given file
def open_get(file):
    lines = []
    for line in file:
        line = line.split()
        if len(line) == 1:
            lines.append(float(line[0]))
        else:
            lines.append([float(e) for e in line])
    
    return lines

def feature_scale(x):
    row, col = x.shape[0], x.shape[1]

    # TODO: split original feature data into groups of same feature
    x_num_lst = np.split(x, col, 1)

    x_num_lst_scaled = []
    # TODO: get single feature of all samples and scale them 
    for x_num_i in x_num_lst:
        x_num_i_mean = np.mean(x_num_i) # TODO: get mean of same feature group
        x_num_i_rlen = np.max(x_num_i) - np.min(x_num_i) # TODO: get range length of same feature group 
        x_num_i_scaled = (x_num_i - x_num_i_mean) / x_num_i_rlen # TODO: get scaled data of same feature group
        x_num_lst_scaled.append(x_num_i_scaled) # group different same feature groups

    # make scaled data into different sample groups
    x_scaled = np.hstack(x_num_lst_scaled)
    print(f"after feature scaling: {x_scaled}\n")
    return x_scaled

# TODO: get cost or loss of current model
def get_cost(x, y, args):
    cost = 0
    for x_i, y_i in zip(x, y):
        cost += -y_i * np.log(h(x_i, args)) - (1 - y_i) * np.log(1 - h(x_i, args))
    return cost
# TODO: calculate hypothesis model
def h(x, args):
   return 1 / (1 + np.exp(-x.dot(args))) 

# TODO: scale single sample features
def single_sample_scale(x, train_x):
   x_mean = np.mean(train_x)
   return (x - x_mean) / (np.max(train_x) - np.min(train_x))

def single_sample_ascale(x, train_x):
    return (np.max(train_x) - np.min(train_x)) * x + np.mean(train_x)
def gd_solve(tmp_x, y):
    # TODO: add first x of argx[0]
    x = np.array([np.concatenate([np.array([1]), x_i]) for x_i in tmp_x])
    # TODO: initialize basic arguments
    epoches = 10000
    learn_rate = 0.01
    args = np.zeros(x.shape[1])
    
    pre_cost = float("inf")
    current_cost = 0
    cost_lst = []
    iter_lst = [] 
    for i in range(epoches):
        gd = np.zeros(x.shape[1])
        for x_i, y_i in zip(x, y):
            gd += (h(x_i, args) - y_i) * x_i 
            
        args -= learn_rate * gd
        current_cost = get_cost(x, y, args)
        cost_lst.append(current_cost)
        iter_lst.append(i+1)
        if (i+1) % 1000 == 0: 
            print(f"{i+1}th epoch: current_cost = {current_cost}")
            print(f"args = {args}\n")
        if pre_cost < current_cost:
            break

    plot_loss(iter_lst, cost_lst, 'loss trend of gradient descent')
    def predict(ori_x, train_x):
        scaled_x = [1]
        scaled_x.extend(single_sample_scale(ori_x, train_x)) 
        return 1.0 if args.dot(scaled_x) > 0 else 0.0
    return predict, args

def plot_scatter(tmp_x, tmp_y, title, xlabel, ylabel, args):
    x_lst = np.split(tmp_x, np.shape(tmp_x)[1], 1) 
    x1, x2 = x_lst[0].reshape((tmp_x.shape[0], 1)), x_lst[1].reshape((tmp_x.shape[0], 1))
    model_x1 = np.arange(np.min(x1), np.max(x1))
    scaled_model_x2 = [(-args[0] - single_sample_scale(model_x1_i, tmp_x) * args[1])/args[2] for model_x1_i in model_x1]
    model_x2 = [single_sample_ascale(scaled_model_x2_i, tmp_x) for scaled_model_x2_i in scaled_model_x2]  
    # Creating plot
    admitted_x, unadmitted_x = [], []
    for x_i, y_i in zip(tmp_x, tmp_y):
        if y_i == 1.0:
            admitted_x.append(x_i)
        else:
            unadmitted_x.append(x_i)
    
    plt.scatter(np.array(admitted_x)[:,0], np.array(admitted_x)[:,1], color='red', label='admitted')
    plt.scatter(np.array(unadmitted_x)[:,0], np.array(unadmitted_x)[:,1], color='blue', label='admitted')
    plt.legend()
    plt.plot(model_x1, model_x2, label = 'learned model')   
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
       
    # show plot
    plt.show()
    
def plot_loss(iter_lst, cost_lst, title):
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.plot(iter_lst, cost_lst, c='red')
    plt.show()
    return   

def main():
    # TODO: get data of feature x
    with open('softmax_data/Exam/train/x.txt') as x_file:
        x = open_get(x_file)

    # TODO: get data of label y
    with open('softmax_data/Exam/train/y.txt') as y_file:
        y = open_get(y_file)
    
    # TODO: get test data set
    with open('softmax_data/Exam/test/x.txt') as test_x_file:
        test_x = open_get(test_x_file)

    with open('softmax_data/Exam/test/y.txt') as test_y_file:
        test_y = open_get(test_y_file)

    # TODO: range of x is larger than y, make feature scaling 
    scaled_x = feature_scale(np.array(x))
    gd_predict, gd_args = gd_solve(scaled_x, np.array(y))
   
    # TODO: picture learned model after gradient descent
    plot_scatter(np.array(x), np.array(y), 'training data set', 'x1', 'x2', gd_args)
    plot_scatter(np.array(test_x), np.array(test_y), 'Test data set', 'x1', 'x2', gd_args)



    correct = 0
    num_test = len(test_y)
    for test_x_i, test_y_i in zip(np.array(test_x), test_y):
       if gd_predict(test_x_i, np.array(x)) == test_y_i:
           correct += 1

    print(f"Test susccessfully: {correct / num_test * 100}%")
if __name__ == "__main__":
    main()
