#!/usr/bin/env python
# title: two-class perceptron vs logic regression and multi-class perceptron vs softmax regression
# programmer: Zhixuan Qin
# last modification time: 2024-11-30

import numpy as np
import matplotlib.pyplot as plt
class Perceptron:

    def __init__(self, num_class, learn_rate = 0.0001, epochs = 20000, threshold = 0.0):
        self.num_class = num_class
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.threshold = threshold
        self.args = 0
        self.x = 0
        self.y = 0

    def train(self, x, y):
        n_sample, n_feature = x.shape
        batch_size = n_sample // 4
        self.args = np.zeros(n_feature + 1)
        self.x, self.y = x, y    
        
        pre_cost, current_cost = float('inf'), 0
        scaled_x = scale_feature(x)
        scaled_x = np.array([[1] + x_i for x_i in scaled_x.tolist()])
        
        loss = []
        itera = []
        best_args, best_cost = self.args, float('inf')
        # TODO: train model step by epoch
        for epoch in range(self.epochs):
            error_count = 0
            # TODO: get random number arr of range(n_sample)
            permutation = np.random.permutation(n_sample)
            for i in range(0, n_sample, batch_size):
                # TODO: get batch data
                batch_idx = permutation[i:i + batch_size]
                x_batch = scaled_x[batch_idx]
                y_batch = y[batch_idx]

                x_batch_scaled = scale_feature(x_batch)
                for x_i, y_i in zip(x_batch_scaled, y_batch):
                    result = x_i.dot(self.args) > self.threshold
                    error = y_i - result
                    if error != 0:
                        error_count += 1
                        for index, value in enumerate(x_i):
                            self.args[index] += self.learn_rate * error * value
            current_cost = self.get_cost(scaled_x, y, self.args)
            itera.append(epoch)
            loss.append(current_cost)
            if epoch % 1000 == 0:
                print(f"{epoch}th cost: ", current_cost)
            if current_cost < best_cost:
                best_args = self.args
                best_error_cost = current_cost
            if error_count == 0:
                break

        self.args = best_args
        print(f"best_args = {best_args}, best_cost = {best_cost}")
        # TODO: draw loss line
        plot_init('loss by two-class perceptron', 'iteration', 'loss')
        plt.plot(itera, loss, color='red')
        plt.show()
    def get_cost(self, x, y, args):
        cost = 0
        for x_i, y_i in zip(x, y):
            hx = args.dot(x_i) > self.threshold
            cost += (hx - y_i) * (args.T).dot(x_i)
        return cost

    def plot(self, title, xlabel, ylabel):
        scaled_x1 = scale_feature(np.array(self.x))[:,0]
        scaled_x2 = (-self.args[0] - self.args[1]*scaled_x1) / self.args[2]
        model_x2 = reverse_scale_feature(scaled_x2, np.array(self.x))
        plot_init(title, xlabel, ylabel)
        plt.plot(np.array(self.x)[:,0], model_x2, color='red')
        plt.show()

    def test(self, test_x, test_y):
        test_x = scale_feature(np.array([np.array([1] + test_x_i.tolist()) for test_x_i in test_x]))
        correct, n_sample = 0, test_x.shape[0]
        for test_x_i, test_y_i in zip(test_x, test_y):
            result = self.args.dot(test_x_i) > 0
            if result == test_y_i:
                correct += 1
        print(f"two-class perceptron model test: {correct / n_sample * 100}%")

def reverse_scale_feature(scaled_x2, train_data):
    x2 = train_data[:,1]
    x2_mean = np.mean(x2)
    x2_std = np.max(x2) - np.min(x2)
    reverse_scaled_x2 = scaled_x2 * x2_std + x2_mean
    return reverse_scaled_x2
    
def plot_init(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# TODO: scale feature by z-score normalization, ref: https://en.wikipedia.org/wiki/Feature_scaling
def scale_feature(x):
    # TODO: get different feature groups
    feature_lst = np.split(x, x.shape[1], 1) 
    scaled_f_lst = []
    for feature in feature_lst:
        mean = feature.mean(axis=0)
        std = np.max(feature) - np.min(feature)
    # TODO: avoid std == 0
        if std == 0:
            std = 1
        feature_normalized = (feature - mean) / std
        scaled_f_lst.append(feature_normalized)
    x_scaled = np.hstack(scaled_f_lst)
    return x_scaled
        
# TODO: open file and get data
def open_get(file):
    lines = []
    for line in file:
        line = line.split()
        if len(line) == 1:
            lines.append(float(line[0]))
        else:
            lines.append([float(e) for e in line])

    return lines
# TODO: get train data set
def get_train_data():
    with open('softmax_data/Exam/train/x.txt') as x_file:
        x = open_get(x_file)

    with open('softmax_data/Exam/train/y.txt') as y_file:
        y = open_get(y_file)

    return np.array(x), np.array(y)
   
# TODO: get test data
def get_test_data():
    with open('softmax_data/Exam/test/x.txt') as x_file:
        x = open_get(x_file)

    with open('softmax_data/Exam/test/y.txt') as y_file:
        y = open_get(y_file)

    return np.array(x), np.array(y)

# TODO: draw train data set
def plot_scatter_data(x, y, title, xlabel, ylabel):
    admitted_x, unadmitted_x = [], []
    for x_i, y_i in zip(x, y):
        if y_i == 1.0:
            admitted_x.append(x_i)
        else:
            unadmitted_x.append(x_i)
    plot_init(title, xlabel, ylabel)
    plt.scatter(np.array(admitted_x)[:,0], np.array(admitted_x)[:,1], color='red', label='admitted')
    plt.scatter(np.array(unadmitted_x)[:,0], np.array(unadmitted_x)[:,1], color='blue', label='unadmitted')
    plt.legend()

class Logic_regression:
    def __init__(self, train_x, train_y, epochs = 10000, learn_rate = 0.001):
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.train_x = train_x
        self.train_y = train_y
        
    def train(self):


def main():
    x, y = get_train_data()
    # TODO: get learned model by two-class perceptron
    pertron2 = Perceptron(2)
    pertron2.train(x, y)

    # TODO: plot train data set
    plot_scatter_data(x, y, 'train data set', 'x1', 'x2')
    # TODO: draw model trained by two-class perceptron
    pertron2.plot('two-class perceptron model around train data set', 'x1', 'x2')
    test_x, test_y = get_test_data()
    # TODO: use two-class perceptron model to validate test data set
    pertron2.test(test_x, test_y)
    plot_scatter_data(test_x, test_y, 'test data set', 'x1', 'x2')
    pertron2.plot('two-class perceptron model around test data set', 'x1', 'x2')
    logic_reg = Logic_regression()
    
if __name__ == "__main__":
    main()
