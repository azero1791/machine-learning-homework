#!/usr/bin/env python
# title: two-class perceptron vs logic regression and multi-class perceptron vs softmax regression
# programmer: Zhixuan Qin
# last modification time: 2024-11-30

import numpy as np
import matplotlib.pyplot as plt
# TODO: two-class perceptron class
class Perceptron2:

    def __init__(self, learn_rate = 1e-4, epochs = 15000, threshold = 0.0):
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
        best_error_count = x.shape[0]
        # TODO: train model step by epoch
        for epoch in range(self.epochs):
            error_count = 0
            # TODO: get random number arr of range(n_sample)
            permutation = np.random.permutation(n_sample)
            for i in range(0, n_sample, batch_size):
                # TODO: get batch data using SGD
                batch_idx = permutation[i:i + batch_size]
                x_batch = scaled_x[batch_idx]
                y_batch = y[batch_idx]

                for x_i, y_i in zip(x_batch, y_batch):
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
                print(f"(two-class perceptron with SDG){epoch}th cost: ", current_cost)
            #if current_cost < best_cost and epoch > self.epochs * 2 / 3:
            #    print(f"best epoch = {epoch}\n")
            #    best_args = self.args
            #    best_cost = current_cost
            if error_count < best_error_count and epoch > self.epochs * 2 /3:
                print(f"best epoch = {epoch}\n")
                best_args = self.args
                best_error_count = error_count
            if error_count == 0:
                break

        self.args = best_args
        print(f"(two-class perceptron with SGD)best_scaled_args = {best_args}, best_error_count = {best_error_count}")
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
        scaled_x = scale_feature(self.x)
        scaled_x1 = scaled_x[:,0]
        x2 = np.array(self.x[:,1])
        scaled_x2 = (- self.args[1]*scaled_x1) / self.args[2]
        model_x2 = reverse_scale_feature(scaled_x2, x2) - self.args[0]
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

# TODO:multi-class perceptron class 
class Perceptronm:
    def __init__(self, num_class, train_x, train_y, learn_rate = 1e-4, epochs = 15000):
        self.learn_rate = learn_rate
        self.num_class = 2
        self.epochs = epochs
        # TODO: initialize arguments of every class
        self.args = []
        for _ in range(num_class):
            self.args.append(np.zeros(train_x.shape[1] + 1))
        self.args = np.array(self.args)
        self.train_x = train_x
        self.train_y = train_y

    # TODO: get maximum probability class of some sample
    def get_max_pro_class(self, x_i):
        max_class , max_pro = 0, self.args[0].dot(x_i)
        for class_i, args_i in enumerate(self.args):
            current_pro = args_i.dot(x_i)
            if current_pro > max_pro:
                max_class = class_i
                max_pro = current_pro

        return max_class
    def train(self):
        x, y = self.train_x, self.train_y
        n_sample, n_feature = x.shape
        batch_size = n_sample // 4

        cost = float('inf')
        scaled_x = scale_feature(x)
        scaled_x = np.array([[1] + x_i for x_i in scaled_x.tolist()])

        loss, itera = [], []
        best_args, best_error_count = self.args, x.shape[0]

        # TODO: train model step by epoch
        for epoch in range(self.epochs):
            error_count = 0
            # TODO: get random number arr of range(n_sample)
            permutation = np.random.permutation(n_sample)
            for i in range(0, n_sample, batch_size):
                # TODO: get batch data using SGD
                batch_idx = permutation[i:i+batch_size]
                x_batch = scaled_x[batch_idx]
                y_batch = y[batch_idx]
                
                # TODO: train step by sample
                for x_i, y_i in zip(x_batch, y_batch):
                    max_class = self.get_max_pro_class(x_i)
                    # TODO: train step by args of every class
                    for class_i in range(self.args.shape[0]):
                        if class_i == max_class and class_i != y_i:
                            self.args[class_i] -= self.learn_rate * x_i
                        elif class_i == y_i and class_i != max_class:
                            self.args[class_i] += self.learn_rate * x_i
                        if class_i == max_class and class_i == y_i:
                            error_count = error_count
                        else:
                            error_count += 1
            # TODO: remove more useless error_count for every sample
            error_count -= n_sample
            current_cost = self.get_cost(scaled_x, y, self.args)
            itera.append(epoch)
            loss.append(current_cost)
            if epoch % 1000 == 0:
                print(f"(multi-class perceptron with SDG){epoch}th cost: ", current_cost)
            if error_count < best_error_count and epoch > self.epochs * 2 / 3:
                print(f"bese epoch = {epoch}\n")
                best_args = self.args
                best_error_count = error_count
            if error_count == 0:
                break

        self.args = best_args
        print(f"(multi-perceptron with SGD)best_scaled_args = {best_args}, best_error_count = {best_error_count}")
        # TODO: draw loss line
        plot_init('loss by multi-class perceptron', 'iteration', 'loss')
        plt.plot(itera, loss, color = 'red')
        plt.show()

    def get_cost(self, scaled_x, y, args):
        cost = 0
        for x_i, y_i in zip(scaled_x, y):
            max_class = self.get_max_pro_class(x_i)
            hx = (max_class - y_i) != 0
            cost += hx * self.args[max_class].T.dot(x_i)
        return cost
    # TODO: plot learned model 
    def plot(self, title, xlabel, ylabel):
        scaled_x = scale_feature(self.train_x)
        scaled_x1 = scaled_x[:,0]
        x2 = np.array(self.train_x[:,1])
        for class_i, args_i in enumerate(self.args):
            scaled_x2 = (- args_i[1]*scaled_x1) / args_i[2]
            model_x2 = reverse_scale_feature(scaled_x2, x2) - args_i[0] 
            plt.plot(np.array(self.train_x)[:,0], model_x2, color='red', label = f"class-{class_i}")
        plt.legend()
        plt.show()
    # TODO: test data set
    def test(self, test_x, test_y):
        test_x = scale_feature(np.array([np.array([1] + test_x_i.tolist()) for test_x_i in test_x]))
        correct, n_sample = 0, test_x.shape[0]
        for test_x_i, test_y_i in zip(test_x, test_y):
            max_class = self.get_max_pro_class(test_x_i)
            if max_class == test_y_i:
                correct += 1
        print(f"multi-class perceptron model test: {correct / n_sample * 100}%")

        
# TODO: set initial arguments of plt
def plot_init(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# TODO: scale feature by minMax normalization, ref: https://en.wikipedia.org/wiki/Feature_scaling
def scale_feature(x):
    # TODO: get different feature groups
    feature_lst = np.split(x, x.shape[1], 1) 
    scaled_f_lst = []
    for feature in feature_lst:
        mean = feature.mean(axis=0)
        rlen = np.max(feature) - np.min(feature)
    # TODO: avoid rlen == 0
        if rlen == 0:
            rlen = 1
        feature_normalized = (feature - mean) / rlen
        scaled_f_lst.append(feature_normalized)
    x_scaled = np.hstack(scaled_f_lst)
    return x_scaled
        
# TODO: scale feature reversely
def reverse_scale_feature(scaled_feature, feature):
    mean = np.mean(feature)
    rlen = np.max(feature) - np.min(feature)
    return scaled_feature * rlen + mean
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
        self.args = np.zeros(train_x.shape[1]+1)
    def h(self, x_i, args):
        return 1 / (1 + np.exp(-x_i.dot(args)))

    def train(self):
        n_sample, n_feature = self.train_x.shape
        batch_size = n_sample // 4
        pre_cost, current_cost = 0, 0
        
        scaled_x = scale_feature(self.train_x)
        scaled_x = np.array([[1] + x_i for x_i in scaled_x.tolist()])
        y = self.train_y
        
        loss, itera = [], []
        best_args, best_cost = self.args, float('inf')

        # TODO: train model step by epoch using SDG
        for epoch in range(self.epochs):
            permutation = np.random.permutation(n_sample)
            for i in range(0, n_sample, batch_size):
                # TODO: get batch data
                batch_idx = permutation[i:i+batch_size]
                x_batch = scaled_x[batch_idx]
                y_batch = y[batch_idx]

                gd = np.zeros(x_batch.shape[1])
                for x_i, y_i in zip(x_batch, y_batch):
                    gd += (self.h(x_i, self.args) - y_i) * x_i
                    
                self.args -= self.learn_rate * gd
            
            current_cost = self.get_cost(scaled_x, y)
            loss.append(current_cost)
            itera.append(epoch)
            # if epoch % 1000 == 0:
                # print(f"(logic regression with SGD){epoch}th epoch: current_cost = {current_cost}")
            if current_cost < best_cost and epoch > self.epochs * 2 / 3:
                # print (f"(logic regression with SGD)best epoch = {epoch}\n")
                best_args, best_cost = self.args, current_cost
        # TODO: assign best arguments
        self.args = best_args
        print(f"(logic regression with SGD)best_scaled_args = {best_args}, best_cost = {best_cost}") 
        # TODO:draw loss line
        # plot_init('loss by logic regression with SGD', 'iteration', 'loss')
        # plt.plot(itera, loss, color='red')
        # plt.show()

    # TODO: return cost of samples data
    def get_cost(self, x, y):
        cost = 0
        for x_i, y_i in zip(x, y):
            cost += -y_i * np.log(self.h(x_i, self.args) - (1 - y_i) * np.log(1 - self.h(x_i, self.args)))
        return cost

    # TODO:valiate test data
    def test(self, test_x, test_y):
        test_x = scale_feature(np.array([[1] + test_x_i.tolist() for test_x_i in test_x]))
        correct, n_sample = 0, test_x.shape[0]
        for test_x_i, test_y_i in zip(test_x, test_y):
            result = self.args.dot(test_x_i) > 0
            if result == test_y_i:
                correct += 1
        print(f"logic regression with SGD test: {correct / n_sample * 100}%")
        
class Softmax_regression:
    def __init__(self, num_class, train_x, train_y, epochs = 10000, learn_rate = 0.001):
        self.epochs = epochs
        self.num_class = num_class
        self.learn_rate = learn_rate
        self.train_x = train_x
        self.train_y = train_y
        self.args = []
        for _ in range(num_class):
            self.args.append(np.zeros(train_x.shape[1]+1))
    def h(self, x_i, args):
        all_exp = 0
        for args_i in self.args:
            all_exp += np.exp(args_i.dot(x_i))
        hx = np.exp(args.dot(x_i)) / all_exp
        return hx
    def train(self):
        n_sample, n_feature = self.train_x.shape
        batch_size = n_sample // 4
        pre_cost, current_cost = 0, 0
        
        scaled_x = scale_feature(self.train_x)
        scaled_x = np.array([[1] + x_i for x_i in scaled_x.tolist()])
        y = self.train_y
        
        loss, itera = [], []
        best_args, best_cost = self.args, float('inf')

        # TODO: train model step by epoch using SDG
        for epoch in range(self.epochs):
            permutation = np.random.permutation(n_sample)
            for i in range(0, n_sample, batch_size):
                # TODO: get batch data
                batch_idx = permutation[i:i+batch_size]
                x_batch = scaled_x[batch_idx]
                y_batch = y[batch_idx]

                gd = np.zeros(x_batch.shape[1])
                for x_i, y_i in zip(x_batch, y_batch):
                    for class_i, args_i in enumerate(self.args):
                        self.args[class_i] -= self.learn_rate * (self.h(x_i, self.args[class_i]) - y_i) * x_i
            
            current_cost = self.get_cost(scaled_x, y)
            loss.append(current_cost)
            itera.append(epoch)
            # if epoch % 1000 == 0:
               #  print(f"(softmax regression with SGD){epoch}th epoch: current_cost = {current_cost}")
            if current_cost < best_cost and epoch > self.epochs * 2 / 3:
                # print (f"(logic regression with SGD)best epoch = {epoch}\n")
                best_args, best_cost = self.args, current_cost
        # TODO: assign best arguments
        self.args = best_args
        print(f"(logic regression with SGD)best_scaled_args = {best_args}, best_cost = {best_cost}") 
        # TODO:draw loss line
        # plot_init('loss by logic regression with SGD', 'iteration', 'loss')
        # plt.plot(itera, loss, color='red')
        # plt.show()

    # TODO: return cost of samples data
    def get_cost(self, x, y):
        loss = 0
        for x_i, y_i in zip(x, y):
            for class_i, args_i in enumerate(self.args):
                if class_i == y_i:
                    loss += -np.log(self.h(x_i, args_i))
        return loss

    # TODO: get max probability class of some sample
    def get_max_pro_class(self, x_i):
        max_class, max_pro = 0, 0
        for class_i, args_i in enumerate(self.args):
            current_pro = self.h(x_i, args_i)
            if max_pro < current_pro:
                max_pro = current_pro
                max_class = class_i
        return max_class
    # TODO: valiate test data
    def test(self, test_x, test_y):
        test_x = scale_feature(np.array([[1] + test_x_i.tolist() for test_x_i in test_x]))
        correct, n_sample = 0, test_x.shape[0]
        for test_x_i, test_y_i in zip(test_x, test_y):
            max_class = self.get_max_pro_class(test_x_i)
            if max_class == test_y_i:
                correct += 1
        print(f"logic regression with SGD test: {correct / n_sample * 100}%")

def prompt(str):
    print('-'*80)
    print(str)
    print('-'*80)
def main():
    x, y = get_train_data()

    # TODO: get learned model by two-class perceptron
    pertron2 = Perceptron2()

    prompt("two-class perceptron with SGD training...")
    pertron2.train(x, y)

    # TODO: plot train data set
    plot_scatter_data(x, y, 'two-class perceptron model around train data set', 'x1', 'x2')

    # TODO: draw model trained by two-class perceptron
    pertron2.plot('two-class perceptron model around train data set', 'x1', 'x2')
    test_x, test_y = get_test_data()

    # TODO: use two-class perceptron model to validate test data set
    pertron2.test(test_x, test_y)
    plot_scatter_data(test_x, test_y, 'two-class perceptron model around test data set', 'x1', 'x2')
    pertron2.plot('two-class perceptron model around test data set', 'x1', 'x2')
    
    prompt("logic regression with SGD training...")
    
    # TODO: get learned model by logic regression with SDG
    logic_reg = Logic_regression(x, y)
    logic_reg.train()
    logic_reg.test(test_x, test_y)

    # TODO: get learned model by multi-class perceptron
    pertronm = Perceptronm(2, x, y)

    prompt("multi-class perceptron with SGD training...")
    pertronm.train()

    # TODO: draw train data set
    plot_scatter_data(x, y, 'multi-class perceptron model around train data set', 'x1', 'x2')

    # TODO: draw model trained by multi-class pertron
    pertronm.plot('multi-class perceptron model around train data set', 'x1', 'x2')

    # TODO: use multi-class perceptron model to validate test data set
    pertronm.test(test_x, test_y)
    plot_scatter_data(test_x, test_y, 'multi-class perceptron moderl around test data set', 'x1', 'x2')
    pertronm.plot('multi-class perceptron moderl around test data set', 'x1', 'x2')
    
    # TODO: get softmax model by multi-class perceptron
    softmax_reg = Softmax_regression(2, x, y)
    prompt ("softmax regression with SGD training...")
    softmax_reg.train()
    softmax_reg.test(test_x, test_y)

    prompt("TEST SUMMARY:")
    pertron2.test(test_x, test_y)
    logic_reg.test(test_x, test_y)
    pertronm.test(test_x, test_y)
    softmax_reg.test(test_x, test_y)
    
if __name__ == "__main__":
    main()
