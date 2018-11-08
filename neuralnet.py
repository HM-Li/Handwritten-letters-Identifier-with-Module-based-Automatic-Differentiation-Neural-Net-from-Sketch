# %%

import numpy as np
import itertools
import os
import sys
import random


class BasicPackage:

    def read_file(self, path, splitter):
        """read_file read a file from a path and return a list

        Args:
            path (string): os
            splitter (string): splitter
        """
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            contents = np.array(
                list([float(unit) for unit in line.split(splitter)] for line in lines))
        return np.array(contents)

    def split_X_y(self, dataset):
        """split_X_y split X and y from the dataset and do transformation

        Args:
            dataset (list): formatted input data

        Returns:
            list: [description]
        """

        y = dataset[:, 0].astype(int)
        # convert y to int while x still float
        X = dataset[:, 1:]
        return X, y

    def write_file(self, list, path):
        """write_file #  write to a file

        Args:
            list ([type]): [description]
            path ([type]): [description]
        """

        with open(path, 'w') as file:
            for line in list:
                file.write(str(line)+'\n')


class OneHidenLayerNN:
    w1 = None
    w2 = None
    avg_cross_entropy_train_history = None
    avg_cross_entropy_test_history = None
    num_epoch = None
    hidden_units = None
    init_flag = None
    learning_rate = None
    n_features = None
    n_classes = None

    def __init__(self, learning_rate,  num_epoch, hidden_units, init_flag, shaffle=False):
        # initialize the instance
        self.learning_rate = float(learning_rate)
        self.shaffle = shaffle
        self.num_epoch = int(num_epoch)
        self.hidden_units = int(hidden_units)
        self.init_flag = init_flag

    def addBias(self, X):
        # add X0 at the first line of X
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        return X_new

    # The weights are initialized randomly from a uniform distribution from -0.1 to 0.1. The bias parameters are initialized to zero
    def init_weights_random(self):
        self.w1 = np.random.uniform(-0.1, 0.1,
                                    self.hidden_units * (self.n_features))
        self.w1 = self.w1.reshape(self.hidden_units, self.n_features)
        self.w2 = np.random.uniform(-0.1, 0.1,
                                    self.n_classes * (self.hidden_units))
        self.w2 = self.w2.reshape(self.n_classes, self.hidden_units)
        # add bias parameters column at beginning
        w1_new = np.zeros((self.w1.shape[0], self.w1.shape[1] + 1))
        w1_new[:, 1:] = self.w1
        self.w1 = w1_new
        w2_new = np.zeros((self.w2.shape[0], self.w2.shape[1] + 1))
        w2_new[:, 1:] = self.w2
        self.w2 = w2_new
    # All weights are initialized to 0.

    def init_weights_zero(self):
        self.w1 = np.zeros(self.hidden_units * (self.n_features + 1))
        self.w1 = self.w1.reshape(self.hidden_units, self.n_features + 1)
        self.w2 = np.zeros(self.n_classes * (self.hidden_units + 1))
        self.w2 = self.w2.reshape(self.n_classes, self.hidden_units + 1)

    def fit(self, X_train, y_train):
        """fit build up the vocabulary; initiate coefficients; format X

        Args:
            X ([{index:value}]): a list of condensedly stored sparse feature space; the key should be integer

        returns: formatted X with X0 as 1
        """
        # initialize number of features
        self.n_features = X_train.shape[1]
        # initialize number of classes
        self.n_classes = len(np.unique(y_train))
        if (init_flag == '1'):
            random.seed(1)
            self.init_weights_random()
        else:
            random.seed(1)
            self.init_weights_zero()

    def transform(self, X_train, y_train, X_test, y_test):
        # train model with X and y
        # initialize entropy history
        self.avg_cross_entropy_train_history = []
        self.avg_cross_entropy_test_history = []
        # self.training_errors = []

        # add bias to x
        X_train2 = self.addBias(X_train)
        X_test2 = self.addBias(X_test)

        for epoch in range(self.num_epoch):
            # each epoch
            # cross_entropy_train = 0.0
            y_one_hot = self.one_hot(y_train)
            # SGD
            for index, x in enumerate(X_train2):
                # get onhot y
                y = y_one_hot[index]
                # NNForward
                _, a, z, b, y_hat, j = self.NNForward(x, y, self.w1, self.w2)
                # NNBackward
                gw1, gw2 = self.NNBackward(
                    x, y, self.w1, self.w2, a, z, b, y_hat, j)
                # update w1 w2
                self.w1 = self.w1 - self.learning_rate*gw1
                self.w2 = self.w2 - self.learning_rate*gw2
                # update cross entropy
                # cross_entropy_train += self.cal_cross_entropy(
                #     y, y_hat.reshape(len(y_hat),))
                # for each epoch print out cross entropy
            # avg_cross_entropy_train = cross_entropy_train/X_train2.shape[0]
            _, avg_cross_entropy_train = self.predict_prob(X_train2, y_train)

            _, avg_cross_entropy_test = self.predict_prob(X_test2, y_test)
            # update avg cross entropy to the memory
            self.avg_cross_entropy_train_history.append(
                [epoch, avg_cross_entropy_train])
            self.avg_cross_entropy_test_history.append(
                [epoch, avg_cross_entropy_test])
            print("epoch="+str(epoch+1), "crossentropy(train):",
                  avg_cross_entropy_train)
            print("epoch="+str(epoch+1), "crossentropy(test):",
                  avg_cross_entropy_test)
        y_train_hat = self.predict(X_train2, y_train)
        y_test_hat = self.predict(X_test2, y_test)
        error_train = self.cal_error_rate(y_train, y_train_hat)
        error_test = self.cal_error_rate(y_test, y_test_hat)
        print("error(train):", error_train)
        print("error(test):", error_test)

    def one_hot(self, y):
        # do one hot transfer
        # label should be numberic from 0
        return np.eye(self.n_classes)[y]

    def NNForward(self, x, y, w1, w2):
        # for example x and y do module based forward computation
        a = self.linearForward(x, w1)
        z = self.sigmoidForward(a, w1)
        # add bias layer
        z = np.insert(z, 0, 1)
        z = z.reshape(len(z), 1)

        b = self.linearForward(z, w2)
        y_hat = self.softmaxForward(b, w2)
        j = self.crossEntropyForward(y, y_hat)
        return x, a, z, b, y_hat, j

    def NNBackward(self, x, y, w1, w2, a, z, b, y_hat, j):
        gj = 1
        y = y.reshape(len(y), 1)
        x = x.reshape(len(x), 1)
        gy_hat = self.crossEntropyBackward(y, y_hat, j, gj)
        # gb = self.softmaxBackward(b, y_hat, gy_hat)
        gb = (y_hat-y).T
        gw2, gz = self.linearBackward(z, w2, b, gb.T)
        gz = np.delete(gz, 0)
        gz = gz.reshape(len(gz), 1)

        z = np.delete(z, 0)
        z = z.reshape(len(z), 1)
        ga = self.sigmoidBackward(a, z, gz)
        gw1, gx = self.linearBackward(x, w1, a, ga)
        return gw1, gw2

    # sigmoid module
    def sigmoidForward(self, a, w):
        b = self.cal_sigmoid(a)
        return b

    def sigmoidBackward(self, a, b, gb):
        ga = np.multiply(np.multiply(gb, b), (1-b))
        return ga

    # softmax module
    def softmaxForward(self, a, w):
        b = self.cal_softmax(a)
        return b

    def softmaxBackward(self, a, b, gb):
        ga = gb.T.dot(np.diag(b) - b.dot(b.T))
        return ga

    # linear module
    def linearForward(self, a, w1):
        b = np.dot(w1, a)
        return b

    def linearBackward(self, a, w, b, gb):
        gw = np.dot(gb, a.T)
        ga = w.T.dot(gb)
        return gw, ga
    # Cross-Entropy Module

    def crossEntropyForward(self, a, a_hat):
        b = -a.T.dot(np.log(a_hat))
        return b

    def crossEntropyBackward(self, a, a_hat, b, gb):
        ga_hat = -gb*(np.divide(a, a_hat))
        return ga_hat

    def sparse_dot_product(self, dict1, vec2):
        """sparse_dot_product do sparse dot product for two vectors

        Args:
            dict1 (dict): a condensed saved sparse vector, start with the interception
            vec2 (array): a sparse vector
        """
        product = 0.0
        for k, v in dict1.items():
            product += vec2[k]*v
        return product

    def cal_cross_entropy(self, y, y_hat):
        # y is one-hot vector
        # calcuate the average cross entropy
        return -sum(np.multiply(y, np.log(y_hat)))

    def cal_sigmoid(self, a):
        # a is a vector
        return 1.0/(1.0+np.exp(-a))

    def cal_softmax(self, a):
        # a is a vector
        return np.exp(a)/sum(np.exp(a))

    def predict_entry(self, x, y):
        # take an entry and a list of coefficient predict the probability
        _, a, z, b, y_hat, j = self.NNForward(x, y, self.w1, self.w2)
        return y_hat.reshape(len(y_hat),)

    def predict_prob(self, X, Y):
        # X should be added with one bias column
        y_one_hot = self.one_hot(Y)
        # SGD
        Y_hat = []
        for index, x in enumerate(X):
            Y_hat.append(self.predict_entry(x, y_one_hot[index]))
        cross_entropy_test = 0.0
        Y_hat = np.array(Y_hat)
        for y, y_hat in zip(y_one_hot, Y_hat):
            cross_entropy_test += self.cal_cross_entropy(y, y_hat)
        return Y_hat, cross_entropy_test/Y_hat.shape[0]

    def predict(self, X, Y):
        # X should be added with one bias column
        # predict given X and trained coefficient and return a list of labels
        Y_hat, _ = self.predict_prob(X, Y)
        prediction = []
        for y_hat in Y_hat:
            prediction.append(np.argmax(y_hat))
        return np.array(prediction)

    # calculate the error rate for two lists and return the number
    def cal_error_rate(self, attribute_list, prediction_list):
        assert((type(attribute_list) is np.ndarray)
               & (type(prediction_list) is np.ndarray)
               & (attribute_list.size == prediction_list.size)), "Input list format problem."
        right_count = (attribute_list == prediction_list).sum()
        error_rate = (len(attribute_list)-right_count)/len(attribute_list)
        return error_rate
    # calculate regularized loss function
    def cal_cross_entropy_reg(y, y_hat, w1, w2, gamma):
        # remove the bias columns
        return -sum(np.multiply(y, np.log(y_hat)))+gamma/2*np.sum(np.power(w1[:, 1:], 2))+gamma/2*np.sum(np.power(w2[:, 1:], 2))


# %%
# load basic package
bp = BasicPackage()
# load arguments
train_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 5\handin', 'largeTrain.csv'
)


test_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 5\handin', 'largeTest.csv'
)

train_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 5\handin', 'largeoutput', 'train_out.labels')
test_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 5\handin', 'largeoutput', 'test_out.labels')
metrics_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 5\handin', 'largeoutput', 'metrics_out.txt')
num_epoch = '100'
hidden_units = '50'
init_flag = '1'
learning_rate = '0.01'

# train_input_path = os.path.join('./', sys.argv[1])
# test_input_path = os.path.join('./', sys.argv[2])
# train_out_path = os.path.join('./', sys.argv[3])
# test_out_path = os.path.join('./', sys.argv[4])
# metrics_out_path = os.path.join('./', sys.argv[5])
# num_epoch = sys.argv[6]
# hidden_units = sys.argv[7]
# init_flag = sys.argv[8]
# learning_rate = sys.argv[9]


training_set = bp.read_file(train_input_path, ',')
test_set = bp.read_file(test_input_path, ',')
X_train, y_train = bp.split_X_y(training_set)

X_test, y_test = bp.split_X_y(test_set)

# %%
# initialize a classifier
clf = OneHidenLayerNN(learning_rate, num_epoch, hidden_units, init_flag)
clf.fit(X_train, y_train)
clf.transform(X_train, y_train, X_test, y_test)
prediction_test = clf.predict(clf.addBias(X_test), y_test)
prediction_train = clf.predict(clf.addBias(X_train), y_train)
crossentropy_list = []
for epoch in range(clf.num_epoch):
    crossentropy_list.append("epoch="+str(epoch+1)+" crossentropy(train): " +
                             str(clf.avg_cross_entropy_train_history[epoch][1]))
    crossentropy_list.append("epoch="+str(epoch+1)+" crossentropy(test): " +
                             str(clf.avg_cross_entropy_test_history[epoch][1]))
# calculate metrics
train_error = clf.cal_error_rate(np.array(y_train), np.array(prediction_train))
test_error = clf.cal_error_rate(np.array(y_test), np.array(prediction_test))
error_list = ['error(train): '+str(train_error),
              'error(test): ' + str(test_error)]
crossentropy_list.extend(error_list)
# # write metrics
bp.write_file(crossentropy_list, metrics_out_path)
bp.write_file(prediction_train, train_out_path)
bp.write_file(prediction_test, test_out_path)

# %%
test_avg_cross_entropy_hist = []
train_avg_cross_entropy_hist = []
for n_hidden_units in ['5','20','50','100','200']:
    clf = OneHidenLayerNN(learning_rate, num_epoch, n_hidden_units, init_flag)
    clf.fit(X_train, y_train)
    clf.transform(X_train, y_train, X_test, y_test)
    prediction_test, test_avg_cross_entropy = clf.predict_prob(clf.addBias(X_test), y_test)
    prediction_train, train_avg_cross_entropy = clf.predict_prob(clf.addBias(X_train), y_train)
    test_avg_cross_entropy_hist.append(
        [n_hidden_units, test_avg_cross_entropy])
    train_avg_cross_entropy_hist.append(
        [n_hidden_units, train_avg_cross_entropy])

#%%
test_avg_cross_entropy_hist = []
train_avg_cross_entropy_hist = []
learning_rate  = '0.001'
clf = OneHidenLayerNN(learning_rate, num_epoch, hidden_units, init_flag)
clf.fit(X_train, y_train)
clf.transform(X_train, y_train, X_test, y_test)
test_avg_cross_entropy_hist = clf.avg_cross_entropy_test_history
train_avg_cross_entropy_hist = clf.avg_cross_entropy_train_history

# %% create chart
import seaborn as sns
import pandas as pd
log_1 = np.array(train_avg_cross_entropy_hist)
log_2 = np.array(test_avg_cross_entropy_hist)
log = pd.DataFrame(log_1, columns=['epoch', 'avg_cross_entropy_hist'])
log2 = pd.DataFrame(log_2, columns=['epoch', 'avg_cross_entropy_hist'])
log['label'] = "train"
log2['label'] = "test"
log3 = pd.concat([log, log2])
log3['avg_cross_entropy_hist'] = log3['avg_cross_entropy_hist'].astype(float)
# plot = sns.lmplot(x="epoch", y="avg_cross_entropy_hist",
#                   hue="label", data=log3, fit_reg=False, size=8)
plot = sns.lineplot(x="epoch", y="avg_cross_entropy_hist",hue="label", sort=False, style="label", markers=True, dashes=False, data=log3,  size=8)
fig = plot.get_figure()
fig.savefig('./handin/large_learning_rate_0.001.png', dpi=100)
# %%
import pickle
with open(os.path.join('./handin', 'large_learning_rate_0.001.pkl'), 'wb') as f:
    pickle.dump(log3, f)
#%%
sns.lineplot(x="epoch", y="avg_cross_entropy_hist_0.01",
             hue="label", sort=False,style="label", markers=True, dashes=False, data=log3.reset_index(drop=True),  size=8)
