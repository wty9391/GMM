#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

epsilon = sys.float_info.epsilon


class wining_pridictor():
    def __init__(self, d=0.1, max_iter=5000, eta=0.001, step=1, eta_decay=0.99, pattern="tanh"):
        self.d = d
        self.max_iter = max_iter
        self.eta = eta
        self.step = step
        self.eta_decay = eta_decay

        if pattern not in ['tanh', 'long_tail', 'log_gaussian']:
            raise Exception("Unsupport winning function {}".format(pattern))

        self.pattern = pattern

    # winning function type:tanh
    def w1(self, b):
        # b is bid price
        # d is hyperparameter need to be fit
        return np.tanh(b * self.d)

    # winning function: w = b/(b+d)
    def w2(self, b):
        # b is bid price
        # d is hyperparameter need to be fit
        return b / (b + self.d)

    # winning function: log-Gaussian \Phi((log(b)-\mu)/\sigma)
    def w3(self, b):
        # b is bid price
        return norm.cdf(np.log(b + epsilon), self.mu, self.sigma)

    def w_der(self, b):
        if self.pattern == 'tanh':
            return self.w_der1(b)
        elif self.pattern == 'long_tail':
            return self.w_der2(b)
        elif self.pattern == 'log_gaussian':
            return self.w_der3(b)
        else:
            raise Exception("Unsupport winning function {}".format(self.pattern))

    def w_der1(self, b):
        # b is bid price
        # d is hyperparameter need to be fit
        return self.d * (1 - np.tanh(b * self.d) ** 2)

    def w_der2(self, b):
        # b is bid price
        # d is hyperparameter need to be fit
        return self.d / ((b + self.d) ** 2)

    def w_der3(self, b):
        # b is bid price
        return norm.pdf(np.log(b + epsilon), self.mu, self.sigma) * 1 / ((b + epsilon) * self.sigma)

    def w_der_d1(self, b):
        return b * (1 - np.tanh(b * self.d) ** 2)

    #    def w_der_d_kl(self,b,y):
    #        return (self.w_der_d(b)).transpose().dot(
    #                np.log(self.w(b)/y) + 1)

    def w_der_d2(self, b):
        return -b / ((b + self.d) ** 2)

    def fit1(self, x, y):
        print("Now start to fit wining cdf")
        record_num = np.shape(x)[0]

        for i in range(self.max_iter):
            # least squares
            error = self.w1(x) - y
            self.d = self.d - self.eta * (1 / record_num) * (error.transpose().dot(self.w_der_d1(x)))
            loss = 1 / 2 * 1 / (record_num) * (error ** 2).sum()

            # kl divergence
            #            self.d = self.d - self.eta * (1/record_num)*self.w_der_d_kl(x,y)
            #            loss = entropy(self.w(x), y)
            if i % 50 == 0:
                print("epoch {}, d {}, loss {}".format(i, self.d, loss))

            self.eta = self.eta * self.eta_decay

        print("Fit finished")

        # speed up integration
        bins = 2000
        step = self.step
        self.integrate_b1 = np.arange(0, bins + 1, step)
        self.integrate_b_y1 = self.w_der1(self.integrate_b1)
        temp = self.integrate_b1 * self.integrate_b_y1
        self.integrate1 = np.zeros_like(self.integrate_b1, dtype=np.float)
        self.integrate1[0] = temp[0] * step
        for i in range(bins):
            self.integrate1[i + 1] = self.integrate1[i] * 1.0 + temp[i + 1] * step
        self.integrate1[0] = self.integrate1[1]  # incase of zero

    def fit2(self, x, y):
        print("Now start to fit wining cdf")
        record_num = np.shape(x)[0]
        for i in range(self.max_iter):
            error = self.w2(x) - y
            self.d = self.d - self.eta * (1 / record_num) * (error.transpose().dot(self.w_der_d2(x)))
            loss = 1 / 2 * 1 / (record_num) * (error ** 2).sum()
            if i % 50 == 0:
                print("epoch {}, loss {}".format(i, loss))

            self.eta = self.eta * self.eta_decay
        print("Fit finished")

        # speed up integration
        bins = 2000
        step = self.step
        self.integrate_b2 = np.arange(0, bins + 1, step)
        self.integrate_b_y2 = self.w_der2(self.integrate_b2)
        temp = self.integrate_b2 * self.integrate_b_y2
        self.integrate2 = np.zeros_like(self.integrate_b2, dtype=np.float)
        self.integrate2[0] = temp[0] * step
        for i in range(bins):
            self.integrate2[i + 1] = self.integrate2[i] * 1.0 + temp[i + 1] * step
        self.integrate2[0] = self.integrate2[1]  # incase of zero

    def fit3(self, pays):
        print("Now start to fit wining cdf")
        self.mu, self.sigma = norm.fit(np.log(pays + epsilon))
        print("Fit finished")

        # speed up integration
        bins = 2000
        step = self.step
        self.integrate_b3 = np.arange(0, bins + 1, step)
        self.integrate_b_y3 = self.w_der3(self.integrate_b3)
        temp = self.integrate_b3 * self.integrate_b_y3
        self.integrate3 = np.zeros_like(self.integrate_b3, dtype=np.float)
        self.integrate3[0] = temp[0] * step
        for i in range(bins):
            self.integrate3[i + 1] = self.integrate3[i] * 1.0 + temp[i + 1] * step
        self.integrate3[0] = self.integrate3[1]  # incase of zero

    def plot1(self, x, y):
        y_prediction = self.w1(x)
        plt.plot(y, label='truth')
        plt.plot(y_prediction, label='w=tanh({0:.5f}x)'.format(self.d))
        plt.legend()

    def plot2(self, x, y):
        y_prediction = self.w2(x)
        plt.plot(y, label='truth')
        plt.plot(y_prediction, label='w=b/(b+{0:.5f})'.format(self.d))
        plt.legend()

    def predict(self, x):
        if self.pattern == 'tanh':
            return self.predict1(x)
        elif self.pattern == 'long_tail':
            return self.predict2(x)
        elif self.pattern == 'log_gaussian':
            return self.predict3(x)
        else:
            raise Exception("Unsupport winning function {}".format(self.pattern))

    def predict1(self, x):
        return self.w1(x)

    def predict2(self, x):
        return self.w2(x)

    def predict3(self, x):
        return self.w3(x)

    def predict_win_price(self, x):
        if self.pattern == 'tanh':
            return self.predict_win_price1(x)
        elif self.pattern == 'long_tail':
            return self.predict_win_price2(x)
        elif self.pattern == 'log_gaussian':
            return self.predict_win_price3(x)
        else:
            raise Exception("Unsupport winning function {}".format(self.pattern))

    def predict_win_price1(self, x):
        x = np.round(x)
        r = np.zeros_like(x)
        for i in range(np.shape(x)[0]):
            index = int(x[i])
            try:
                r[i] = self.integrate1[index]
            except IndexError:
                r[i] = self.integrate1[-1]

        return r

    def predict_win_price2(self, x):
        x = np.round(x)
        r = np.zeros_like(x)
        for i in range(np.shape(x)[0]):
            index = int(x[i])
            try:
                r[i] = self.integrate2[index]
            except IndexError:
                r[i] = self.integrate2[-1]

        return r

    def predict_win_price3(self, x):
        x = np.round(x)
        r = np.zeros_like(x)
        for i in range(np.shape(x)[0]):
            index = int(x[i])
            try:
                r[i] = self.integrate3[index]
            except IndexError:
                r[i] = self.integrate3[-1]

        return r
