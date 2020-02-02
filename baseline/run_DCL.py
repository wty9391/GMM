import tensorflow as tf
import numpy as np
import os
import time
from sklearn.metrics import *
from myutil import *
import sys
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy,wasserstein_distance
import random

class DWPP:
    def __init__(self, lr, batch_size, dimension, util_train, util_test, campaign, reg_lambda, sigma):
        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.util_train = util_train
        self.util_test = util_test
        self.reg_lambda = reg_lambda
        self.emb_size = 20

        self.train_data_amt = util_train.get_data_amt()
        self.test_data_amt = util_test.get_data_amt()

        # output dir
        self.model_name = "{}_{}_{}".format(self.lr, self.reg_lambda, self.batch_size)
        self.output_dir = "output/dwpp/{}/{}/".format(campaign, self.model_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # reset graph
        tf.reset_default_graph()

        # field params
        # TODO  modified by wty
        # self.field_sizes = self.util_train.feat_sizes
        self.field_sizes = self.util_train.nonzero_feat_sizes
        self.field_num = self.util_train.field_size

        # placeholders
        self.X = [tf.sparse_placeholder(tf.float32, name='x_%d'%i) for i in range(0, self.field_num)]
        self.z = tf.placeholder(tf.float32, [None, 1])
        self.b = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.all_prices = tf.placeholder(tf.float32, [None, 300])

        # embedding layer
        self.var_map = {}
        # for truncated
        self.var_map['embed_0'] = tf.Variable(
                tf.truncated_normal([self.field_sizes[0], 1], dtype=tf.float32))
        for i in range(1, self.field_num):
            self.var_map['embed_%d' % i] = tf.Variable(
                tf.truncated_normal([self.field_sizes[i], self.emb_size], dtype=tf.float32))
        
        # after embedding
        w0 = [self.var_map['embed_%d' % i] for i in range(self.field_num)]
        self.dense_input = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(self.field_num)], 1)

        self.layer1 = tf.layers.dense(self.dense_input, 50, activation=tf.nn.relu)
        self.layer2 = tf.layers.dense(self.layer1, 30, activation=tf.nn.relu)
        self.u = tf.layers.dense(self.layer2, 1, activation=tf.nn.relu)
        self.sigma = 1#tf.get_variable('sigma', [], dtype=tf.float32)
        
        self.pz = tf.exp(-(self.z-self.u)*(self.z-self.u)/(2*self.sigma*self.sigma)) / self.sigma
        self.p_all = tf.exp(-(self.all_prices-self.u)*(self.all_prices-self.u)/(2*self.sigma*self.sigma)) / self.sigma
        self.w_all = tf.cumsum(self.p_all, axis=1)
        idx_b = tf.stack([tf.reshape(tf.range(tf.shape(self.b)[0]), (-1,1)), tf.cast(self.b - 1, tf.int32)], axis=-1)
        self.wb = tf.gather_nd(self.w_all, idx_b)

        self.loss = tf.losses.mean_squared_error(self.z*self.y, self.u*self.y) + tf.losses.mean_squared_error(self.wb*(1-self.y), tf.zeros_like(self.wb)*(1-self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        # session initialization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)

    def train(self):
        step = 0
        epoch = 0
        batch_loss = []
        loss_list = []

        while True:
            x_batch_field, b_batch, z_batch, y_batch, all_prices = self.util_train.get_batch_data_sorted_dwpp(step)
            feed_dict = {}
            for j in range(len(self.X)):
                feed_dict[self.X[j]] = tf.SparseTensorValue(x_batch_field[j], [1] * len(x_batch_field[j]),
                                                                  [self.batch_size, self.field_sizes[j]])
            feed_dict[self.b] = b_batch
            feed_dict[self.z] = z_batch
            feed_dict[self.y] = y_batch
            feed_dict[self.all_prices] = all_prices
            
            batch_loss.append(self.sess.run(self.loss, feed_dict))
            self.sess.run(self.train_step, feed_dict)
            batch_loss.append(self.sess.run(self.loss, feed_dict))
            step += 1

            if step * self.batch_size - epoch * int(0.1 * self.train_data_amt) >= int(0.1 * self.train_data_amt):
                loss = np.mean(batch_loss[step - int(int(0.1 * self.train_data_amt) / self.batch_size) - 1:])
                loss_list.append(loss)
                print("train loss of epoch-{0} is {1}".format(epoch, loss))
                epoch += 1

            # stop condition
            if epoch * 0.1 * self.train_data_amt <= 3 * self.train_data_amt:
                continue
            if (loss_list[-1] - loss_list[-2] > 0 and loss_list[-2] - loss_list[-3] > 0):
                break
            if epoch * 0.1 * self.train_data_amt >= 5 * self.train_data_amt:
                break
    
    def test(self):
        batch_num = int(self.test_data_amt / self.batch_size)
        anlp_batch = []
        auc_batch = []
        logloss_batch = []
        pzs = []
        wbs = []
        ys = []
        predicted_market_price = []
        market_price = []
        for i in range(batch_num):
            x_batch_field, b_batch, z_batch, y_batch, all_prices = self.util_test.get_batch_data_sorted_dwpp(i)
            feed_dict = {}
            for j in range(len(self.X)):
                feed_dict[self.X[j]] = tf.SparseTensorValue(x_batch_field[j], [1] * len(x_batch_field[j]),
                                                                  [self.batch_size, self.field_sizes[j]])
            feed_dict[self.b] = b_batch
            feed_dict[self.z] = z_batch
            feed_dict[self.y] = y_batch
            feed_dict[self.all_prices] = all_prices
            ys += y_batch.reshape(-1,).tolist()
            pz, wb, z_predict = self.sess.run([self.pz, self.wb, self.u], feed_dict)
            
            # print(self.sess.run(self.u, feed_dict))
            # print(self.sess.run(self.z-self.u, feed_dict))
            # print(self.sess.run(self.y, feed_dict))
            # break
            pz[pz == 0] = 1e-20
            pzs += pz.reshape(-1,).tolist()
            wbs += wb.reshape(-1,).tolist()
            predicted_market_price.extend(z_predict.reshape(-1,).tolist())
            market_price.extend(z_batch.reshape(-1,).tolist())
            
        ANLP = np.average(-np.log(pzs))
        AUC = roc_auc_score(ys, wbs)
        LOGLOSS = log_loss(ys, wbs)

        min_price = 0
        max_price = 300
        pdf_truth = {}
        cdf_truth = {}
        pdf_model = {}
        cdf_model = {}

        for i in range(len(market_price)):
            price = market_price[i]
            if price in pdf_truth:
                pdf_truth[price] += 1.0
            else:
                pdf_truth[price] = 1.0

        for i in range(len(predicted_market_price)):
            price = predicted_market_price[i]
            if price in pdf_model:
                pdf_model[price] += 1.0
            else:
                pdf_model[price] = 1.0

        for price in range(min_price, max_price + 1):
            if price not in pdf_truth:
                pdf_truth[price] = 1.0
            if price not in pdf_model:
                pdf_model[price] = 1.0

        for price in pdf_truth:
            pdf_model[price] = pdf_model[price] / len(predicted_market_price)
            pdf_truth[price] = pdf_truth[price] / len(market_price)

        for price in pdf_truth:
            p = 0
            for j in pdf_truth:
                p += pdf_truth[j] if j <= price else 0.0
            cdf_truth[price] = p

            p = 0
            for j in pdf_model:
                p += pdf_model[j] if j <= price else 0.0
            cdf_model[price] = p

        zs = list(range(min_price, max_price + 1))
        WD_pdf = wasserstein_distance([pdf_truth[z] for z in zs], [pdf_model[z] for z in zs])
        KL_pdf = entropy([pdf_truth[z] for z in zs], [pdf_model[z] for z in zs])
        WD_cdf = wasserstein_distance([cdf_truth[z] for z in zs], [cdf_model[z] for z in zs])
        KL_cdf = entropy([cdf_truth[z] for z in zs], [cdf_model[z] for z in zs])
        MSE = mean_squared_error(market_price, predicted_market_price)

        print(self.model_name)
        print("Algorithm\ttest_KL_pdf\ttest_WD_pdf\ttest_KL_cdf\ttest_WD_cdf\ttest_MSE\ttest_ANLP")
        print("DCL\t{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}\t{5:.6f}"
              .format(KL_pdf, WD_pdf, KL_cdf, WD_cdf, MSE, ANLP))

        print("AUC: {}".format(AUC))
        print("Log-Loss: {}".format(LOGLOSS))
        print("ANLP: {}".format(ANLP))

        with open(self.output_dir + 'result.txt', 'w') as f:
            f.writelines(["AUC:{}\tANLP:{}\tLog-Loss:{}".format(AUC, ANLP, LOGLOSS)])
            f.writelines(["Algorithm\ttest_KL_pdf\ttest_WD_pdf\ttest_KL_cdf\ttest_WD_cdf\ttest_MSE\ttest_ANLP"])
            f.writelines(["DCL\t{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}\t{5:.6f}"
                         .format(KL_pdf, WD_pdf, KL_cdf, WD_cdf, MSE, ANLP)])


    def output_s(self):
        batch_num = int(self.test_data_amt / self.batch_size)
        output = np.ones([self.batch_size, 300])
        for i in range(batch_num):
            x_batch_field, b_batch, z_batch, y_batch, all_prices = self.util_test.get_batch_data_sorted_dwpp(i)
            feed_dict = {}
            for j in range(len(self.X)):
                feed_dict[self.X[j]] = tf.SparseTensorValue(x_batch_field[j], [1] * len(x_batch_field[j]),
                                                                  [self.batch_size, self.field_sizes[j]])
            feed_dict[self.b] = b_batch
            feed_dict[self.z] = z_batch
            feed_dict[self.y] = y_batch
            feed_dict[self.all_prices] = all_prices
            output = np.vstack([output, self.sess.run(self.w_all, feed_dict)])
        print(output.shape)
        np.savetxt(self.output_dir + 's.txt', 1 - output[self.batch_size:,], delimiter='\t', fmt='%.4f')


if __name__ == '__main__':
    # 1458
    if len(sys.argv) < 2:
        print("Please input campaign")
        sys.exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = random.choice(['0', '1', '2', '3'])
    campaign = sys.argv[1]
    train_file = '../result/' + campaign + '/train.yzbx.txt'
    test_file = '../result/' + campaign + '/test.yzbx.txt'
    feat_index = '../result/' + campaign + '/featindex.txt'

    # hyper parameters
    # lrs = [1e-3, 1e-2, 5e-3]
    lrs = [1e-3]
    batch_sizes = [512]
    # reg_lambdas = [1e-5, 1e-4, 1e-3]
    reg_lambdas = [1e-4]
    sigmas = [1]
    dimension = int(open(feat_index).readlines()[-1].split('\t')[1]) + 1

    params = []

    for lr in lrs:
        for batch_size in batch_sizes:
            for reg_lambda in reg_lambdas:
                for sigma in sigmas:
                    util_train = Util(train_file, feat_index, batch_size, 'train')
                    util_test = Util(test_file, feat_index, batch_size, 'test')
                    params.append([lr, batch_size, util_train, util_test, reg_lambda, sigma])

    # search hyper parameters
    random.shuffle(params)
    for para in params:
        dwpp = DWPP(lr=para[0], batch_size=para[1], dimension=dimension, util_train=para[2], util_test=para[3], campaign=campaign,
                          reg_lambda=para[4], sigma=para[5])
        dwpp.train()
        dwpp.test()
        # dwpp.output_s()
