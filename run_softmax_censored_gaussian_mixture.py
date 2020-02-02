import sys
import math
import pickle
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error

import myutil.init_util as init_util
from myutil.dataset_processor import Censored_processor
from softmax_censored_gaussian_mixture import SoftmaxCensoredGaussianMixture

# ./result/1458 0
if len(sys.argv) < 3:
    print('Usage: .py result_root_path plot_figure')
    exit(-1)

plot_figure = sys.argv[2]

bidder = pickle.load(open(sys.argv[1] + '/truthful_bidder', 'rb'))
x_train = pickle.load(open(sys.argv[1] + '/x_train', 'rb'))
z_train = pickle.load(open(sys.argv[1] + '/z_train', 'rb'))
x_test = pickle.load(open(sys.argv[1] + '/x_test', 'rb'))
z_test = pickle.load(open(sys.argv[1] + '/z_test', 'rb'))

(record_size, feature_size) = np.shape(x_train)

_, mean, variance = init_util.init_gaussian_mixture_parameter(10, min_z=1, max_z=250)
model = SoftmaxCensoredGaussianMixture(bidder, feature_size, len(mean), mean=mean, variance=variance)
model.fit(z_train, x_train, sample_rate=0.5, epoch=30, batch_size=1024, eta_w=5e-1, eta_mean=3e1, eta_variance=3e2, labda=0.0, verbose=1)
print(model)

processor_train = Censored_processor()
processor_train.load(x_train, z_train, bidder)
processor_test = Censored_processor()
processor_test.load(x_test, z_test, bidder)

zs = list(range(processor_train.min_price, processor_train.max_price))
pdf_train_truth = [processor_train.truth["pdf"][z] for z in zs]
cdf_train_truth = [processor_train.truth["cdf"][z] if processor_train.truth["cdf"][z] != 0 else 1e-6 for z in zs]
pdf_train_win = [processor_train.win["pdf"][z] for z in zs]
cdf_train_win = [processor_train.win["cdf"][z] for z in zs]
pdf_train_lose = [processor_train.lose["pdf"][z] for z in zs]
cdf_train_lose = [processor_train.lose["cdf"][z] for z in zs]
pdf_test = [processor_test.truth["pdf"][z] for z in zs]
cdf_test = [processor_test.truth["cdf"][z] if processor_test.truth["cdf"][z] != 0 else 1e-6 for z in zs]

pdf_train_survive = [processor_train.survive["pdf"][z] for z in zs]
cdf_train_survive = [processor_train.survive["cdf"][z] if processor_train.survive["cdf"][z] != 0 else 1e-6 for z in zs]

pdf_train_mix = model.pdf_overall(zs, x_train)
cdf_train_mix = model.cdf_overall(zs, x_train)
pdf_test_mix = model.pdf_overall(zs, x_test)
cdf_test_mix = model.cdf_overall(zs, x_test)

train_WD_CGM = wasserstein_distance(cdf_train_truth, cdf_train_mix)
train_KL_CGM = entropy(cdf_train_truth, cdf_train_mix)
test_WD_CGM = wasserstein_distance(cdf_test, cdf_test_mix)
test_KL_CGM = entropy(cdf_test, cdf_test_mix)
train_MSE_CGM = mean_squared_error(z_train, model.predict_z(x_train))
test_MSE_CGM = mean_squared_error(z_test, model.predict_z(x_test))
train_ANLP_CGM = model.ANLP(z_train, x_train)
test_ANLP_CGM = model.ANLP(z_test, x_test)

train_WD_survive = wasserstein_distance(cdf_train_truth, cdf_train_survive)
train_KL_survive = entropy(cdf_train_truth, cdf_train_survive)
test_WD_survive = wasserstein_distance(cdf_test, cdf_train_survive)
test_KL_survive = entropy(cdf_test, cdf_train_survive)

train_WD_full = wasserstein_distance(cdf_train_truth, cdf_train_truth)
train_KL_full = entropy(cdf_train_truth, cdf_train_truth)
test_WD_full = wasserstein_distance(cdf_test, cdf_train_truth)
test_KL_full = entropy(cdf_test, cdf_train_truth)

print("Algorithm\t train_KL\t train_WD\t test_KL\t testWD\ttrain_MSE\ttest_MSE\ttrain_ANLP\ttest_ANLP")
print("CGM\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t{4:.6f}\t{5:.6f}\t{6:.6f}\t{7:.6f}"
      .format(train_KL_CGM, train_WD_CGM, test_KL_CGM, test_WD_CGM, train_MSE_CGM,
              test_MSE_CGM, train_ANLP_CGM, test_ANLP_CGM))
print("survive\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t"
      .format(train_KL_survive, train_WD_survive, test_KL_survive, test_WD_survive))
print("full\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t"
      .format(train_KL_full, train_WD_full, test_KL_full, test_WD_full))

pickle.dump(model, open(sys.argv[1]+'/CGMM', 'wb'))
