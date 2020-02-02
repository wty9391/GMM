import sys
import math
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error

import myutil.encoder as encoder
from myutil.dataset_processor import Processor
import myutil.init_util as init_util
from softmax_gaussian_mixture import SoftmaxGaussianMixture

# ./result/1458 10 0
if len(sys.argv) < 3:
    print('Usage: train_init.py result_root_path K plot_figure')
    exit(-1)

plot_figure = sys.argv[3]
K = int(sys.argv[2])

read_batch_size = 1e6

X_train = pickle.load(open(sys.argv[1] + '/x_train', 'rb'))
X_test = pickle.load(open(sys.argv[1] + '/x_test', 'rb'))
Z_train = pickle.load(open(sys.argv[1] + '/z_train', 'rb'))
Z_test = pickle.load(open(sys.argv[1] + '/z_test', 'rb'))

train_pay = Processor()
test_pay = Processor()

train_pay.load_by_array(Z_train)
test_pay.load_by_array(Z_test)
train_pay.validate()
test_pay.validate()

_, mean, variance = init_util.init_gaussian_mixture_parameter(K, min_z=1, max_z=50)
model = SoftmaxGaussianMixture(feature_dimension=feature_size, label_dimension=len(mean),
                               mean=mean,
                               variance=variance)
model.fit(np.array(train_pay.data).reshape((record_size, 1)), X_train,
          sample_rate=1, epoch=100, batch_size=8192, softmax_epoch=1, eta=1e0, labda=0e-5, verbose=1)
print(model)

zs = list(range(1, train_pay.max_price))
mix_pdf_train = model.pdf_overall(zs, X_train)
mix_cdf_train = model.cdf_overall(zs, X_train)
mix_pdf_test = model.pdf_overall(zs, X_test)
mix_cdf_test = model.cdf_overall(zs, X_test)
train_dataset_cdf = [train_pay.cdf[z] for z in zs]
test_dataset_cdf = [train_pay.cdf[z] if train_pay.cdf[z] > 0 else 1e-6 for z in zs]

train_WD = wasserstein_distance(train_dataset_cdf, mix_cdf_train)
train_KL = entropy(train_dataset_cdf, mix_cdf_train)
test_WD = wasserstein_distance(test_dataset_cdf, mix_cdf_test)
test_KL = entropy(test_dataset_cdf, mix_cdf_test)

min_z = 1
max_z = 300
if sys.argv[1].find("yoyi") >= 0:
    min_z = 1
    max_z = 1000
train_MSE = mean_squared_error(Z_train, model.predict_z(X_train, min_z=min_z, max_z=max_z))
test_MSE = mean_squared_error(Z_test, model.predict_z(X_test, min_z=min_z, max_z=max_z))
train_ANLP = model.ANLP(Z_train, X_train)
test_ANLP = model.ANLP(Z_test, X_test)


print("KL of cdf in train dataset: ", train_KL)
print("WD of cdf in train dataset: ", train_WD)
print("KL of pdf in train dataset: ", entropy(
    init_util.cdf_to_pdf(train_dataset_cdf),
    init_util.cdf_to_pdf(mix_cdf_train)))
print("WD of pdf in train dataset: ", wasserstein_distance(
    init_util.cdf_to_pdf(train_dataset_cdf),
    init_util.cdf_to_pdf(mix_cdf_train)))
print("MSE in train dataset: ", train_MSE)
print("ANLP in train dataset: ", train_ANLP)
print("KL of cdf in test dataset: ", test_KL)
print("WD of cdf in test dataset: ", test_WD)
print("KL of pdf in test dataset: ", entropy(
    init_util.cdf_to_pdf(test_dataset_cdf),
    init_util.cdf_to_pdf(mix_cdf_test)))
print("WD of pdf in test dataset: ", wasserstein_distance(
    init_util.cdf_to_pdf(test_dataset_cdf),
    init_util.cdf_to_pdf(mix_cdf_test)))
print("MSE in test dataset: ", test_MSE)
print("ANLP in test dataset: ", test_ANLP)

pickle.dump(model, open(sys.argv[1]+'/GMM', 'wb'))

