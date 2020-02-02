import sys
import math
import pickle
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error

import CLR as mixture_model
from myutil.dataset_processor import Censored_processor

# ../result/1458 1

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

processor_train = Censored_processor()
processor_train.load(x_train, z_train, bidder)
processor_test = Censored_processor()
processor_test.load(x_test, z_test, bidder)

model = mixture_model.MixtureModel(feature_size, bidder, variance=5)
model.fit(z_train, x_train, epoch=50, batch_size=512, eta_w=5e-4, verbose=1)


zs = list(range(processor_train.min_price, processor_train.max_price))

pdf_test = [processor_test.truth["pdf"][z] if processor_test.truth["pdf"][z] != 0 else 1e-6 for z in zs]
cdf_test = [processor_test.truth["cdf"][z] if processor_test.truth["cdf"][z] != 0 else 1e-6 for z in zs]

lm_predict = np.round(model.lm_predict(x_test)[:, 0]).tolist()
clm_predict = np.round(model.clm_predict(x_test)[:, 0]).tolist()
mm_predict = np.round(model.predict(x_test)[:, 0]).tolist()

lm_result = processor_train._count(lm_predict)
clm_result = processor_train._count(clm_predict)
mm_result = processor_train._count(mm_predict)

pdf_lm = [lm_result["pdf"][z] if lm_result["pdf"][z] != 0 else 1e-6 for z in zs]
cdf_lm = [lm_result["cdf"][z] if lm_result["cdf"][z] != 0 else 1e-6 for z in zs]
pdf_clm = [clm_result["pdf"][z] if clm_result["pdf"][z] != 0 else 1e-6 for z in zs]
cdf_clm = [clm_result["cdf"][z] if clm_result["cdf"][z] != 0 else 1e-6 for z in zs]
pdf_mm = [mm_result["pdf"][z] if mm_result["pdf"][z] != 0 else 1e-6 for z in zs]
cdf_mm = [mm_result["cdf"][z] if mm_result["cdf"][z] != 0 else 1e-6 for z in zs]

test_WD_lm_cdf = wasserstein_distance(cdf_test, cdf_lm)
test_KL_lm_cdf = entropy(cdf_test, cdf_lm)
test_WD_lm_pdf = wasserstein_distance(pdf_test, pdf_lm)
test_KL_lm_pdf = entropy(pdf_test, pdf_lm)

test_WD_clm_cdf = wasserstein_distance(cdf_test, cdf_clm)
test_KL_clm_cdf = entropy(cdf_test, cdf_clm)
test_WD_clm_pdf = wasserstein_distance(pdf_test, pdf_clm)
test_KL_clm_pdf = entropy(pdf_test, pdf_clm)

test_WD_mm_cdf = wasserstein_distance(cdf_test, cdf_mm)
test_KL_mm_cdf = entropy(cdf_test, cdf_mm)
test_WD_mm_pdf = wasserstein_distance(pdf_test, pdf_mm)
test_KL_mm_pdf = entropy(pdf_test, pdf_mm)

test_MSE_lm = mean_squared_error(z_test, lm_predict)
test_MSE_clm = mean_squared_error(z_test, clm_predict)
test_MSE_mm = mean_squared_error(z_test, mm_predict)

test_ANLP_mm = model.anlp(x_test, z_test)

print("Algorithm\t cdf_test_KL\t cdf_test_WD\t pdf_test_KL\t pdf_test_WD\t test_MSE\t test_ANLP")
print("lm\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t {4:.6f}\t NA"
      .format(test_KL_lm_cdf, test_WD_lm_cdf, test_KL_lm_pdf, test_WD_lm_pdf, test_MSE_lm))
print("clm\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t {4:.6f}\t NA"
      .format(test_KL_clm_cdf, test_WD_clm_cdf, test_KL_clm_pdf, test_WD_clm_pdf, test_MSE_clm))
print("CLR\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t {4:.6f}\t {5:.6f}"
      .format(test_KL_mm_cdf, test_WD_mm_cdf, test_KL_mm_pdf, test_WD_mm_pdf, test_MSE_mm, test_ANLP_mm))