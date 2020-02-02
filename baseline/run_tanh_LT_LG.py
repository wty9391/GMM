import sys
import pickle
import numpy as np
from scipy.stats import norm
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error

from baseline.tanh_LT_LG import wining_pridictor
from myutil.dataset_processor import Processor


def cdf_to_pdf(cdf):
    return [cdf[i + 1] - cdf[i] if cdf[i + 1] - cdf[i] > 0 else 1e-6 for i in range(0, len(cdf) - 1)]


# ../1458
if len(sys.argv) < 2:
    print('Usage: .py result_root_path')
    exit(-1)

epsilon = sys.float_info.epsilon


z = pickle.load(open(sys.argv[1] + '/z_train', 'rb'))
z_test = pickle.load(open(sys.argv[1] + '/z_test', 'rb'))
m_test, _ = z_test.shape


processor = Processor()
processor.load_by_array(z)
processor_test = Processor()
processor_test.load_by_array(z_test)

zs = list(range(processor.min_price, processor.max_price + 1))
zs_arange = np.arange(processor.min_price, processor.max_price + 1, 1)

pdf_train = [processor.pdf[z] if processor.pdf[z] != 0 else 1e-6 for z in zs]
cdf_train = [processor.cdf[z] for z in zs]

pdf_test = [processor_test.pdf[z] if processor_test.pdf[z] != 0 else 1e-6 for z in zs]
cdf_test = [processor_test.cdf[z] if processor_test.cdf[z] != 0 else 1e-6 for z in zs]

tanh = wining_pridictor(d=0.01, max_iter=500, eta=1e-3, pattern="tanh")
long_tail = wining_pridictor(d=30,max_iter=2000,eta=1e3,pattern="long_tail")
log_gaussian = wining_pridictor(pattern="log_gaussian")

tanh.fit1(zs_arange, cdf_train)
long_tail.fit2(zs_arange, cdf_train)
log_gaussian.fit3(np.array(processor.data))

tanh_cdf = tanh.predict(zs_arange)
long_tail_cdf = long_tail.predict(zs_arange)
log_gaussian_cdf = log_gaussian.predict(zs_arange)
tanh_pdf = cdf_to_pdf(tanh_cdf) + [1e-6]
long_tail_pdf = cdf_to_pdf(long_tail_cdf) + [1e-6]
log_gaussian_pdf = cdf_to_pdf(log_gaussian_cdf) + [1e-6]

tanh_expected_z = sum([z * p for (z, p) in zip(zs, tanh_pdf)])
long_tail_expected_z = sum([z * p for (z, p) in zip(zs, long_tail_pdf)])
log_gaussian_expected_z = sum([z * p for (z, p) in zip(zs, log_gaussian_pdf)])

pdf_test_WD_tanh = wasserstein_distance(pdf_test, tanh_pdf)
pdf_test_KL_tanh = entropy(pdf_test, tanh_pdf)
cdf_test_WD_tanh = wasserstein_distance(cdf_test, tanh_cdf)
cdf_test_KL_tanh = entropy(cdf_test, tanh_cdf)
test_MSE_tanh = mean_squared_error(z_test, [tanh_expected_z] * m_test)
test_ANLP_tanh = sum([-np.log(tanh_pdf[z[0] - processor_test.min_price]) for z in z_test.tolist()]) / m_test

pdf_test_WD_LT = wasserstein_distance(pdf_test, long_tail_pdf)
pdf_test_KL_LT = entropy(pdf_test, long_tail_pdf)
cdf_test_WD_LT = wasserstein_distance(cdf_test, long_tail_cdf)
cdf_test_KL_LT = entropy(cdf_test, long_tail_cdf)
test_MSE_LT = mean_squared_error(z_test, [long_tail_expected_z] * m_test)
test_ANLP_LT = sum([-np.log(long_tail_pdf[z[0] - processor_test.min_price]) for z in z_test.tolist()]) / m_test

pdf_test_WD_LG = wasserstein_distance(pdf_test, log_gaussian_pdf)
pdf_test_KL_LG = entropy(pdf_test, log_gaussian_pdf)
cdf_test_WD_LG = wasserstein_distance(cdf_test, log_gaussian_cdf)
cdf_test_KL_LG = entropy(cdf_test, log_gaussian_cdf)
test_MSE_LG = mean_squared_error(z_test, [log_gaussian_expected_z] * m_test)
test_ANLP_LG = sum([-np.log(log_gaussian_pdf[z[0] - processor_test.min_price]) for z in z_test.tolist()]) / m_test

print("Algorithm\tKL_pdf\tWD_pdf\tKL_cdf\tWD_cdf\tMSE\tANLP")
print("tanh\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t {4:.6f}\t {5:.6f}"
      .format(pdf_test_KL_tanh, pdf_test_WD_tanh, cdf_test_KL_tanh, cdf_test_WD_tanh, test_MSE_tanh, test_ANLP_tanh))
print("LT\t {0:.6f}\t {1:.6f}\t {2:.6f}\t {3:.6f}\t {4:.6f}\t {5:.6f}"
      .format(pdf_test_KL_LT, pdf_test_WD_LT, cdf_test_KL_LT, cdf_test_WD_LT, test_MSE_LT, test_ANLP_LT))
print("LG\t {0:.6f}\t {1:.6f} {2:.6f}\t {3:.6f}\t {4:.6f}\t {5:.6f}"
      .format(pdf_test_KL_LG, pdf_test_WD_LG, cdf_test_KL_LG, cdf_test_WD_LG, test_MSE_LG, test_ANLP_LG))











