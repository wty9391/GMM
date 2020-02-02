import pickle
import sys

import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error

from myutil.dataset_processor import Censored_processor


def cdf_to_pdf(cdf):
    return [cdf[i + 1] - cdf[i] if cdf[i + 1] - cdf[i] > 0 else 1e-6 for i in range(0, len(cdf) - 1)]


# ../result/1458
if len(sys.argv) < 2:
    print('Usage: .py result_root_path')
    exit(-1)
    
    
bidder = pickle.load(open(sys.argv[1] + '/truthful_bidder', 'rb'))
x = pickle.load(open(sys.argv[1] + '/x_train', 'rb'))
z = pickle.load(open(sys.argv[1] + '/z_train', 'rb'))
x_test = pickle.load(open(sys.argv[1] + '/x_test', 'rb'))
z_test = pickle.load(open(sys.argv[1] + '/z_test', 'rb'))
m_test, _ = z_test.shape

processor = Censored_processor()
processor.load(x, z, bidder)
processor_test = Censored_processor()
processor_test.load(x_test, z_test, bidder)

zs = list(range(processor.min_price, processor.max_price + 1))
truth_pdf = [processor.truth["pdf"][z] if processor.truth["pdf"][z] != 0 else 1e-6 for z in zs]
truth_cdf = [processor.truth["cdf"][z] if processor.truth["cdf"][z] != 0 else 1e-6 for z in zs]
survive_pdf = [processor.survive["pdf"][z] if processor.survive["pdf"][z] != 0 else 1e-6 for z in zs]
survive_cdf = [processor.survive["cdf"][z] if processor.survive["cdf"][z] != 0 else 1e-6 for z in zs]
survive_expected_z = np.round(sum([z * processor.survive["pdf"][z] for z in zs]))

pdf_test = [processor_test.truth["pdf"][z] if processor_test.truth["pdf"][z] != 0 else 1e-6 for z in zs]
cdf_test = [processor_test.truth["cdf"][z] if processor_test.truth["cdf"][z] != 0 else 1e-6 for z in zs]


test_WD_survive_pdf = wasserstein_distance(pdf_test, survive_pdf)
test_KL_survive_pdf = entropy(pdf_test, survive_pdf)
test_WD_survive_cdf = wasserstein_distance(cdf_test, survive_cdf)
test_KL_survive_cdf = entropy(cdf_test, survive_cdf)

test_MSE_survive = mean_squared_error(z_test, [survive_expected_z] * m_test)
test_ANLP_survive = sum(
    [-np.log(processor.survive["pdf"][z[0]]) if z[0] > 0 else -np.log(processor.survive["pdf"][1]) for z in
     z_test.tolist()]) / m_test


print("Algorithm\tKL_pdf\tWD_pdf\tKL_cdf\tWD_cdf\tMSE\tANLP")
print("SM\t{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}\t{5:.6f}"
      .format(test_KL_survive_pdf, test_WD_survive_pdf, test_KL_survive_cdf, test_WD_survive_cdf, test_MSE_survive,
              test_ANLP_survive))










