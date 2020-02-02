from .BASE_MODEL import BASE_RNN
import sys
import os
import random

from scipy.stats import entropy,wasserstein_distance
from sklearn.metrics import mean_squared_error
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = random.choice(['0', '1', '2', '3'])


#default parameter
FEATURE_SIZE = 16 # dataset input fields count
MAX_DEN = 580000 # max input data demension
EMB_DIM = 32
BATCH_SIZE = 300
MAX_SEQ_LEN = 330
TRAING_STEPS = 10000000
STATE_SIZE = 256
GRAD_CLIP = 5.0
L2_NORM = 0.001
ADD_TIME = True
ALPHA = 1.2 # coefficient for cross entropy
BETA = 0.2 # coefficient for anlp
# input_file="3476" #toy dataset

# 0.0001 2997
if len(sys.argv) < 3:
    print("Please input learning rate and campaign")
    sys.exit(0)

LR = float(sys.argv[1])
input_file = sys.argv[2]
LR_ANLP = LR
RUNNING_MODEL = BASE_RNN(EMB_DIM=EMB_DIM,
                         FEATURE_SIZE=FEATURE_SIZE,
                         BATCH_SIZE=BATCH_SIZE,
                         MAX_DEN=MAX_DEN,
                         MAX_SEQ_LEN=MAX_SEQ_LEN,
                         TRAING_STEPS=TRAING_STEPS,
                         STATE_SIZE=STATE_SIZE,
                         LR=LR,
                         GRAD_CLIP=GRAD_CLIP,
                         L2_NORM=L2_NORM,
                         INPUT_FILE=input_file,
                         ALPHA=ALPHA,
                         BETA=BETA,
                         ADD_TIME_FEATURE=ADD_TIME,
                         FIND_PARAMETER=False,
                         ANLP_LR=LR,
                         DNN_MODEL=False,
                         DISCOUNT=1,
                         ONLY_TRAIN_ANLP=False,
                         LOG_PREFIX="dlf")
RUNNING_MODEL.create_graph()
_,_,_,a,z = RUNNING_MODEL.run_model()
ps = [p[2]-p[1] for p in a]

min_price = 0
max_price = 300
pdf_truth = {}
cdf_truth = {}
pdf_model = {}
cdf_model = {}
ps_sum = 0.0
market_price = RUNNING_MODEL.test_data.market_price.tolist()
predicted_market_price = []

integral = np.array(ps) * np.array(z)

for i in range(len(market_price)):
    price = RUNNING_MODEL.test_data.market_price[i]
    if price in pdf_truth:
        pdf_truth[price] += 1.0
    else:
        pdf_truth[price] = 1.0
    predicted_market_price.append(np.sum(integral[i*300:(i+1)*300]))

for i in range(len(z)):
    price = z[i]
    if price in pdf_model:
        pdf_model[price] += ps[i]
    else:
        pdf_model[price] = ps[i]

for price in range(min_price, max_price + 1):
    if price not in pdf_truth:
        pdf_truth[price] = 1.0
    if price not in pdf_model:
        pdf_model[price] = 1e-12

for price in pdf_truth:
    pdf_model[price] = pdf_model[price] / sum(ps)
    pdf_truth[price] = pdf_truth[price] / len(market_price)


print("sum ps: ", sum(ps))
print("average ps: ", sum(ps)*300/len(ps))

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


print("Algorithm\ttest_KL_pdf\ttest_WD_pdf\ttest_KL_cdf\ttest_WD_cdf\ttest_MSE")
print("DLF\t{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}"
      .format(KL_pdf, WD_pdf, KL_cdf, WD_cdf, MSE))

