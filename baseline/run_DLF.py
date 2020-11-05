from BASE_MODEL import BASE_RNN
import sys
import os
import random
import pandas as pd

from scipy.stats import entropy,wasserstein_distance
from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = random.choice(['0', '1', '2', '3'])


#default parameter
FEATURE_SIZE = 16  # dataset input fields count
MAX_DEN = 580000  # max input data demension
EMB_DIM = 32
BATCH_SIZE = 300
MAX_SEQ_LEN = 330
TRAING_STEPS = 1000000
STATE_SIZE = 300
GRAD_CLIP = 5.0
L2_NORM = 0.001
ADD_TIME = True
ALPHA = 1.2  # coefficient for cross entropy
BETA = 0.2  # coefficient for anlp
# input_file="3476" #toy dataset

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
min_price = 1
max_price = 300
price_range_size = max_price - min_price + 1
step = 5

RUNNING_MODEL.create_graph()
_,_,_,a,z,truth_z = RUNNING_MODEL.run_model(min_z=min_price, max_z=max_price, step=step)

ps = [p[2]-p[1] for p in a]
pdf_truth = {}
cdf_truth = {}
pdf_model = {}
cdf_model = {}
ps_sum = 0.0
# market_price = RUNNING_MODEL.test_data.market_price.tolist()
predicted_market_price = []
nlp = []

integral = np.array(ps) * np.array(z)

for i in range(len(truth_z)):
    price = truth_z[i]
    if price in pdf_truth:
        pdf_truth[price] += 1.0
    else:
        pdf_truth[price] = 1.0
    predicted_market_price.append(np.sum(integral[i*price_range_size:(i+1)*price_range_size]))
    # maybe need normalization
    nlp.append(-np.log(ps[i * price_range_size + price - min_price]))

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
    pdf_truth[price] = pdf_truth[price] / len(truth_z)


print("sum ps: ", sum(ps))
print("average ps: ", sum(ps)*price_range_size/len(ps))

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


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(zs, [pdf_truth[z] for z in zs], color="#01ff07", label='truth')
ax1.plot(zs, [pdf_model[z] for z in zs], color="#4b006e", label='DLF')
ax2.plot(zs, [cdf_truth[z] for z in zs], color="#01ff07", label='truth')
ax2.plot(zs, [cdf_model[z] for z in zs], color="#4b006e", label='DLF')

ax1.legend()
ax1.set_ylabel("pdf", fontsize=15)
ax1.set_xlabel("market price", fontsize=12)
ax1.tick_params(labelsize=10)
ax2.legend()
ax2.set_ylabel("cdf", fontsize=15)
ax2.set_xlabel("market price", fontsize=12)
ax2.tick_params(labelsize=10)
fig = plt.gcf()
fig.set_size_inches(6, 5)
plt.tight_layout()
plt.savefig('./market_price_distribution_' + input_file + '.pdf', format='pdf')


WD_pdf = wasserstein_distance([pdf_truth[z] for z in zs], [pdf_model[z] for z in zs])
KL_pdf = entropy([pdf_truth[z] for z in zs], [pdf_model[z] for z in zs])
WD_cdf = wasserstein_distance([cdf_truth[z] for z in zs], [cdf_model[z] for z in zs])
KL_cdf = entropy([cdf_truth[z] for z in zs], [cdf_model[z] for z in zs])
MSE = mean_squared_error(truth_z, predicted_market_price)
anlp = sum(nlp) / len(truth_z)

print("Algorithm\ttest_KL_pdf\ttest_WD_pdf\ttest_KL_cdf\ttest_WD_cdf\ttest_MSE\ttest_ANLP")
print("DLF\t{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}\t{5:.6f}"
      .format(KL_pdf, WD_pdf, KL_cdf, WD_cdf, MSE, anlp))

total_record = RUNNING_MODEL.train_data.size
win_record = RUNNING_MODEL.train_data.winData.size
win_rate = win_record*1.0 / total_record*1.0

df = pd.DataFrame(data=[[input_file, total_record, win_record, win_rate,
                        'DLF', KL_pdf, WD_pdf, anlp, MSE]],
                  columns=['campaign', 'total_record', 'win_record', 'win_rate',
                           'algorithm', 'KL_pdf', 'WD_pdf', 'ANLP', 'MSE'])
output_path = '../baseline_report.csv'
df.to_csv(output_path, mode='a', index=False, sep='\t', header=not os.path.exists(output_path))

