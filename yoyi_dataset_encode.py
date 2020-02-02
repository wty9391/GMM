import sys
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack

import myutil.encoder as encoder
import myutil.truthful_bidder as truthful_bidder

# ../make-yoyi-data/original-data/sample/train.yzx.txt ../make-yoyi-data/original-data/sample/test.yzx.txt ./result/yoyi_sample


if len(sys.argv) < 4:
    print('Usage: .py trian_log_path test_log_path result_root_path')
    exit(-1)

read_batch_size = 1e6

f_train_log = open(sys.argv[1], 'r', encoding="utf-8")
f_test_log = open(sys.argv[2], 'r', encoding="utf-8")

num_features = 1785106
pay_scale = 1

if sys.argv[3].find("yoyi_sample") > 0:
    num_features = 3036119
    pay_scale = 1e-3


yoyi = encoder.Encoder_yoyi(num_features, pay_scale)
X_train_raw = []
X_train = csr_matrix((0, num_features), dtype=np.int8)
Y_train = np.zeros((0, 1), dtype=np.int8)
Z_train = np.zeros((0, 1), dtype=np.int16)
X_test_raw = []
X_test = csr_matrix((0, num_features), dtype=np.int8)
Y_test = np.zeros((0, 1), dtype=np.int8)
Z_test = np.zeros((0, 1), dtype=np.int16)

count = 0
f_train_log.seek(0)
for line in f_train_log:
    X_train_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_train = vstack((X_train, yoyi.encode(X_train_raw)))
        Y_train = np.vstack((Y_train, yoyi.get_col(X_train_raw, "click")))
        Z_train = np.vstack((Z_train, yoyi.get_col(X_train_raw, "payprice")))
        X_train_raw = []
if X_train_raw:
    X_train = vstack((X_train, yoyi.encode(X_train_raw)))
    Y_train = np.vstack((Y_train, yoyi.get_col(X_train_raw, "click")))
    Z_train = np.vstack((Z_train, yoyi.get_col(X_train_raw, "payprice")))
    X_train_raw = []

count = 0
f_test_log.seek(0)
for line in f_test_log:
    X_test_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_test = vstack((X_test, yoyi.encode(X_test_raw)))
        Y_test = np.vstack((Y_test, yoyi.get_col(X_test_raw, "click")))
        Z_test = np.vstack((Z_test, yoyi.get_col(X_test_raw, "payprice")))
        X_test_raw = []
if X_test_raw:
    X_test = vstack((X_test, yoyi.encode(X_test_raw)))
    Y_test = np.vstack((Y_test, yoyi.get_col(X_test_raw, "click")))
    Z_test = np.vstack((Z_test, yoyi.get_col(X_test_raw, "payprice")))
    X_test_raw = []

# yoyi datasets has much useless features (zero columns) which should be removed to accelerate the learning
feature_count = X_test.sum(axis=0)
nonzero = np.array((feature_count != 0).tolist()[0])
X_train = (X_train.tocsc()[:, nonzero]).tocsr()
X_test = (X_test.tocsc()[:, nonzero]).tocsr()

bidder = truthful_bidder.Truthful_bidder()
bidder.fit(X_train, Y_train, Z_train)
bidder.evaluate(X_test, Y_test)

pickle.dump(X_train, open(sys.argv[3]+'/x_train', 'wb'))
pickle.dump(Y_train, open(sys.argv[3]+'/y_train', 'wb'))
pickle.dump(Z_train, open(sys.argv[3]+'/z_train', 'wb'))
pickle.dump(X_test, open(sys.argv[3]+'/x_test', 'wb'))
pickle.dump(Y_test, open(sys.argv[3]+'/y_test', 'wb'))
pickle.dump(Z_test, open(sys.argv[3]+'/z_test', 'wb'))
pickle.dump(bidder, open(sys.argv[3]+'/truthful_bidder', 'wb'))

f_train_log.close()
f_test_log.close()













