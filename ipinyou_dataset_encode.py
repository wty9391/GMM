import sys
import pickle
import shutil
import numpy as np
from scipy.sparse import csr_matrix, vstack

import myutil.encoder as encoder
import myutil.truthful_bidder as truthful_bidder

# ../make-ipinyou-data/1458/train.log.txt ../make-ipinyou-data/1458/test.log.txt ../make-ipinyou-data/1458/featindex.txt ./result/1458
# ../make-ipinyou-data/2259/train.log.txt ../make-ipinyou-data/2259/test.log.txt ../make-ipinyou-data/2259/featindex.txt ./result/2259
# ../make-ipinyou-data/2261/train.log.txt ../make-ipinyou-data/2261/test.log.txt ../make-ipinyou-data/2261/featindex.txt ./result/2261
# ../make-ipinyou-data/2821/train.log.txt ../make-ipinyou-data/2821/test.log.txt ../make-ipinyou-data/2821/featindex.txt ./result/2821
# ../make-ipinyou-data/3358/train.log.txt ../make-ipinyou-data/3358/test.log.txt ../make-ipinyou-data/3358/featindex.txt ./result/3358
# ../make-ipinyou-data/3427/train.log.txt ../make-ipinyou-data/3427/test.log.txt ../make-ipinyou-data/3427/featindex.txt ./result/3427
# ../make-ipinyou-data/3476/train.log.txt ../make-ipinyou-data/3476/test.log.txt ../make-ipinyou-data/3476/featindex.txt ./result/3476
# ../make-ipinyou-data/all/train.log.txt ../make-ipinyou-data/all/test.log.txt ../make-ipinyou-data/all/featindex.txt ./result/all
#
if len(sys.argv) < 5:
    print('Usage: .py trian_log_path test_log_path feat_path result_root_path')
    exit(-1)

read_batch_size = 1e6

f_train_log = open(sys.argv[1], 'r', encoding="utf-8")
f_test_log = open(sys.argv[2], 'r', encoding="utf-8")

f_train_yzbx = open(sys.argv[4]+'/train.yzbx.txt', 'w+', encoding="utf-8")  # for DLF
f_test_yzbx = open(sys.argv[4]+'/test.yzbx.txt', 'w+', encoding="utf-8")  # for DLF

shutil.copyfile(sys.argv[3], sys.argv[4]+'/featindex.txt')

# init name_col
name_col = {}
s = f_train_log.readline().split('\t')
for i in range(0, len(s)):
    name_col[s[i].strip()] = i

ipinyou = encoder.Encoder_ipinyou(sys.argv[3], name_col)
X_train_raw = []
X_train = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_train = np.zeros((0, 1), dtype=np.int8)
B_train = np.zeros((0, 1), dtype=np.int16)
Z_train = np.zeros((0, 1), dtype=np.int16)
X_test_raw = []
X_test = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_test = np.zeros((0, 1), dtype=np.int8)
B_test = np.zeros((0, 1), dtype=np.int16)
Z_test = np.zeros((0, 1), dtype=np.int16)

count = 0
f_train_log.seek(0)
f_train_log.readline()  # first line is header
for line in f_train_log:
    X_train_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
        Y_train = np.vstack((Y_train, ipinyou.get_col(X_train_raw, "click")))
        Z_train = np.vstack((Z_train, ipinyou.get_col(X_train_raw, "payprice")))
        X_train_raw = []
if X_train_raw:
    X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
    Y_train = np.vstack((Y_train, ipinyou.get_col(X_train_raw, "click")))
    Z_train = np.vstack((Z_train, ipinyou.get_col(X_train_raw, "payprice")))
    X_train_raw = []

count = 0
f_test_log.seek(0)
f_test_log.readline()  # first line is header
for line in f_test_log:
    X_test_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
        Y_test = np.vstack((Y_test, ipinyou.get_col(X_test_raw, "click")))
        Z_test = np.vstack((Z_test, ipinyou.get_col(X_test_raw, "payprice")))
        X_test_raw = []
if X_test_raw:
    X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
    Y_test = np.vstack((Y_test, ipinyou.get_col(X_test_raw, "click")))
    Z_test = np.vstack((Z_test, ipinyou.get_col(X_test_raw, "payprice")))
    X_test_raw = []


bidder = truthful_bidder.Truthful_bidder()
bidder.fit(X_train, Y_train, Z_train)
bidder.evaluate(X_test, Y_test)
B_train = np.asmatrix(bidder.bid(X_train)).transpose()
B_test = np.asmatrix(bidder.bid(X_test)).transpose()

for i in range(X_train.shape[0]):
    f_train_yzbx.write(str(Y_train[i, 0]) + " ")
    f_train_yzbx.write(str(Z_train[i, 0]) + " ")
    f_train_yzbx.write(str(int(B_train[i, 0])) + " ")
    f_train_yzbx.write(":1 ".join("{0:d}".format(n) for n in X_train.getrow(i).indices.tolist()[0:16]))  # The length of x in DLF is fixed
    f_train_yzbx.write(":1\n")

for i in range(X_test.shape[0]):
    f_test_yzbx.write(str(Y_test[i, 0]) + " ")
    f_test_yzbx.write(str(Z_test[i, 0]) + " ")
    f_test_yzbx.write(str(int(B_test[i, 0])) + " ")
    f_test_yzbx.write(":1 ".join("{0:d}".format(n) for n in X_test.getrow(i).indices.tolist()[0:16]))  # The length of x in DLF is fixed
    f_test_yzbx.write(":1\n")

pickle.dump(X_train, open(sys.argv[4]+'/x_train', 'wb'))
pickle.dump(Y_train, open(sys.argv[4]+'/y_train', 'wb'))
pickle.dump(B_train, open(sys.argv[4]+'/b_train', 'wb'))
pickle.dump(Z_train, open(sys.argv[4]+'/z_train', 'wb'))
pickle.dump(X_test, open(sys.argv[4]+'/x_test', 'wb'))
pickle.dump(Y_test, open(sys.argv[4]+'/y_test', 'wb'))
pickle.dump(Z_test, open(sys.argv[4]+'/z_test', 'wb'))
pickle.dump(bidder, open(sys.argv[4]+'/truthful_bidder', 'wb'))

f_train_log.close()
f_test_log.close()
f_train_yzbx.close()
f_test_yzbx.close()


