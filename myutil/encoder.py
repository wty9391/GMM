import numpy as np
from scipy.sparse import csr_matrix
import random
from itertools import repeat

class Encoder_ipinyou:
    def __init__(self, seat_path, name_col, limit=float('Inf')):
        self.seat_path = seat_path
        self.name_col = name_col # feature_name:origin_index
        self.limit = limit
        self.feat = {} # origin_index\:content:encode_index
        
        self.oses = ["windows", "ios", "mac", "android", "linux"]
        self.browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie"]
        #f1s and f1sp are all features to be one hot encoded
        # self.f1s = ["weekday", "hour", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight", "slotvisibility", "slotformat", "creative", "IP"]
        self.f1s = ["weekday", "hour", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight",
                    "slotvisibility", "slotformat", "creative"]
        self.f1sp = ["useragent", "slotprice"]
        self.special = ["usertag"] #modify this makes no change, just write here
        
        f = open(seat_path, 'r')
        
        first = True
        for l in f:
            s = l.split('\t')
            # first line is truncate
            if first:
                first = False
            encode_index = int(s[1])
            self.feat[s[0]] = encode_index
        f.close()
        
    def sparse_encode(self,x,row_index=0):
        # x is one record seperated by \t
        s = x.split('\t')
        row_ind = [row_index]
        col_ind = [0]
        data = [1]
        
        for f in self.f1s: # every direct first order feature
            origin_index = self.name_col[f]
            content = s[origin_index]
            encode_index = self.get_encode_index(origin_index,content)
            
            row_ind.append(row_index)
            col_ind.append(encode_index)
            data.append(1)
        
        for f in self.f1sp:
            origin_index = self.name_col[f]
            content = self.featTrans(f, s[origin_index])
            encode_index = self.get_encode_index(origin_index,content)
            
            row_ind.append(row_index)
            col_ind.append(encode_index)
            data.append(1)
            
        
        origin_index = self.name_col["usertag"]
        tags = self.getTags(s[origin_index])
        for content in tags:
            encode_index = self.get_encode_index(origin_index,content)
            
            row_ind.append(row_index)
            col_ind.append(encode_index)
            data.append(1)
        
        return row_ind,col_ind,data
        
    def encode_one(self,x):
        row_ind,col_ind,data=self.sparse_encode(x)
        return csr_matrix((data,(row_ind,col_ind)),shape=(1,len(self.feat)),dtype=np.int8)
    
    def encode(self,X,ignore=0):
        row_ind_all = []
        col_ind_all = []
        data_all = []
        
        count = 0
        ignore_count = 0
        for line in X:
            if count >= self.limit:
                continue
            
            if ignore_count < ignore:
                ignore_count += 1
                continue
            
            _,col_ind,_ = self.sparse_encode(line,count)
            
            row_ind = (np.ones((len(col_ind),),dtype=np.int8) * count).tolist()
            data = np.ones((len(col_ind),),dtype=np.int8).tolist()
            col_ind_all.extend(col_ind)
            row_ind_all.extend(row_ind)
            data_all.extend(data)
            
            count += 1
            
            # if random.random() < 0.000005:
            #     print("{} records have been encoded".format(count))
              
        print("All {} records have been encoded".format(count))
        return csr_matrix((data_all,(row_ind_all,col_ind_all)),shape=(count,len(self.feat)),dtype=np.int8)

    def get_encode_index(self,origin_index,content):
        feat_index = str(origin_index) + ':' + content
        if feat_index not in self.feat:
            feat_index = str(origin_index) + ':other'
            #print("[{}] is not found, use [{}]".format(str(origin_index) + ':' + content,feat_index))

        return self.feat[feat_index]
    
    def get_labels(self,Y,ignore=0,lable="click"):
        labels_all = []
        
        count = 0
        ignore_count = 0
        for line in Y:
            if count >= self.limit:
                continue
            
            if ignore_count < ignore:
                ignore_count += 1
                continue
            
            s = line.split('\t')
            labels_all.append(s[self.name_col[lable]])
            count += 1
        return np.array(labels_all, dtype=np.int8).reshape((count,1))

    def get_col(self, data, col_name):
        labels_all = []

        count = 0
        for line in data:
            s = line.split('\t')
            labels_all.append(s[self.name_col[col_name]])
            count += 1
        return np.array(labels_all, dtype=np.int16).reshape((count, 1))

    def featTrans(self, name, content):
        content = content.lower()
        if name == "useragent":
            operation = "other"
            for o in self.oses:
                if o in content:
                    operation = o
                    break
            browser = "other"
            for b in self.browsers:
                if b in content:
                    browser = b
                    break
            return operation + "_" + browser
        if name == "slotprice":
            price = int(content)
            if price > 100:
                return "101+"
            elif price > 50:
                return "51-100"
            elif price > 10:
                return "11-50"
            elif price > 0:
                return "1-10"
            else:
                return "0"
    
    def getTags(self, content):
        if content == '\n' or len(content) == 0:
            return ["null"]
        return content.strip().split(',')

    
class Encoder_yoyi:
    def __init__(self, num_features, pay_scale, limit=float('Inf')):
        self.num_features = num_features
        self.limit = limit
        self.pay_scale = pay_scale

    def sparse_encode(self, x, row_index=0):
        # x is one record seperated by \t
        s = x.split('\t')

        col_ind = list(map(lambda x: int(x.split(":")[0]), s[2:]))
        data = list(map(lambda x: int(x.split(":")[1]), s[2:]))
        row_ind = list(repeat(row_index, len(s[2:])))

        return row_ind, col_ind, data

    def encode_one(self, x):
        row_ind, col_ind, data = self.sparse_encode(x)
        return csr_matrix((data, (row_ind, col_ind)), shape=(1, self.num_features), dtype=np.int8)

    def encode(self, X):
        row_ind_all = []
        col_ind_all = []
        data_all = []

        count = 0
        ignore_count = 0
        for line in X:
            row_ind, col_ind, data = self.sparse_encode(line, count)
            row_ind_all.extend(row_ind)
            col_ind_all.extend(col_ind)
            data_all.extend(data)

            count += 1

            # if random.random() < 0.000005:
            #     print("{} records have been encoded".format(count))

        print("All input {} records have been encoded,length:{}".format(count, len(data_all)))
        # print(max(row_ind_all), max(col_ind_all), self.num_features)
        return csr_matrix((data_all, (row_ind_all, col_ind_all)), shape=(count, self.num_features), dtype=np.int8)

    def get_col(self, data, col_name):
        labels_all = []

        count = 0
        for line in data:
            s = line.split('\t')
            if col_name == "click":
                labels_all.append(s[0])
            elif col_name == "payprice":
                labels_all.append(int(s[1])*self.pay_scale)
            else:
                continue
            count += 1
        return np.array(labels_all, dtype=np.int16).reshape((count, 1))


