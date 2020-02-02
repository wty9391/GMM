import random
import numpy as np
import re
FEAT_NUM = 16
NORMALIZE_PRICE = 300

class Util:
    def __init__(self, input_file, featindex, batch_size, op):
        self.input_file = input_file
        self.featindex = featindex
        self.batch_size = batch_size
        self.field_size = 0

        # key = pos, value = (field_num, nth value)
        self.feat_dict = {}

        # generate feat_dict
        fin = open(featindex, 'r')
        # a sample
        # 12:d033f09d6bf6554688420510708bd8ae     2305    1
        # field:value   position    count
        lines = fin.readlines()
        self.feat_sizes = [0] * (max([int(line.split('\t')[0].split(':')[0]) for line in lines[1:100]]) + 1)

        for line in lines:
            split1 = line.split('\t')
            field_value = split1[0].split(':')

            # features sizes
            if field_value[0] == 'truncate':
                self.feat_dict[0] = (0, 0)
                self.feat_sizes[0] += 1
            else:
                field = int(field_value[0])
                pos = int(split1[-2])
                nth_value = int(self.feat_sizes[field])
                self.feat_dict[pos] = (field, nth_value)
                self.feat_sizes[field] += 1
        fin.close()

        self.nonzero_feat_sizes = [size for size in self.feat_sizes if size > 0]
        self.nonzero_feat_index = [i for i in range(0, len(self.feat_sizes)) if self.feat_sizes[i] > 0]
        self.field_size = len(self.nonzero_feat_sizes)

        # get data from input file and sample
        fin = open(input_file, 'r')
        lines = fin.readlines()
        input_x = []
        for line in lines:
            # if len([m.start() for m in re.finditer(':1', line)]) < FEAT_NUM:
            #     print("Warning: bad data found:", line)
            #     continue
            line = line[:-1].replace(':1', '')
            items = line.split(' ')
            input_x.append([int(x) for x in items[1:]])
        fin.close()
        print("data set size: ", len(input_x))

        self.x = []
        self.b = []
        self.z = []
        self.y = []
        for zbx in input_x:
            if (zbx[0] > 0):
                self.z.append(zbx[0])
                self.b.append(zbx[1])
                self.x.append(zbx[2:])
                if zbx[0] >= zbx[1]:
                    self.y.append(0)
                else:
                    self.y.append(1)
        self.b = np.array(self.b).reshape(-1, 1)
        self.z = np.array(self.z).reshape(-1, 1)
        self.y = np.array(self.y).reshape(-1, 1)
        self.x = np.array(self.x)
        # get batch number
        self.batch_num = int(len(self.y) / self.batch_size)
        self.data_amt = self.z.shape[0]

    def partition(self, x_batches):
        res = []
        for batch in x_batches:
            batch_res = []
            field_res = {}
            for rec in batch:
                field_nth = self.feat_dict[rec[1]] #(field, nth_value)
                rec[1] = field_nth[1]
                if field_nth[0] in field_res:
                    field_res[field_nth[0]].append(rec)
                else:
                    field_res[field_nth[0]] = [rec]

            # TODO  modified by wty
            for key in range(len(self.feat_sizes)):
                if key in field_res:
                    # if len(field_res[key]) < self.batch_size:
                    #     print("Warning: bad field_res found for", key,
                    #           "desired length:", self.batch_size, "actual length:", len(field_res[key]))
                    batch_res.append(field_res[key])
                elif self.feat_sizes[key] > 0:
                    print("Warning: bad field found:", key)
                    batch_res.append([[0, 0]])
                    # batch_res.append([[i, 0] for i in range(0, self.batch_size)])
                else:
                    pass
            res.append(batch_res)
        return res

    def generate_indices(self, x):
        indices = []
        for i in range(len(x)):
            for pos in x[i]:
                indices.append([i, pos])
        return indices

    def sample(self, x):
        res_x = x
        for i in range(len(x)):
            zbx = x[i]
            if zbx[0] == 0: continue
            if zbx[0] < zbx[1]:
                # sample from [1, z]
                sample_b = random.randint(1, zbx[0])
                res_x.append([zbx[0]] + [sample_b] + zbx[2:])

                # sample from [z + 1, b]
                sample_b = random.randint(zbx[0] + 1, zbx[1])
                res_x.append([zbx[0]] + [sample_b] + zbx[2:])

                # sample from [b + 1, 2(b + 1)]
                sample_b = random.randint(zbx[1] + 1, 2 * (zbx[1] + 1))
                res_x.append([zbx[0]] + [sample_b] + zbx[2:])
            else:
                # sample from [1, b]
                sample_b = random.randint(1, zbx[1])
                res_x.append([zbx[0]] + [sample_b] + zbx[2:])
        return res_x

    def get_batch_data(self, num):
        if (num % self.batch_num) == 0:
            print('shuffle')
            index = [i for i in range(self.data_amt)]
            shuffled_index = random.shuffle(index)
            self.b = self.b[shuffled_index].reshape(-1, 1)
            self.z = self.z[shuffled_index].reshape(-1, 1)
            self.y = self.y[shuffled_index].reshape(-1, 1)
            self.x = self.x[shuffled_index].reshape(self.data_amt, self.field_size)
        pos = num % self.batch_num
        x_field_batch = self.partition([self.generate_indices(self.x[pos * self.batch_size : (pos + 1) * self.batch_size].tolist())])[0]
        b_batch = self.b[pos * self.batch_size : (pos + 1) * self.batch_size, :]
        z_batch = self.z[pos * self.batch_size : (pos + 1) * self.batch_size, :]
        y_batch = self.y[pos * self.batch_size : (pos + 1) * self.batch_size, :]
        return x_field_batch, b_batch, z_batch, y_batch
    
    def get_batch_data_sorted(self, num):
        if (num % self.batch_num) == 0:
            print('shuffle')
            index = [i for i in range(self.data_amt)]
            shuffled_index = random.shuffle(index)
            self.b = self.b[shuffled_index].reshape(-1, 1)
            self.z = self.z[shuffled_index].reshape(-1, 1)
            self.y = self.y[shuffled_index].reshape(-1, 1)
            self.x = self.x[shuffled_index].reshape(self.data_amt, self.field_size)
        pos = num % self.batch_num
        x_batch = self.x[pos * self.batch_size : (pos + 1) * self.batch_size]
        b_batch = self.b[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        z_batch = self.z[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        y_batch = self.y[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)

        #sort
        base = (b_batch * (1 - y_batch) + z_batch * y_batch).reshape(-1,)
        sort_index = np.argsort(base)
        x_batch_field_sort = self.partition([self.generate_indices(x_batch[sort_index].tolist())])[0]
        b_batch_sort = b_batch[sort_index]
        z_batch_sort = z_batch[sort_index]
        y_batch_sort = y_batch[sort_index]
        return x_batch_field_sort, b_batch_sort, z_batch_sort, y_batch_sort

    def get_batches_data(self, from_, num):
        pos = from_ % self.batch_num
        x_field_batch = self.partition([self.generate_indices(self.x[pos : pos + num].tolist())])[0]
        b_batch = self.b[pos : pos + num]
        z_batch = self.z[pos : pos + num]
        y_batch = self.y[pos : pos + num]
        return x_field_batch, b_batch, z_batch, y_batch

    def get_batch_data_origin_with_ks(self, num, ks_const):
        if (num % self.batch_num) == 0:
            print('shuffle')
            index = [i for i in range(self.data_amt)]
            shuffled_index = random.shuffle(index)
            self.b = self.b[shuffled_index].reshape(-1, 1)
            self.z = self.z[shuffled_index].reshape(-1, 1)
            self.y = self.y[shuffled_index].reshape(-1, 1)
            ks_const = ks_const[shuffled_index].reshape(-1, 1)
            self.x = self.x[shuffled_index].reshape(self.data_amt, self.field_size)
        pos = num % self.batch_num
        x_batch = self.generate_indices(self.x[pos * self.batch_size : (pos + 1) * self.batch_size].tolist())
        b_batch = self.b[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        z_batch = self.z[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        y_batch = self.y[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        ks_batch = ks_const[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        return x_batch, b_batch, z_batch, y_batch, ks_batch

    def get_batch_data_origin(self, num):
        if (num % self.batch_num) == 0:
            print('shuffle')
            index = [i for i in range(self.data_amt)]
            shuffled_index = random.shuffle(index)
            self.b = self.b[shuffled_index].reshape(-1, 1)
            self.z = self.z[shuffled_index].reshape(-1, 1)
            self.y = self.y[shuffled_index].reshape(-1, 1)
            self.x = self.x[shuffled_index].reshape(self.data_amt, self.field_size)
        pos = num % self.batch_num
        x_batch = self.generate_indices(self.x[pos * self.batch_size : (pos + 1) * self.batch_size].tolist())
        b_batch = self.b[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        z_batch = self.z[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        y_batch = self.y[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        return x_batch, b_batch, z_batch, y_batch

    def get_batch_data_origin_sorted(self, num):
        if (num % self.batch_num) == 0:
            print('shuffle')
            index = [i for i in range(self.data_amt)]
            shuffled_index = random.shuffle(index)
            self.b = self.b[shuffled_index].reshape(-1, 1)
            self.z = self.z[shuffled_index].reshape(-1, 1)
            self.y = self.y[shuffled_index].reshape(-1, 1)
            self.x = self.x[shuffled_index].reshape(self.data_amt, self.field_size)
        pos = num % self.batch_num
        x_batch = self.x[pos * self.batch_size : (pos + 1) * self.batch_size]
        b_batch = self.b[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        z_batch = self.z[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)
        y_batch = self.y[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float64)

        #sort
        base = (b_batch * (1 - y_batch) + z_batch * y_batch).reshape(-1,)
        sort_index = np.argsort(base)
        x_batch_sort = self.generate_indices(x_batch[sort_index].tolist())
        b_batch_sort = b_batch[sort_index]
        z_batch_sort = z_batch[sort_index]
        y_batch_sort = y_batch[sort_index]
        return x_batch_sort, b_batch_sort, z_batch_sort, y_batch_sort

    def get_all_data_origin(self):
        return self.generate_indices(self.x.tolist()), self.b.astype(np.float64), self.z.astype(np.float64), self.y.astype(np.float64)
    
    def get_all_data_origin_sort(self):
        base = (self.b * (1 - self.y) + self.z * self.y).reshape(-1,)
        sort_index = np.argsort(base)
        x_sort = self.generate_indices(self.x[sort_index].tolist())
        b_sort = self.b[sort_index].astype(np.float64)
        z_sort = self.z[sort_index].astype(np.float64)
        y_sort = self.y[sort_index].astype(np.float64)
        return x_sort, b_sort, z_sort, y_sort
    
    def get_batch_data_sorted_dwpp(self, num):
        if (num % self.batch_num) == 0:
            print('shuffle')
            index = [i for i in range(self.data_amt)]
            shuffled_index = random.shuffle(index)
            self.b = self.b[shuffled_index].reshape(-1, 1)
            self.z = self.z[shuffled_index].reshape(-1, 1)
            self.y = self.y[shuffled_index].reshape(-1, 1)
            self.x = self.x[shuffled_index].reshape(self.data_amt, self.field_size)
        pos = num % self.batch_num
        x_batch = self.x[pos * self.batch_size : (pos + 1) * self.batch_size]
        b_batch = self.b[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float32)
        z_batch = self.z[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float32)
        y_batch = self.y[pos * self.batch_size : (pos + 1) * self.batch_size, :].astype(np.float32)

        #sort
        base = (b_batch * (1 - y_batch) + z_batch * y_batch).reshape(-1,)
        sort_index = np.argsort(base)
        x_batch_field_sort = self.partition([self.generate_indices(x_batch[sort_index].tolist())])[0]
        b_batch_sort = b_batch[sort_index]
        z_batch_sort = z_batch[sort_index]
        y_batch_sort = y_batch[sort_index]
        return x_batch_field_sort, b_batch_sort, z_batch_sort, y_batch_sort, np.array([[i for i in range(1, 301)] for j in range(self.batch_size)])


    def get_max_z(self):
        return np.max(self.y * self.z).astype(np.float64)

    def get_data_amt(self):
        return self.data_amt
