import numpy as np
import mxnet as mx
from mxnet import nd, image
import gc
import pickle

def save_batch(data_path, batch_size, data_size=300000):
    id_list = np.arange(1, data_size+1).tolist()
    batch_num = 0
    for i in range(0, data_size, batch_size):
        batch_img_dict = {}
        if batch_num in []:
            batch_num += 1
            continue
        batch_list = id_list[i: min(i + batch_size, data_size)]
        for j in batch_list:
            img = image.imread(data_path + '%s.png' % j)
            img = img.asnumpy()
            batch_img_dict[j] = img

        print('start saving batch_%s' % batch_num)
        pickle.dump(batch_img_dict, open('batch_%s.pkl' % batch_num, 'wb'))
        print('complete saving batch_%s' % batch_num)
        gc.collect()
        batch_num += 1

def load_batch():
    for i in range(10):
        batch_img_dict = pickle.load(open('batch_%s.pkl' % i, 'rb'))



# batch_size = 30000
# data_path = '/home/casey/test/'
# data_size = 300000
# save_batch(data_path, batch_size, data_size)
load_batch()