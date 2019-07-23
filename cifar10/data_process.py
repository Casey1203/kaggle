import numpy as np
import mxnet as mx
from mxnet import nd, image
import gc
import pickle

def save_batch(data_path, batch_size, data_size=300000):
    id_list = np.arange(1, data_size+1).tolist()
    batch_num = 0
    batch_img = None
    for i in range(0, data_size, batch_size):
        batch_list = id_list[i: min(i + batch_size, data_size)]
        for j in batch_list:
            img = image.imread(data_path + '%s.png' % j)
            img = img.reshape(1, 3, 32, 32)
            if batch_img is not None:
                batch_img = nd.concatenate([batch_img, img])
            else:
                batch_img = img
        batch_img = batch_img.asnumpy()
        print('start saving batch_%s' % batch_num)
        pickle.dump(batch_img, 'batch_%s.pkl' % batch_num)
        print('complete saving batch_%s' % batch_num)
        batch_img = None
        gc.collect()
        batch_num += 1

batch_size = 10000
data_path = '/home/casey/test/'
data_size = 300000
save_batch(data_path, batch_size, data_size)
