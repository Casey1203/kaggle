from mxnet import nd
from mxnet.gluon import loss as gloss

def log_rmse(net, features, labels): # 评价指标
    # 将小于1的值设成1，使得取对数时数值更稳定
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * gloss.L2Loss()(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()

