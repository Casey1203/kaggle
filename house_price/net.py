from mxnet.gluon import nn
from mxnet import init
import mxnet as mx

def get_net(drop_prob=0.5):
    net = nn.Sequential()
    net.add(
        # nn.Dropout(drop_prob),
        nn.Dense(256, activation='relu'),
        # nn.Dropout(0.5),
        # nn.Dense(128, activation='relu'),
        # nn.Dropout(0.5),
        # nn.Dense(16, activation='relu'),
        # nn.Dropout(0.5),
        nn.Dense(1)
    )
    net.initialize(init.Xavier(), ctx=mx.cpu(0))
    return net

