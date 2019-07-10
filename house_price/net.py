from mxnet.gluon import nn
from mxnet import init

def get_net(drop_prob=0.5):
    net = nn.Sequential()
    net.add(
        # nn.Dropout(drop_prob),
        nn.Dense(512, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(128, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(64, activation='relu'),
        nn.Dropout(drop_prob),
        nn.Dense(32, activation='relu'),
        nn.Dense(1)
    )
    net.initialize(init.Xavier())
    return net

