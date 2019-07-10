from mxnet.gluon import nn
from mxnet import init

def get_net(drop_prob=0.5):
    net = nn.Sequential()
    net.add(
        # nn.Dropout(drop_prob),
        nn.Dense(512, activation='relu'),
        nn.Dropout(drop_prob),
        nn.Dense(128, activation='relu'),
        nn.Dropout(drop_prob),
        # nn.Dense(64, activation='relu'),
        # nn.Dropout(drop_prob),
        nn.Dense(1)
    )
    net.initialize(init.Xavier())
    return net

