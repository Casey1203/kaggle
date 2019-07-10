# coding: utf-8
from util.tune_param import get_k_fold_data
from mxnet.gluon import data as gdata
from mxnet import autograd, gluon
from copy import deepcopy

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, obj_loss, eval_metric):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        if epoch == 100:
            learning_rate /= 2.
        # if epoch == 100:
        #     learning_rate /= 2.
        # if epoch == 200:
        #     learning_rate /= 2.
        # if epoch == 300:
        #     learning_rate /= 2.
        #
        # if epoch == 350:
        #     learning_rate /= 2.
        # if epoch == 380:
        #     learning_rate /= 2.
        # if epoch == 450:
        #     learning_rate /= 10.
        trainer.set_learning_rate(learning_rate)
        for X, y in train_iter:
            with autograd.record():
                l = obj_loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(eval_metric(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(eval_metric(net, test_features, test_labels))
    return train_ls, test_ls


def k_fold(net, k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        net_cp = deepcopy(net)
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net_cp, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        #     d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
        #                  range(1, num_epochs + 1), valid_ls,
        #                  ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k