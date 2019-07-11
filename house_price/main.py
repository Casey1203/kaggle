from house_price.net import get_net
from util.train import train
from util.plot import semilogy, multiple_semilogy
from util.metric import log_rmse
from mxnet.gluon import loss as gloss
from house_price.data_process import *
from util.tune_param import get_k_fold_data

num_epochs = 150
lr = 0.05
weight_decay = 100
batch_size = 32

# train_labels = nd.log(train_labels) # 对y取对数

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size, gloss.L2Loss(), log_rmse)
    # semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    file_name = 'submission_{}_{}_{}_{}.csv'.format(str(num_epochs), str(lr), str(weight_decay), str(batch_size))
    submission.to_csv(file_name, index=False)

def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    param_setting = 'num_epoch: {}, lr: {}, weight_decay: {}, batch_size: {}'.format(
        str(num_epochs), str(lr), str(weight_decay), str(batch_size))
    print(param_setting)
    train_l_sum, valid_l_sum = 0, 0
    train_l_list, valid_l_list = [], []
    param_df = pd.DataFrame()
    shuffle_list = nd.shuffle(nd.arange(0, X_train.shape[0]))
    X_train_shuffle = X_train[shuffle_list]
    y_train_shuffle = y_train[shuffle_list]
    real_k = k-1
    for i in range(real_k):
        data = get_k_fold_data(k, i, X_train_shuffle, y_train_shuffle) # train_data, train_label, valid_data, valid_label
        net = get_net()
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate, weight_decay, batch_size, gloss.L2Loss(), log_rmse)
        # param = net.collect_params()
        # param_series = pd.Series(index=feature_list, data=param['dense%s_weight' % i].data().asnumpy()[0])
        # param_df = pd.concat([param_df, param_series], axis=1)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        #     semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
        #                  range(1, num_epochs + 1), valid_ls,
        #                  ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, train_ls[-1], valid_ls[-1]))
        train_l_list.append(train_ls)
        valid_l_list.append(valid_ls)
    print(net.collect_params())
    multiple_semilogy(
        [range(1, num_epochs + 1)] * real_k, train_l_list, ['epochs']*real_k, ['rmse']*real_k,
        [range(1, num_epochs + 1)] * real_k, valid_l_list, ['train', 'valid'] * real_k
    )
    # param_df.to_csv('param_df.csv')
    return train_l_sum / real_k, valid_l_sum / real_k


# train_and_pred(train_features, test_features, train_labels, test_data,
#                num_epochs, lr, weight_decay, batch_size)

train_l_mean, valid_l_mean = k_fold(
    k=5, X_train=train_features, y_train=train_labels, num_epochs=num_epochs,
    learning_rate=lr, weight_decay=weight_decay, batch_size=batch_size
)
print('train_l_mean', train_l_mean, 'valid_l_mean', valid_l_mean)