import numpy as np
import pandas as pd
import os.path as osp

import torch
import torch.nn.functional as func
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from Model import GCN
from Dataset import ConnectivityData


def GCN_train(loader):
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = func.cross_entropy(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def GCN_test(loader):
    model.eval()

    pred = []
    label = []
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred.append(func.softmax(output, dim=1).max(dim=1)[1])
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    epoch_sen = tp / (tp + fn)
    epoch_spe = tn / (tn + fp)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    return epoch_sen, epoch_spe, epoch_acc


dataset = ConnectivityData('./data_demo')

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = np.genfromtxt(osp.join(dataset.raw_dir, 'labels.csv'))
eval_metrics = np.zeros((skf.n_splits, 3))

for n_fold, (train, test) in enumerate(skf.split(labels, labels)):

    model = GCN(dataset.num_features, dataset.num_classes, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_dataset, test_dataset = dataset[train.tolist()], dataset[test.tolist()]

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(50):
        loss = GCN_train(train_loader)
        _, _, train_acc, = GCN_test(train_loader)
        test_sen, test_spe, test_acc = GCN_test(test_loader)
        print('CV: {:03d}, Epoch: {:03d}, Loss: {:.5f}, Train ACC: {:.5f}, Test ACC: {:.5f}'
              .format(n_fold + 1, epoch + 1, loss, train_acc, test_acc))

    eval_metrics[n_fold, 0] = test_sen
    eval_metrics[n_fold, 1] = test_spe
    eval_metrics[n_fold, 2] = test_acc

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'ACC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
