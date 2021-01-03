import argparse

import torch
from torch.optim import Adam
import numpy as np
from sklearn.metrics import roc_auc_score

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from transform import JunctionTree
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--no_inter_message_passing', action='store_true')
args = parser.parse_args()
print(args)


class OGBTransform(object):
    # OGB saves atom and bond types zero-index based. We need to revert that.
    def __call__(self, data):
        data.x[:, 0] += 1
        data.edge_attr[:, 0] += 1
        return data


transform = Compose([OGBTransform(), JunctionTree()])

name = 'ogbg-molmuv'
dataset = PygGraphPropPredDataset(name, 'data', pre_transform=transform)

dataset = dataset.shuffle()
num_train, num_trainval = round(0.8 * len(dataset)), round(0.9 * len(dataset))
train_dataset = dataset[:num_train]
val_dataset = dataset[num_train:num_trainval]
test_dataset = dataset[num_trainval:]

train_loader = DataLoader(train_dataset, 128, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, 1000, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, 1000, shuffle=False, num_workers=12)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
model = Net(hidden_channels=args.hidden_channels,
            out_channels=dataset.num_tasks, num_layers=args.num_layers,
            dropout=args.dropout,
            inter_message_passing=not args.no_inter_message_passing).to(device)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        out = model(data)[mask]
        y = data.y.to(torch.float)[mask]
        loss = torch.nn.BCEWithLogitsLoss()(out, y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(model(data))
        y_trues.append(data.y)

    y_pred = torch.cat(y_preds, dim=0).cpu().numpy()
    y_true = torch.cat(y_trues, dim=0).cpu().numpy()

    rocauc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    return sum(rocauc_list) / len(rocauc_list)


test_perfs = []
for run in range(1, 5):
    print()
    print(f'Run {run}:')
    print()

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.001)

    best_val_perf = test_perf = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        train_perf = test(train_loader)
        val_perf = test(val_loader)

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = test(test_loader)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_perf:.4f}, Val: {val_perf:.4f}, '
              f'Test: {test_perf:.4f}')

    test_perfs.append(test_perf)

test_perf = torch.tensor(test_perfs)
print('===========================')
print(f'Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}')
