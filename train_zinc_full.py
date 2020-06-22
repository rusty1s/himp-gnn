import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader

from transform import JunctionTree
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--no_inter_message_passing', action='store_true')
args = parser.parse_args()
print(args)

root = 'data/ZINC'
transform = JunctionTree()

train_dataset = ZINC(root, split='train', pre_transform=transform)
val_dataset = ZINC(root, split='val', pre_transform=transform)
test_dataset = ZINC(root, split='test', pre_transform=transform)

train_loader = DataLoader(train_dataset, 256, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, 1000, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, 1000, shuffle=False, num_workers=12)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
model = Net(hidden_channels=args.hidden_channels, out_channels=1,
            num_layers=args.num_layers, dropout=args.dropout,
            inter_message_passing=not args.no_inter_message_passing).to(device)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = (model(data).squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        total_error += (model(data).squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


test_maes = []
for run in range(1, 5):
    print()
    print(f'Run {run}:')
    print()

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=5, min_lr=0.00001)

    best_val_mae = test_mae = float('inf')
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_mae = test(val_loader)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            test_mae = test(test_loader)

        print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, '
              f'Val: {val_mae:.4f}, Test: {test_mae:.4f}')

    test_maes.append(test_mae)

test_mae = torch.tensor(test_maes)
print('===========================')
print(f'Final Test: {test_mae.mean():.4f} ± {test_mae.std():.4f}')
