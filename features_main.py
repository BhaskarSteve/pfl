import os
import copy
import argparse
import numpy as np
import torch
import csv
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
import torch.nn as nn
from update import LocalUpdate
from models import ScatterCNN, ScatterLinear
from utils import get_dataset, average_weights, exp_details

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--global_ep', type=int, default=2)
parser.add_argument('--num_users', type=int, default=10)
parser.add_argument('--frac', type=float, default=1.0)
parser.add_argument('--partition', type=str, default='iid', choices=['iid', 'non-iid'], help="Data partitioning strategy: 'iid' or 'non-iid'")

parser.add_argument('--local_bs', type=int, default=64, help='local batch size')
parser.add_argument('--local_ep', type=int, default=1)

parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Adam weight decay')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model', type=str, default='linear', help='linear or cnn')

parser.add_argument('--disable_dp', action='store_true', help='disable differential privacy')
parser.add_argument('--epsilon', type=float, default=2.93, help='epsilon for (epsilon, delta)-DP')
parser.add_argument('--delta', type=float, default=1e-5, help='delta for (epsilon, delta)-DP')
parser.add_argument('--max_norm', type=float, default=1.0, help='clip per-sample gradients')
parser.add_argument('--sigma', type=float, default=1.0, help='noise multiplier')

parser.add_argument('--norm', type=str, default='group')
parser.add_argument('--num_groups', type=int, default=27)
args = parser.parse_args()

# define paths
path_project = os.path.abspath('..')
exp_details(args)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Device: ', device)

# load dataset and user groups
train_dataset, test_dataset, user_groups = get_dataset(args, True)
train_labels = train_dataset.targets
test_labels = test_dataset.targets
if args.dataset == 'cifar':
    test_labels = torch.tensor(test_labels)
    train_labels = torch.tensor(train_labels)

train_features = torch.load(f'features/{args.dataset}_train.pt')
test_features = torch.load(f'features/{args.dataset}_test.pt')

if args.model == 'linear':
    train_features = train_features.view(train_features.size(0), -1)
    test_features = test_features.view(test_features.size(0), -1)
    global_model = ScatterLinear(train_features.shape[1], 10, args.num_groups).to(device)
elif args.model == 'cnn':
    global_model = ScatterCNN(args.dataset, train_features.shape[1], 10, args.num_groups).to(device)
else:
    raise ValueError('Unknown model: {}'.format(args.model))

train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

# Set the model to train and send it to device.
global_model.to(device)
global_model.train()
print(global_model)

# copy weights
global_weights = global_model.state_dict()

# Training
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0

def test_inference(model, test_dataset, device):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

for epoch in range(args.global_ep):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    client = 1
    for idx in idxs_users:
        print(f'Client {client}/{m}')
        client += 1
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], device=device)
        w, loss, noise_multiplier = local_model.update_weights(
            model=copy.deepcopy(global_model), global_round=epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

    # update global weights
    global_weights = average_weights(local_weights)

    # update global weights
    global_model.load_state_dict(global_weights)

    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    # Calculate avg training accuracy over all users at every epoch
    list_acc, list_loss = [], []
    global_model.eval()
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], device=device)
        acc, loss = local_model.inference(model=global_model)
        list_acc.append(acc)
        list_loss.append(loss)
    train_accuracy.append(sum(list_acc)/len(list_acc))


    print(f' \nAvg Training Stats after {epoch+1} global rounds:')
    print(f'Training Loss : {np.mean(np.array(train_loss))}')
    print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


    # Test inference after completion of training
    test_acc, test_loss = test_inference(global_model, test_dataset, device)
    print(f' \n Results after {epoch+1} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

file_name = f"results/{args.partition}/features_main/{args.dataset}.csv"
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, mode='a') as file:
    writer = csv.writer(file)
    if not args.disable_dp:
        writer.writerow([args.dataset, args.global_ep, args.local_ep, args.local_bs, args.optimizer, args.lr, args.num_users, (not args.disable_dp, args.epsilon, args.delta, args.max_norm, noise_multiplier), args.model, 100*test_acc])
    else:
        writer.writerow([args.dataset, args.global_ep, args.local_ep, args.local_bs, args.optimizer, args.lr, args.num_users, (not args.disable_dp), args.model, 100*test_acc])