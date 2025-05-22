import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kymatio.torch import Scattering2D

from utils import get_dataset, compute_scattering_features

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--rotations', type=int, default=8)
# parser.add_argument('--norm', type=str, default='group')
# parser.add_argument('--num_groups', type=int, default=9)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Device: ', device, flush=True)

train_dataset, test_dataset = get_dataset(args)
if args.dataset == 'mnist' or args.dataset == 'fmnist':
    image_shape = (28, 28)
    num_channels = 3969

    train_images, train_labels = train_dataset.data.float().unsqueeze(1), train_dataset.targets
    test_images, test_labels = test_dataset.data.float().unsqueeze(1), test_dataset.targets
elif args.dataset == 'cifar':
    image_shape = (32, 32)
    num_channels = 15552

    train_images, train_labels = torch.from_numpy(train_dataset.data).permute(0, 3, 1, 2).float(), train_dataset.targets
    test_images, test_labels = torch.from_numpy(test_dataset.data).permute(0, 3, 1, 2).float(), test_dataset.targets
else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))

scattering = Scattering2D(J=args.depth, shape=image_shape, L=args.rotations).to(device)
train_scattering_features = compute_scattering_features(train_images, scattering, args.batch_size, device)
test_scattering_features = compute_scattering_features(test_images, scattering, args.batch_size, device)

train_features = torch.from_numpy(train_scattering_features).float()
test_features = torch.from_numpy(test_scattering_features).float()

# if args.norm == 'group':
#     group_norm = nn.GroupNorm(args.num_groups, num_channels)
#     train_features = group_norm(train_features, grad=False)
#     test_features = group_norm(test_features, grad=False)

torch.save(train_features, f'features/{args.dataset}_train.pt')
torch.save(test_features, f'features/{args.dataset}_test.pt')