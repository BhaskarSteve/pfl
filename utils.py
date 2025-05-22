import copy
import numpy as np
import torch
from torchvision import datasets, transforms

def iid(dataset, num_users):
    num_items = max(int(len(dataset)/num_users), int(len(dataset)/10))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    return dict_users


def non_iid_label_split(dataset, num_users, num_shards_per_user=2):
    '''
    Splits dataset into non-iid partitions by labels.
    Each user gets data from num_shards_per_user labels.
    '''
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    shards_per_class = num_users * num_shards_per_user // num_classes
    idxs = np.arange(len(dataset))
    # Sort by label
    idxs_labels = np.vstack((idxs, labels)).T
    idxs_labels = idxs_labels[idxs_labels[:,1].argsort()]
    idxs = idxs_labels[:,0]

    shard_size = len(dataset) // (num_users * num_shards_per_user)
    shards = [set(idxs[i*shard_size:(i+1)*shard_size]) for i in range(num_users * num_shards_per_user)]

    dict_users = {i: set() for i in range(num_users)}
    shard_idxs = np.arange(num_users * num_shards_per_user)
    np.random.shuffle(shard_idxs)
    for i in range(num_users):
        assigned_shards = shard_idxs[i*num_shards_per_user:(i+1)*num_shards_per_user]
        for shard in assigned_shards:
            dict_users[i].update(shards[shard])
    return dict_users


def get_dataset(args, fed=False):
    data_dir = '../data/'
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                    transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                    transform=apply_transform)
    elif args.dataset == 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                          transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                         transform=apply_transform)
    elif args.dataset == 'cifar':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    else:
        exit('Error: unrecognized dataset')
    if fed:
        partition = getattr(args, 'partition', 'iid')
        if partition == 'non-iid':
            user_groups = non_iid_label_split(train_dataset, args.num_users)
        else:
            user_groups = iid(train_dataset, args.num_users)
        return train_dataset, test_dataset, user_groups
    else:
        return train_dataset, test_dataset


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.global_ep}\n')

    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def compute_scattering_features(images, scattering, batch_size, device):
    scattering_features = []
    num_samples = images.size(0)
    for i in range(0, num_samples, batch_size):
        print('Processing samples [{}/{}]'.format(i, num_samples))
        batch_images = images[i:i+batch_size].to(device)
        if images.size(2) == 28:
            Sx = scattering(batch_images)
            Sx = Sx.cpu().numpy()
            # Flatten the features (for linear models)
            # Sx = Sx.reshape(Sx.shape[0], -1)
        
            # Squeeze the batch dimension (for CNN models)
            Sx = Sx.squeeze(1)
        elif images.size(2) == 32:
            Sx = scattering(batch_images).view(batch_images.size(0), 243, 8, 8)
            Sx = Sx.cpu().numpy()
        scattering_features.append(Sx)
    scattering_features = np.concatenate(scattering_features, axis=0)
    return scattering_features