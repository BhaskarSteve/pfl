import argparse
import torch
import csv
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from utils import get_dataset
from models import CNNMnist, CNNCifar

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Adam weight decay')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--activation', type=str, default='relu', help='activation function')
parser.add_argument('--scale', type=float, default=2.0, help='scale for tempered sigmoid')
parser.add_argument('--temp', type=float, default=2.0, help='temperature for tempered sigmoid')
parser.add_argument('--offset', type=float, default=1.0, help='offset for tempered sigmoid')

parser.add_argument('--disable_dp', action='store_true', help='disable differential privacy')
parser.add_argument('--epsilon', type=float, default=2.93, help='epsilon for (epsilon, delta)-DP')
parser.add_argument('--delta', type=float, default=1e-5, help='delta for (epsilon, delta)-DP')
parser.add_argument('--max_norm', type=float, default=1.0, help='clip per-sample gradients')
parser.add_argument('--sigma', type=float, default=1.0, help='noise multiplier')

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
    model = CNNMnist(args=args)
elif args.dataset == 'cifar':
    model = CNNCifar(args=args)
else:
    exit('Error: unrecognized dataset')
model.to(device)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
criterion = torch.nn.NLLLoss().to(device)
epoch_loss = []

# privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)

# model, optimizer, trainloader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=trainloader,
#     noise_multiplier=args.sigma,
#     max_grad_norm=args.max_per_sample_grad_norm,
# )

if not args.disable_dp:
    print('Differential Privacy is enabled', flush=True)
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_norm,
    )
    print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_norm}", flush=True)

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

for epoch in range(args.epochs):
    model.train()
    batch_loss = []
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(images), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()), flush=True)
        batch_loss.append(loss.item())

    loss_avg = sum(batch_loss)/len(batch_loss)
    print('\nTrain loss:', loss_avg, flush=True)
    epoch_loss.append(loss_avg)

    test_acc, test_loss = test_inference(model, test_dataset, device)
    print("Test Accuracy: {:.2f}%".format(100*test_acc), flush=True)

file_name = f"base/{args.dataset}.csv"
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, mode='a') as file:
    writer = csv.writer(file)
    if args.activation != 'tempered':
        if not args.disable_dp:
            writer.writerow([args.dataset, args.epochs, args.batch_size, args.optimizer, args.lr, (not args.disable_dp, args.epsilon, args.delta, args.max_norm, optimizer.noise_multiplier), args.activation, 100*test_acc])
        else:
            writer.writerow([args.dataset, args.epochs, args.batch_size, args.optimizer, args.lr, (not args.disable_dp), args.activation, 100*test_acc])
    else:
        if not args.disable_dp:
            writer.writerow([args.dataset, args.epochs, args.batch_size, args.optimizer, args.lr, (not args.disable_dp, args.epsilon, args.delta, args.max_norm, optimizer.noise_multiplier), (args.activation, args.scale, args.temp, args.offset), 100*test_acc])
        else:
            writer.writerow([args.dataset, args.epochs, args.batch_size, args.optimizer, args.lr, (not args.disable_dp), (args.activation, args.scale, args.temp, args.offset), 100*test_acc])