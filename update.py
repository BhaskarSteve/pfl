import torch
from opacus import PrivacyEngine
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, device, logger=None):
        self.args = args
        # self.logger = logger
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
            # dataset, list(idxs))
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)
        self.device = device
        self.criterion = nn.NLLLoss().to(self.device)

    # def train_val_test(self, dataset, idxs):
    #     idxs_train = idxs[:int(0.8*len(idxs))]
    #     idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
    #     idxs_test = idxs[int(0.9*len(idxs)):]

    #     trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
    #                              batch_size=self.args.local_bs, shuffle=True)
    #     validloader = DataLoader(DatasetSplit(dataset, idxs_val),
    #                              batch_size=int(len(idxs_val)/10), shuffle=False)
    #     testloader = DataLoader(DatasetSplit(dataset, idxs_test),
    #                             batch_size=int(len(idxs_test)/10), shuffle=False)
    #     return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        epoch_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        
        if not self.args.disable_dp:
            print('Differential Privacy is enabled', flush=True)
            privacy_engine = PrivacyEngine()
            model, optimizer, self.trainloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                epochs=(self.args.global_ep * self.args.local_ep),
                target_epsilon=self.args.epsilon,
                target_delta=self.args.delta,
                max_grad_norm=self.args.max_norm,
            )
            print(f"Using sigma={optimizer.noise_multiplier} and C={self.args.max_norm}", flush=True)

        for iter in range(self.args.local_ep):
            model.train()
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, 100. * batch_idx / len(self.trainloader), loss.item()), flush=True)
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if not self.args.disable_dp:
            model_dict = {key[8:]: value for key, value in model.state_dict().items()}
            return model_dict, sum(epoch_loss) / len(epoch_loss), optimizer.noise_multiplier
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), 0
        

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        # for batch_idx, (images, labels) in enumerate(self.testloader):
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


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
