# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# For data preprocess
import numpy as np
import csv
import os
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score 
from feature_extraction import *

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

class YelpDataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 features,
                 labels,
                 feature_ids,
                 mode='train',):
        self.mode = mode
        # Read data into numpy arrays
        """
        originally label = -1 is fake review and label = 1 is real review
        for training purpose I make fake reviews as label 1 and real reviews as label 0
        """
        # labels = np.array([1 if label == -1 else 0 for label in labels ])
        labels = np.array(labels)
        # print("labels", labels)
        data = np.array(features).astype(float)
        # data = np.array(data)[:, 1:].astype(float)

        feats = feature_ids

        if mode == 'test':
            # Testing data
            data = data[:, feats]
            # data = pca_transformer.transform(data)
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            target = labels
            print(len(data[0]))
            data = data[:, feats]
            # data = pca_transformer.transform(data)
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 20 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 20 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data = \
            (self.data- self.data.mean(dim=0, keepdim=True)) \
            / self.data.std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of Yelp Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title='', type=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/loss_{type}.png')
    plt.show()
    


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
      model.eval()
      preds, targets = [], []
      for x, y in dv_set:
          x, y = x.to(device), y.to(device)
          with torch.no_grad():
              pred = model(x)
              preds.append(pred.detach().cpu())
              targets.append(y.detach().cpu())
      preds = torch.cat(preds, dim=0).numpy()
      targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
    


# Construct dataloader
def prep_dataloader(mode, batch_size, data, labels,feature_ids, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = YelpDataset(data, labels, mode=mode, feature_ids=feature_ids)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            
    return dataloader


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 2)
        self.out =  nn.Linear(2, 1)
        # cross entropy loss
        self.criterion = nn.BCELoss()

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        x = torch.relu(self.layer1(x))
        x = self.out (self.layer2(x))
        x = torch.sigmoid(x)
        
        return x.squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # L2 regularization here
        # l2_lambda = 0.001
        # l2_norm = sum(p.pow(2.0).sum()
        #           for p in self.parameters())
        return self.criterion(pred, target) #+ l2_lambda*l2_norm

def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_entropy = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            BCE_loss = model.cal_loss(pred, y)  # compute loss
            BCE_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(BCE_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_entropy = dev(dv_set, model, device)
        if dev_entropy < min_entropy:
            # Save model if your model improved
            min_entropy = dev_entropy
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_entropy))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['dev'].append(dev_entropy)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_entropy, loss_record

def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            entropy_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += entropy_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds
    
def run_DL(data, label, feature_ids, top2_features, n_epoches = 2, batch_size = 10, type=''):
    device = get_device()                 # get the current available device ('cpu' or 'cuda')
    rootdir = os.getcwd()
    os.makedirs(os.path.join(rootdir, 'CODE', 'models'), exist_ok=True)  # The trained model will be saved to ./models/
    target_only = True                   # TODO: Using 40 states & 2 tested_positive features

    config = {
        'n_epochs': n_epoches,                # maximum number of epochs
        'batch_size': batch_size,               # mini-batch size for dataloader
        'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
        'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
            # 'lr': 0.001,                 # learning rate 
            # 'momentum': 0.9              # momentum 
        },
        'early_stop': 500,               # early stopping epochs (the number epochs since your model's last improvement)
        'save_path': os.path.join(rootdir, 'CODE', 'models', 'model.pth')  # your model will be saved here
    }


    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
    tr_set = prep_dataloader('train', config['batch_size'], X_train, y_train, feature_ids)
    dv_set = prep_dataloader('dev', config['batch_size'], X_train, y_train, feature_ids)
    tt_set = prep_dataloader('test', config['batch_size'], X_test, y_test, feature_ids)
    
    top1, top2 = top2_features
    feature_1, feature_2 = X_test[top1].to_numpy(), X_test[top2].to_numpy()

    model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
    # load the best model
    model.load_state_dict(torch.load(config['save_path']))
    plot_learning_curve(model_loss_record, title='deep model', type=type)
    preds = np.around(test(tt_set, model, device))
    # plot top 2 feature space
    cdict = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots()
    for g in np.unique(preds):
        ix = list(np.where(preds == g)[0])
        ax.scatter(feature_1[ix],feature_2[ix], c = cdict[g], label = g, s = 100)
    ax.legend()
    plt.xlabel(top1)
    plt.ylabel(top2)
    plt.legend(loc="upper left")
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/visulaization_{type}.png')
    save_path = ""
    save_pred(preds, save_path + 'pred.csv')
    evaluate(y_test, preds, type)
    
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

# def evaluate(test_labels,y_pred):
#     tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
#     acc = accuracy_score(test_labels, y_pred)
#     precision = precision_score(test_labels, y_pred)
#     sensitivity = recall_score(test_labels, y_pred)
#     specificity = tn/(tn+fp)
#     metrics_val = [acc, precision, sensitivity, specificity]
#     metrics = pd.DataFrame()
#     metrics["metic_type"] = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity']
#     metrics["value"] = metrics_val
#     metrics.to_csv("Metrics.csv", header=False, index=False)
