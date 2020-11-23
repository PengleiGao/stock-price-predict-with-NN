import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import math
from data_prepro import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help = "dataset", type = str)
args = parser.parse_args()
data_name = args.dataset

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(18)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(18)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if 'sp500' in data_name:
    data_path = 'dataset/SP500_new.csv'
elif 'csi300' in data_name:
    data_path = 'dataset/CSI300_new.csv'
elif 'nik' in data_name:
    data_path = 'dataset/NIK225_new.csv'
else:
    print("wrong dataset name")

target_col = 0
indep_col = [0, 7]
win_size = 20
pre_T = 1
train_share = 0.9
is_stateful = False
normalize_pattern = 2
generator = getGenerator(data_path)
datagen = generator(data_path, target_col, indep_col, win_size, pre_T,
                    train_share, is_stateful, normalize_pattern)

xtrain, xval, ytrain, yval, y_mean, y_std = datagen.with_target()
print(" --- Data shapes: ", np.shape(xtrain), np.shape(ytrain), np.shape(xval), np.shape(yval))
print("current dataset:", data_name)
y_all = np.concatenate([ytrain,yval])
ymean = torch.Tensor(y_mean).to(device)
ystd = torch.Tensor(y_std).to(device)
batch_size = 64
dataset_train = subDataset(xtrain, ytrain)
dataset_test = subDataset(xval, yval)
train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1,drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1,drop_last=True)


class MLP(nn.Module):
    def __init__(self, input_dim, step):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.step = step
        self.fc1 = nn.Linear(self.input_dim * self.step, 70)
        self.fc2 = nn.Linear(70, 28)
        self.fc3 = nn.Linear(28, 14)
        self.fc4 = nn.Linear(14, 7)
        self.out = nn.Linear(7, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, self.input_dim * self.step)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.out(x)

        return x


input_dim = xtrain.shape[2]
output_dim = 1
step = xtrain.shape[1]
learning_rate = 0.01
num_epochs = 200

path = '/opt/data/private/journal-master/LSTM/mlp/'
best_log_dir = path + data_name + 'mlp_model_.pth'

def train(train_loader):
    loss_mse = []
    for i, (inputs, target) in enumerate(train_loader):

        inputs = Variable(inputs.view(-1, step, input_dim).to(device))
        target = Variable(target.to(device))

        pred_ = model(inputs)
        pred_y = pred_ * ystd + ymean
        loss = criterion(target, pred_y)

        if torch.cuda.is_available():
            loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mse.append(loss.item())
    train_mse = np.array(loss_mse).mean()
    return train_mse

def eval(test_loader):
    rmse = []
    mae = []
    mape = []
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = Variable(torch.Tensor(xval).to(device))
            target = Variable(torch.Tensor(yval).to(device))
        else:
            inputs = Variable(xval)

        pred_ = model(inputs)
        pred_y = pred_ * ystd + ymean

        test_mse = criterion(target, pred_y)
        rmse_test = torch.sqrt(test_mse)
        mae_test = criterionL1(target, pred_y)
        mape_test = torch.mean(torch.abs((target - pred_y) / target))
        return mae_test, rmse_test, mape_test


criterion = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
model = MLP(input_dim, step).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_test_mae = float("inf")
best_test_rmse = float("inf")
best_test_mape = float("inf")
iter = 0

for epoch in range(num_epochs):
    train_loss = train(train_loader)
    test_mae, test_rmse, test_mape = eval(test_loader)
    if test_rmse < best_test_rmse:
        best_test_mae = test_mae
        best_test_rmse = test_rmse
        best_test_mape = test_mape

        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, best_log_dir)

    print('Epoch: {}. Loss: {}'.format(epoch, train_loss))
    print('test_mae: {}. test_rmse: {}. test_rmse: {}'.format(test_mae, test_rmse, test_mape))

print('best_mae: {}. best_rmse: {}. best_mape: {}'.format(best_test_mae, best_test_rmse, best_test_mape))

def prediction():
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = Variable(torch.Tensor(xval).to(device))
            target = Variable(torch.Tensor(yval).to(device))
        else:
            inputs = Variable(xval)

        pred_ = model(inputs)
        pred_y = pred_ * ystd + ymean

        test_mse = criterion(target, pred_y)
        test_rmse = torch.sqrt(test_mse)
        test_mae = criterionL1(target, pred_y)
        test_mape = torch.mean(torch.abs((target - pred_y)/target))
        pred_y.cpu()
        target.cpu()
        df = pd.DataFrame()
        df['R1'] = pred_y.squeeze(1).cpu()
        df['R2'] = target.squeeze(1).cpu()
        RL = df.corr()
        R = RL['R1']['R2']
        #R1 = pd.Series(pred_y.cpu())
        #R2 = pd.Series(target.cpu())
        #R = R1.corr(R2)
        # R1 = np.sum((test_y[:len(test_predict)]-np.mean(test_y))*(test_predict-np.mean(test_predict)))
        # R2 = np.sqrt(np.sum(((test_y[:len(test_predict)] - np.mean(test_y)) ** 2) * ((test_predict - np.mean(test_predict)) ** 2)))
        # R = R1/R2
        U1 = np.sqrt(np.average(pred_y.cpu() - target.cpu()) ** 2)
        U2 = np.sqrt(np.average(target.cpu() ** 2)) + np.sqrt(np.average(pred_y.cpu() ** 2))
        TheilU = U1 / U2
        print('mae: {}. rmse: {}. mape: {}. R: {}. TheilU: {}'.format(test_mae, test_rmse, test_mape, R, TheilU))
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(list(range(len(pred_y.cpu()))), pred_y.cpu(), color='b', label='predict values')
        plt.plot(list(range(len(target.cpu()))), target.cpu(), color='r', label='true values')
        plt.title('Fitted values of predicted part of SP500 dataset')
        #plt.title('Fitted values of predicted part of CSI300 dataset')
        #plt.title('Fitted values of predicted part of NIKKEI225 dataset')
        plt.legend()
        #plt.savefig("fit.png")
        plt.figure(1)
        plt.subplot(2, 1, 2)
        plt.subplots_adjust(hspace=0.5)
        plt.plot(y_all, label='original series')
        plt.plot([None for _ in range(len(ytrain))] + [x for x in pred_y], label='predicted part')
        plt.title('Predicted values among the whole series of SP500 dataset')
        #plt.title('Predicted values among the whole series of CSI300 dataset')
        #plt.title('Predicted values among the whole series of NIKKEI225 dataset')
        plt.legend()
        plt.savefig("./mlp/SP500-mlp-new.eps")
        plt.savefig("./mlp/SP500-mlp-new.jpg")
        #plt.savefig("./mlp/CSI300-mlp-new.eps")
        #plt.savefig("./mlp/CSI300-mlp-new.jpg")
        #plt.savefig("./mlp/NIK225-mlp-new.eps")
        #plt.savefig("./mlp/NIK225-mlp-new.jpg")
        #return test_mae, test_rmse, test_mape, R, TheilU

ckpt_path = best_log_dir
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Loaded ckpt from epoc {}'.format(start_epoch))
else:
    start_epoch = 0
    print('no saved model, start from epoc 1')

prediction()