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

torch.manual_seed(16)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(16)

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


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, torch.tanh(cy))

        return (hy, cy)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device))
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device))
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        cn = c0[0, :, :]
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)

        out = outs[-1].squeeze()

        out = self.fc(out)
        # out.size() --> 100, 10
        return out


input_dim = 7
hidden_dim = 140
layer_dim = 1
output_dim = 1
step = xtrain.shape[1]
learning_rate = 0.01
num_epochs = 200

path = '/opt/data/private/journal-master/LSTM/lstm/'
best_log_dir = path + data_name + '_lstm_model_.pth'

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
        mape_test = torch.mean(torch.abs((target - pred_y)/target))
        return mae_test, rmse_test, mape_test


criterion = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

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
    print('test_mae: {}. test_rmse: {}. test_mape: {}'.format(test_mae, test_rmse, test_mape))

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
        plt.savefig("./lstm/SP500-lstm.eps")
        plt.savefig("./lstm/SP500-lstm.jpg")
        #plt.savefig("./lstm/CSI300-lstm.eps")
        #plt.savefig("./lstm/CSI300-lstm.jpg")
        #plt.savefig("./lstm/NIK225-lstm.eps")
        #plt.savefig("./lstm/NIK225-lstm.jpg")
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
