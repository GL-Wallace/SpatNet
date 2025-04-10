# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from models.tempo_net import TempoNet 
import config as cfg
from utils import utils 
from utils import data_helper


def get_data_loader(x_data, y_data, train_idx, test_idx):
    train_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=train_idx, transform=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=test_idx, transform=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def get_model_and_dataloader(x_spa, x_tempo, y, train_idx, test_idx):
    if cfg.model_name == 'SpaNet':
        model = models.ConvNet(num_channels=cfg.num_channels)
        train_loader, test_loader = get_data_loader(x_data=x_cnn_common, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'TempoNet':
        model = TempoNet(input_dim=cfg.tempo_input_size, out_dim=cfg.tempo_hidden_size, num_layers=cfg.tempo_num_layers, dropout=cfg.tempo_dropout)
        train_loader, test_loader = get_data_loader(x_data=x_tempo, y_data=y, train_idx=train_idx, test_idx=test_idx)
    else:
        print('Model name is not valid.')
        sys.exit(0)
    return model, train_loader, test_loader


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(model, train_loader, test_loader):
    torch.cuda.empty_cache()
    torch.manual_seed(cfg.rand_seed)
    torch.cuda.manual_seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)
    model = model.to(cfg.device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    best_rmse, best_mae, best_r2 = np.inf, np.inf, -np.inf
    best_epoch = 1

    for epoch in range(1, cfg.epochs + 1):
        # print('epoch: {}'.format(epoch))
        model.train()
        loss_list = []
        for batch_idx, data_input in enumerate(train_loader):
            if epoch == 1 and batch_idx == 0:
                print('input_data_shape:')
                for data in data_input:
                    print("这里是什么意思？", data.shape) # data & label
                print()
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            # global_step = batch_idx + (epoch - 1) * int(len(train_loader.dataset) / len(inputs)) + 1

            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                print("检查1",x_input.device)
                print("检查2", next(model.parameters()).device)

                y_pred = model(x_input)
            # y_pred = y_pred.to(cfg.device)
            loss = criterion(y_input, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.cpu().data.numpy()
            loss_list.append(loss_val)
        loss_mean = np.mean(loss_list)
        # print('train_loss = {:.3f}'.format(loss_mean))

        if epoch % cfg.eval_interval != 0:
            continue
        print('epoch: {}'.format(epoch))
        model.eval()
        y_input_list = []
        y_pred_list = []
        for batch_idx, data_input in enumerate(train_loader):
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)

            y_pred_list.extend(y_pred.data.cpu().numpy())
            y_input_list.extend(y_input.data.cpu().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        r2 = metrics.r2_score(y_input_list, y_pred_list)
        print('Train_RMSE = {:.3f}  Train_MAE = {:.3f}  Train_R2 = {:.3f}'.format(rmse, mae, r2))

        y_input_list = []
        y_pred_list = []
        for batch_idx, data_input in enumerate(test_loader):
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)
            y_pred_list.extend(y_pred.data.cpu().numpy())
            y_input_list.extend(y_input.data.cpu().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        r2 = metrics.r2_score(y_input_list, y_pred_list)

        torch.save(model.state_dict(), cfg.model_save_pth)
        print('Test_RMSE  = {:.3f}  Test_MAE  = {:.3f}  Test_R2  = {:.3f}'.format(rmse, mae, r2))
        print()


def main():
    # Basic setting
    device = torch.device('cuda:0' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # Load data
    x_spa, x_tempo, y = utils.generate_xy()
    print('x_spa.shape: {}  x_tempo.shape: {} \n'.format(x_spa.shape, x_tempo.shape))
    # sys.exit(0)

    # Build the model
    train_idx = utils.load_pickle(cfg.f_train_index)
    test_idx = utils.load_pickle(cfg.f_test_index)
    model, train_loader, test_loader = get_model_and_dataloader(x_spa, x_tempo, y, train_idx, test_idx)

    if cfg.device == 'cuda':
        model = model.cuda()
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, model))
    
    # Train the model
    input('Press enter to start training...\n')
    print('START TRAINING\n')
    train_model(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
