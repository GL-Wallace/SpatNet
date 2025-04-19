# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics
from models.tempo_net import TempoNet
from models.spa_net import SpaNet
from models.spat_net import SpatNet
import config as cfg
from utils import utils 
from utils import data_helper


def get_data_loader(x_data, y_data, train_idx, test_idx):
    train_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=train_idx, transform=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=test_idx, transform=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader

def get_data_loader_spat(x_data_spa, x_data_tempo, y_data, train_idx, test_idx):
    train_dataset = data_helper.DatasetSPAT(x_data_spa=x_data_spa, x_data_tempo=x_data_tempo, y_data=y_data, data_index=train_idx, transform=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = data_helper.DatasetSPAT(x_data_spa=x_data_spa, x_data_tempo=x_data_tempo, y_data=y_data, data_index=test_idx, transform=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader

def get_model_and_dataloader(x_spa, x_tempo, y, train_idx, test_idx):
    if cfg.model_name == 'SpaNet':
        model = SpaNet(in_channels=cfg.num_channels)
        train_loader, test_loader = get_data_loader(x_data=x_spa, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'TempoNet':
        model = TempoNet(input_size=cfg.tempo_input_size, hidden_size=cfg.tempo_hidden_size, num_layers=cfg.tempo_num_layers, dropout=cfg.tempo_dropout)
        train_loader, test_loader = get_data_loader(x_data=x_tempo, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'SpatNet':
        model = SpatNet(input_dim_spa = cfg.num_channels, input_dim_tempo=cfg.tempo_input_size, hidden_dim=cfg.hidden_dim, output_dim=1, num_heads=cfg.num_heads)
        train_loader, test_loader = get_data_loader_spat(x_data_spa=x_spa,x_data_tempo=x_tempo, y_data=y, train_idx=train_idx, test_idx=test_idx)
    else:
        print('Model name is not valid.')
        sys.exit(0)
    return model, train_loader, test_loader

def process_input(data_input):
    device = cfg.device
    if len(data_input) >= 3:
        x_input_cnn = data_input[0].to(device)
        x_input_lstm = data_input[1].to(device)
        y_input = data_input[2].to(device)
        return (x_input_cnn, x_input_lstm), y_input
    else:
        x_input = data_input[0].to(device)
        y_input = data_input[1].to(device)
        return (x_input,), y_input

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return rmse, mae, r2
def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_losses = []

    for batch_idx, data_input in enumerate(data_loader):
        # 可选：首个 batch 输出 shape，便于调试
        if epoch == 1 and batch_idx == 0:
            for i, data in enumerate(data_input):
                print(f"Input {i} shape: {data.shape}")

        x_input, y_input = process_input(data_input, device)
        y_pred = model(*x_input)
        y_input = y_input.float()
        y_pred = y_pred.float()
        loss = criterion(y_pred, y_input)

        optimizer.zero_grad()
        loss.backward()

        # 可选：调试梯度用
        # visualize_gradients(model)
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name}: grad norm = {param.grad.norm().item():.4f}")

        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

    return train_losses
def update_history(history, phase, loss, rmse, mae, r2):
    history[phase]['loss'].append(loss)
    history[phase]['rmse'].append(rmse)
    history[phase]['mae'].append(mae)
    history[phase]['r2'].append(r2)
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    y_true, y_pred_all = [], []

    with torch.no_grad():
        for data_input in data_loader:
            x_input, y_input = process_input(data_input, device)
            y_output = model(*x_input)

            y_pred_all.extend(y_output.data.cpu().numpy())
            y_true.extend(y_input.data.cpu().numpy())

    # 全部转成 Tensor 后计算 loss（比 batch 求 loss 后再平均更准确）
    y_pred_all_tensor = torch.tensor(y_pred_all, dtype=torch.float32)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    loss = criterion(y_pred_all_tensor, y_true_tensor).item()

    return y_true, y_pred_all, loss
def train_model(model, train_loader, test_loader):
    device = cfg.device
    set_random_seed(cfg.rand_seed)
    model = model.to(device)
    model.apply(init_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    best_metrics = {
        'rmse': np.inf,
        'mae': np.inf,
        'r2': -np.inf,
        'epoch': 1
    }

    history = {'train': {'loss': [], 'rmse': [], 'mae': [], 'r2': []},
               'eval': {'loss': [], 'rmse': [], 'mae': [], 'r2': []},
               'lr': []}

    for epoch in range(1, cfg.epochs + 1):
        # 训练阶段
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        train_y_true, train_y_pred, train_loss = evaluate_model(model, train_loader, criterion, device)
        train_rmse, train_mae, train_r2 = calculate_metrics(train_y_true, train_y_pred)

        # 保存训练 history
        update_history(history, 'train', train_loss, train_rmse, train_mae, train_r2)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 如果不是 eval interval，则跳过评估
        if epoch % cfg.eval_interval != 0 and epoch != cfg.epochs:
            continue

        # 测试阶段
        test_y_true, test_y_pred, test_loss = evaluate_model(model, test_loader, criterion, device)
        test_rmse, test_mae, test_r2 = calculate_metrics(test_y_true, test_y_pred)
        update_history(history, 'eval', test_loss, test_rmse, test_mae, test_r2)

        # 记录最优模型
        if test_rmse < best_metrics['rmse']:
            best_metrics.update({'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2, 'epoch': epoch})
            torch.save(model.state_dict(), cfg.best_model_path)
            print(f"Best model saved at Epoch {epoch} (Test RMSE: {test_rmse:.4f})")

        # 日志打印
        print(f"Epoch: {epoch} | LR: {optimizer.param_groups[0]['lr']:.3f} | Train Loss: {train_loss:.2f}")
        print(f"       Train: RMSE={train_rmse:.3f}  MAE={train_mae:.3f}  R²={train_r2:.3f}")
        print(f"        Test: RMSE={test_rmse:.3f}  MAE={test_mae:.3f}  R²={test_r2:.3f}\n")

    return history, best_metrics

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
    train_losses, eval_losses, train_rmses, eval_rmses, train_maes, eval_maes, train_r2s, eval_r2s, lrs = train_model(model, train_loader, test_loader)
    plot_training_metrics(train_losses, eval_losses, train_rmses, eval_rmses, train_maes, eval_maes, train_r2s, eval_r2s, lrs)

if __name__ == '__main__':
    main()
