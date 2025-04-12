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
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics
from models.tempo_net import TempoNet
from models.gru_fpn import BiGRU_FPN
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
        # model = BiGRU_FPN(input_dim=cfg.tempo_input_size, out_dim=cfg.tempo_hidden_size, num_layers=cfg.tempo_num_layers, dropout=cfg.tempo_dropout)
        model = BiGRU_FPN(input_size=cfg.tempo_input_size, hidden_size=cfg.tempo_hidden_size, num_layers=cfg.tempo_num_layers, dropout=cfg.tempo_dropout)
        # input_size, hidden_size, num_layers=2, dropout=0.1
        train_loader, test_loader = get_data_loader(x_data=x_tempo, y_data=y, train_idx=train_idx, test_idx=test_idx)
    else:
        print('Model name is not valid.')
        sys.exit(0)
    return model, train_loader, test_loader


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def plot_training_metrics(train_losses, eval_losses, 
                         train_rmses, eval_rmses,
                         train_maes, eval_maes,
                         train_r2s, eval_r2s,lrs,
                         save_path='./log/training_metrics.png'):
    """
    绘制训练和评估指标曲线
    
    参数:
        train_losses: 训练损失列表
        eval_losses: 评估损失列表
        train_rmses: 训练RMSE列表
        eval_rmses: 评估RMSE列表
        train_maes: 训练MAE列表
        eval_maes: 评估MAE列表
        train_r2s: 训练R2列表
        eval_r2s: 评估R2列表
        save_path: 图片保存路径
    """
    plt.figure(figsize=(18, 8))
    
    # 损失曲线
    plt.subplot(3, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.title('Training and Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # RMSE曲线
    plt.subplot(3, 2, 2)
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(eval_rmses, label='Eval RMSE')
    plt.title('RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    # MAE曲线
    plt.subplot(3, 2, 3)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(eval_maes, label='Eval MAE')
    plt.title('MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # R2曲线
    plt.subplot(3, 2, 4)
    plt.plot(train_r2s, label='Train R2')
    plt.plot(eval_r2s, label='Eval R2')
    plt.title('R2 Score')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
        # LR曲线
    plt.subplot(3, 2, 5)
    plt.plot(lrs, label='LR')
    plt.title('LR Score')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def train_model(model, train_loader, test_loader):
    torch.cuda.empty_cache()
    torch.manual_seed(cfg.rand_seed)
    torch.cuda.manual_seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)
    model = model.to(cfg.device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',     # 监控验证损失的最小值 ‌:ml-citation{ref="2,8" data="citationList"}
    factor=0.1,     # 学习率衰减因子（每次降低为当前值的50%）‌:ml-citation{ref="2,6" data="citationList"}
    patience=5,     # 容忍连续5轮验证损失未改善 ‌:ml-citation{ref="2,8" data="citationList"}
    verbose=True    # 打印学习率更新日志 ‌:ml-citation{ref="8" data="citationList"}
)
    best_rmse, best_mae, best_r2 = np.inf, np.inf, -np.inf
    best_epoch = 1

    # 初始化存储训练和评估指标的列表
    train_losses = []
    eval_losses = []
    train_rmses = []
    eval_rmses = []
    train_maes = []
    eval_maes = []
    train_r2s = []
    eval_r2s = []
    lrs = []

    for epoch in range(1, cfg.epochs + 1):
        # print('epoch: {}'.format(epoch))
        model.train()
        loss_list_train = []
        running_loss = 0.0
        for batch_idx, data_input in enumerate(train_loader):
            if epoch == 1 and batch_idx == 0:
                print('input_data_shape:')
                for data in data_input:
                    print( data.shape) # data & label
                print()
                print("len of data input: ",len(data_input) )
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
                y_pred = model(x_input)
            y_input = y_input.float()
            y_pred = y_pred.float()
            loss = F.smooth_l1_loss(y_input, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.cpu().data.numpy()
            loss_list_train.append(loss_val)
        loss_mean = np.mean(loss_list_train)
        train_losses.append(loss_mean)
        print("loss mean: per epoch: ", loss_mean)
        if epoch % cfg.eval_interval != 0:
            continue
        print('epoch: {}'.format(epoch))

        model.eval()
        y_input_list = []
        y_pred_list = []
        loss_list_eval = []
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
            
            # 假设 y_input 和 y_pred 是 PyTorch 的 Tensor
            if torch.isnan(y_pred).any() or torch.isnan(y_input).any():
                print("y_pred: ", y_pred)
                print("Warning: NaN values found in the tensor.")
            loss = F.smooth_l1_loss(y_input, y_pred)

            loss_val = loss.cpu().data.numpy()
            running_loss += loss_val
            loss_list_eval.append(loss_val)
            y_pred_list.extend(y_pred.data.cpu().numpy())
            y_input_list.extend(y_input.data.cpu().numpy())
        avg_train_loss = running_loss / len(train_loader)
        val_loss = avg_train_loss  # 这里简化为训练集损失，实际应用时可以用验证集
        scheduler.step(val_loss)  # 更新学习率
        print(f"Epoch {epoch+1}: Learning Rate is {optimizer.param_groups[0]['lr']:.6f}")
        lrs.append(optimizer.param_groups[0]['lr'])
        loss_mean = np.mean(loss_list_eval)
        eval_losses.append(loss_mean)
        train_rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        train_mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        train_r2 = metrics.r2_score(y_input_list, y_pred_list)
        train_rmses.append(train_rmse)
        train_maes.append(train_mae)
        train_r2s.append(train_r2)
        print('Train_RMSE = {:.3f}  Train_MAE = {:.3f}  Train_R2 = {:.3f}'.format(train_rmse, train_mae, train_r2))



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
            print("预测结果：", y_pred_list[:5])
        eval_rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        eval_mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        eval_r2 = metrics.r2_score(y_input_list, y_pred_list)
        eval_rmses.append(eval_rmse)
        eval_maes.append(eval_mae)
        eval_r2s.append(eval_r2)

        torch.save(model.state_dict(), cfg.model_save_pth)
        print('Test_RMSE  = {:.3f}  Test_MAE  = {:.3f}  Test_R2  = {:.3f}'.format(eval_rmse, eval_mae, eval_r2))
        print()
    return train_losses, eval_losses, train_rmses, eval_rmses, train_maes, eval_maes, train_r2s, eval_r2s, lrs



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
