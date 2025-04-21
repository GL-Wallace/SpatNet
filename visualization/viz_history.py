import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import numpy as np

def plot_training_history(history, save_path=None, dark=False):
    
    sns.set(style="darkgrid" if not dark else "whitegrid")
    if dark:
        plt.style.use('dark_background')

    epochs = range(1, len(history['train']['loss']) + 1)
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Training History Overview", fontsize=18)

    # Loss
    axs[0, 0].plot(epochs, history['train']['loss'], label='Train Loss')
    axs[0, 0].plot(epochs, history['eval']['loss'], label='Eval Loss')
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # RMSE
    axs[0, 1].plot(epochs, history['train']['rmse'], label='Train RMSE')
    axs[0, 1].plot(epochs, history['eval']['rmse'], label='Eval RMSE')
    axs[0, 1].set_title("RMSE")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("RMSE")
    axs[0, 1].legend()

    # MAE
    axs[1, 0].plot(epochs, history['train']['mae'], label='Train MAE')
    axs[1, 0].plot(epochs, history['eval']['mae'], label='Eval MAE')
    axs[1, 0].set_title("MAE")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("MAE")
    axs[1, 0].legend()

    # R2
    axs[1, 1].plot(epochs, history['train']['r2'], label='Train R²')
    axs[1, 1].plot(epochs, history['eval']['r2'], label='Eval R²')
    axs[1, 1].set_title("R² Score")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("R²")
    axs[1, 1].legend()

    # Learning Rate
    axs[2, 0].plot(range(1, len(history['lr']) + 1), history['lr'], label='Learning Rate', color='tab:orange')
    axs[2, 0].set_title("Learning Rate Schedule")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("LR")
    axs[2, 0].legend()

    axs[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    else:
        plt.show()


def visualize_attention(
    attn_weight,
    sample_indices=[0,1,2],
    save_path=None
    ):
    attn_weight = attn_weight.detach().cpu()  # shape: [B, seq, seq]
    B, seq_len, _ = attn_weight.shape
    num_samples = len(sample_indices)

    # 创建每个样本两个子图（nrows=2 * sample）
    fig, axes = plt.subplots(
        nrows=2 ,
        ncols=1 * num_samples,
        figsize=(3 * num_samples, 5 )
    )

    if num_samples == 1:
        axes = [axes]  # 保证 axes 是 list

    time_steps = [f"T-2", f"T-1", "T"] 
    for idx, sample_idx in enumerate(sample_indices):
        attn_matrix = attn_weight[sample_idx]  # [seq, seq]

        # --- 上图：每行 softmax（每个 Query 时间步的分布）
        row_softmax = torch.softmax(attn_matrix, dim=-1).numpy()
        ax_heatmap = axes[0, idx]
        sns.heatmap(
            row_softmax,
            cmap="Blues",
            square=True,
            # annot=False,
            xticklabels=time_steps,
            yticklabels=time_steps,
            ax=ax_heatmap
        )
        ax_heatmap.set_title(f"[Sample {sample_idx+1}]", fontsize=14)
        ax_heatmap.set_xlabel("Time Step", fontsize=14)
        ax_heatmap.set_ylabel("Time Step", fontsize=14)

        # --- 
        row_softmax = torch.softmax(attn_matrix, dim=1).numpy()
        avg_contribution = row_softmax.mean(axis=0)
        ax_bar = axes[1, idx]
        x = np.arange(1,len(time_steps)+1, 1)
        ax_bar.bar(
            x = x,
            height=avg_contribution,
            color=sns.color_palette("Blues")[3],
            edgecolor="white",
            width=0.5,
            align='center', 
        )
        ax_bar.set_ylim(0, 1)  
        ax_bar.set_title(f"[Sample {sample_idx+1}]", fontsize=14)
        ax_bar.set_xlabel("Time Step",fontsize=14)
        ax_bar.set_ylabel("Weight Sum",fontsize=14)

        ax_bar.set_xticklabels(time_steps)
        ax_bar.set_xticks(x)

        for tick in ax_bar.get_xticklabels() + ax_bar.get_yticklabels():
            tick.set_fontsize(12)  
    fig.suptitle("Attention Visualization per Sample", fontsize=24, y=0.99)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Attention 图已保存至: {save_path}")

    plt.show()
