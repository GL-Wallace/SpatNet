import matplotlib.pyplot as plt
import seaborn as sns
import os

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