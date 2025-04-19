import torch
import matplotlib.pyplot as plt

def debug_forward_output(y_pred, y_input):
    """检查前向传播输出是否正常"""
    print(f"[Forward] y_pred: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}, min={y_pred.min():.4f}, max={y_pred.max():.4f}")
    print(f"[Forward] y_input: mean={y_input.mean():.4f}, std={y_input.std():.4f}, min={y_input.min():.4f}, max={y_input.max():.4f}")
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        print("⚠️ Forward output contains NaN or Inf!")

def check_gradient_none(model):
    """检查是否有参数没有梯度传播"""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"🚨 [None Grad] {name} has no gradient!")

def print_gradient_stats(model):
    """打印每一层参数梯度的统计值"""
    print("🔍 Gradient stats per parameter:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name:40s} | mean: {grad.mean():.6f}, std: {grad.std():.6f}, max: {grad.max():.6f}, min: {grad.min():.6f}")

def visualize_gradients(model, title="Gradient Distribution"):
    """绘制所有参数梯度的直方图"""
    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
    if grads:
        all_grads = torch.cat(grads)
        plt.figure(figsize=(6, 4))
        plt.hist(all_grads, bins=100, alpha=0.7, color='steelblue')
        plt.title(title)
        plt.xlabel("Gradient value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def run_debug_checks(model, y_pred, y_input, debug=False):
    if debug:
        debug_forward_output(y_pred, y_input)
        check_gradient_none(model)
        print_gradient_stats(model)
        # visualize_gradients(model)
