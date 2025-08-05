import matplotlib.pyplot as plt
import numpy as np
import os

def plot_actual_vs_estimated(y_true, y_pred, model_name="Model", save_dir="results/plots"):
    """Create and save scatter plots of actual vs estimated values"""
    os.makedirs(save_dir, exist_ok=True)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, label=f"{model_name} Predictions")
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    plt.xlabel("Actual X")
    plt.ylabel("Estimated X")
    plt.title(f"{model_name}: Actual vs Estimated X Values\nRMSE: {rmse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_actual_vs_estimated.png"))
    plt.close()
    
    return rmse

def plot_training_history(train_loss, val_loss, model_name, save_dir="results/plots"):
    """Plot and save training history"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', alpha=0.8)
    plt.plot(val_loss, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_training_history.png"))
    plt.close()

def plot_rmse_comparison(results, save_dir="results/plots"):
    """Create and save a bar plot comparing RMSE values and runtimes"""
    os.makedirs(save_dir, exist_ok=True)
    
    models = [r['name'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    runtimes = [r['runtime'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE comparison
    bars1 = ax1.bar(models, rmse_values, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
    for bar, rmse in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel("Models", fontsize=12, fontweight='bold')
    ax1.set_ylabel("RMSE", fontsize=12, fontweight='bold')
    ax1.set_title("RMSE Comparison", fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Runtime comparison
    bars2 = ax2.bar(models, runtimes, color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
    for bar, runtime in zip(bars2, runtimes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{runtime:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel("Models", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Runtime (seconds)", fontsize=12, fontweight='bold')
    ax2.set_title("Runtime Comparison", fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
