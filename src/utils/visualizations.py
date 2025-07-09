import matplotlib.pyplot as plt
import numpy as np
import os

def plot_actual_vs_estimated(y_true, y_pred, model_name="Model", save_dir="results/plots"):
    """
    Create and save scatter plots of actual vs estimated r values
    
    Args:
        y_true: Array of actual r values
        y_pred: Array of predicted r values
        model_name: Name of the model for the plot title
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays if they're lists
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, label=f"{model_name} Predictions")
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Add labels and title
    plt.xlabel("Actual r")
    plt.ylabel("Estimated r")
    plt.title(f"{model_name}: Actual vs Estimated r Values\nRMSE: {rmse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make plot square
    plt.axis('equal')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_actual_vs_estimated.png"))
    plt.close()
    
    return rmse

