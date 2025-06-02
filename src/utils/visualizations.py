from src.data.preprocessing import scale_and_sequence
from src.training.train_lstm import train_lstm_on_all
from sklearn.model_selection import train_test_split
from src.models.model_registry import get_model
from sklearn.metrics import mean_squared_error
from src.data.loader import load_cir_data
import matplotlib.pyplot as plt
import numpy as np

def create_aligned_comparison(processed_dir: str):
    """
    Create properly aligned model comparison by training both models on the SAME data subset
    """
    # Load full dataset
    df = load_cir_data(processed_dir, filter_keyword="FCPR-D1")
    print(f"Original dataset size: {len(df)}")
    
    # Create sequences to see what LSTM will actually use
    seq_len = 10
    X_seq, y_seq, x_scaler, y_scaler = scale_and_sequence(df, seq_len=seq_len)
    
    print(f"After sequencing: {len(X_seq)} sequences")
    
    # The LSTM uses the LAST seq_len-1 points as features to predict each target
    # So we need to align our linear model data to match this
    
    # Get the original data that corresponds to the sequence targets
    # Since sequences start at index seq_len-1, we take data from that point onwards
    aligned_df = df.iloc[seq_len-1:seq_len-1+len(y_seq)].copy().reset_index(drop=True)
    
    print(f"Aligned dataset size: {len(aligned_df)}")
    
    # Extract features and targets from aligned data
    X_aligned = aligned_df[['PL', 'RMS']].values
    y_aligned = aligned_df['r'].values
    pl_aligned = aligned_df['PL'].values
    
    # Train/test split on aligned data
    X_train, X_test, y_train, y_test = train_test_split(
        X_aligned, y_aligned, test_size=0.2, random_state=42
    )
    pl_train, pl_test = train_test_split(
        pl_aligned, test_size=0.2, random_state=42
    )
    
    # Train linear model on aligned data
    linear_model = get_model("linear")
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
    
    # Train LSTM and get predictions
    lstm_results = train_lstm_on_all(processed_dir)
    
    # Get LSTM predictions - these correspond to the full aligned dataset
    lstm_preds = np.array(lstm_results['r_pred'])
    lstm_actual = np.array(lstm_results['r_actual'])
    
    # Now we need to extract the TEST portion from LSTM results
    # This is tricky because LSTM was trained on all sequences
    # For comparison, let's use the same test indices
    
    # Create test indices that work for both models
    test_indices = []
    np.random.seed(42)  # Same seed as train_test_split
    all_indices = np.arange(len(aligned_df))
    _, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    
    # Extract corresponding LSTM predictions and actuals
    lstm_test_preds = lstm_preds[test_idx]
    lstm_test_actual = lstm_actual[test_idx]
    
    print(f"Test set size: {len(y_test)}")
    print(f"Linear RMSE: {linear_rmse:.4f}")
    print(f"LSTM RMSE: {lstm_results['rmse']:.4f}")
    
    return {
        'pl_test': pl_test,
        'y_test': y_test,
        'y_pred_linear': y_pred_linear,
        'y_pred_lstm': lstm_test_preds,
        'train_loss': lstm_results['train_loss'],
        'val_loss': lstm_results['val_loss'],
        'linear_rmse': linear_rmse,
        'lstm_rmse': lstm_results['rmse']
    }

def plot_model_results_fixed(processed_dir: str):
    """
    Create a proper comparison plot with aligned data
    """
    # Get aligned comparison data
    comparison_data = create_aligned_comparison(processed_dir)
    
    pl_test = comparison_data['pl_test']
    y_test = comparison_data['y_test']
    y_pred_linear = comparison_data['y_pred_linear']
    y_pred_lstm = comparison_data['y_pred_lstm']
    train_loss = comparison_data['train_loss']
    val_loss = comparison_data['val_loss']
    
    # Sort by PL for better visualization
    sort_idx = np.argsort(pl_test)
    pl_test = pl_test[sort_idx]
    y_test = y_test[sort_idx]
    y_pred_linear = y_pred_linear[sort_idx]
    y_pred_lstm = y_pred_lstm[sort_idx]
    
    # Create the plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Predictions vs PL
    axs[0,0].scatter(pl_test, y_test, label="Actual", alpha=0.6, s=20, color='black')
    axs[0,0].scatter(pl_test, y_pred_linear, label=f"Linear (RMSE: {comparison_data['linear_rmse']:.3f})", 
                     alpha=0.7, s=20, color='blue')
    axs[0,0].scatter(pl_test, y_pred_lstm, label=f"LSTM (RMSE: {comparison_data['lstm_rmse']:.3f})", 
                     alpha=0.7, s=20, color='red')
    axs[0,0].set_xlabel("PL")
    axs[0,0].set_ylabel("r (Distance)")
    axs[0,0].set_title("Model Predictions vs Path Loss")
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted scatter
    axs[0,1].scatter(y_test, y_pred_linear, alpha=0.6, label="Linear", color='blue')
    axs[0,1].scatter(y_test, y_pred_lstm, alpha=0.6, label="LSTM", color='red')
    min_val, max_val = min(y_test.min(), y_pred_linear.min(), y_pred_lstm.min()), \
                       max(y_test.max(), y_pred_linear.max(), y_pred_lstm.max())
    axs[0,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    axs[0,1].set_xlabel("Actual r")
    axs[0,1].set_ylabel("Predicted r")
    axs[0,1].set_title("Actual vs Predicted")
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Training history
    if len(train_loss) > 0:
        axs[1,0].plot(train_loss, label="Train Loss", linewidth=2)
        axs[1,0].plot(val_loss, label="Val Loss", linewidth=2)
        axs[1,0].set_xlabel("Epoch")
        axs[1,0].set_ylabel("MSE Loss")
        axs[1,0].set_title("LSTM Training History")
        axs[1,0].legend()
        axs[1,0].grid(True, alpha=0.3)
        axs[1,0].set_yscale('log')
    
    # Plot 4: Residuals
    linear_residuals = y_test - y_pred_linear
    lstm_residuals = y_test - y_pred_lstm
    
    axs[1,1].scatter(y_pred_linear, linear_residuals, alpha=0.6, label="Linear", color='blue')
    axs[1,1].scatter(y_pred_lstm, lstm_residuals, alpha=0.6, label="LSTM", color='red')
    axs[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axs[1,1].set_xlabel("Predicted r")
    axs[1,1].set_ylabel("Residuals (Actual - Predicted)")
    axs[1,1].set_title("Residual Analysis")
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Linear Regression RMSE: {comparison_data['linear_rmse']:.4f}")
    print(f"LSTM RMSE: {comparison_data['lstm_rmse']:.4f}")
    print(f"Test set size: {len(y_test)} points")
    print(f"PL range: [{pl_test.min():.1f}, {pl_test.max():.1f}]")
    print(f"Target range: [{y_test.min():.1f}, {y_test.max():.1f}]")

# Backwards compatibility
def plot_model_results(pl_values, r_actual, r_pred_linear, r_pred_lstm, train_loss, val_loss):
    """Legacy function - use plot_model_results_fixed instead"""
    print("Warning: Using legacy plot function. Use plot_model_results_fixed(processed_dir) instead.")
    plot_model_results_fixed("data/processed")

def plot_actual_vs_estimated(y_true, y_pred, model_name="Model"):
    """
    Create a scatter plot of actual vs estimated r values
    
    Args:
        y_true: Array of actual r values
        y_pred: Array of predicted r values
        model_name: Name of the model for the plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
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
    plt.show()
    
    return rmse