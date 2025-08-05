from src.data.data_loader import load_cir_data
from src.utils.visualizations import plot_actual_vs_estimated, plot_rmse_comparison, plot_training_history
from src.config import DATA_CONFIG
from src.training.train_sklearn import train_all_sklearn_models
from src.training.train_deep import train_all_deep_models
import os

def run_analysis():
    """Run position estimation analysis"""
    
    os.makedirs('results/plots', exist_ok=True)
    
    all_results = []
    
    # Train sklearn models
    sklearn_results = train_all_sklearn_models(DATA_CONFIG['processed_dir'])
    for result in sklearn_results:
        plot_actual_vs_estimated(
            result['y_test'],
            result['y_pred'],
            model_name=result['name'],
            save_dir="results/plots"
        )
        all_results.append(result)
    
    # Train deep learning models
    deep_results = train_all_deep_models(DATA_CONFIG['processed_dir'])
    for result in deep_results:
        plot_actual_vs_estimated(
            result['y_actual'],
            result['y_pred'],
            model_name=result['name'],
            save_dir="results/plots"
        )
        
        # Plot training history for deep models
        plot_training_history(
            result['train_loss'],
            result['val_loss'],
            result['name'],
            save_dir="results/plots"
        )
        all_results.append(result)
    
    # Create comparison plots
    plot_rmse_comparison(all_results, save_dir="results/plots")

if __name__ == "__main__":
    run_analysis()
