from src.data.data_loader import load_cir_data
from src.utils.visualizations import plot_actual_vs_estimated, plot_rmse_comparison
from src.config import DATA_CONFIG
from src.training.train_sklearn import train_all_models_enhanced
from src.training.train_lstm import train_lstm_on_all
from src.training.train_gru import train_gru_on_all
from src.training.train_rnn import train_rnn_on_all
from src.evaluation.metrics import print_rmse
import os
import warnings

warnings.filterwarnings('ignore')

def run_analysis():
    """Run simplified analysis with all models"""
    
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)
    
    print("Starting Position Estimation Analysis")
    print("=" * 50)
    
    # Dictionary to store RMSE results for comparison
    rmse_results = {}
    
    # 1. Load data
    try:
        df = load_cir_data(DATA_CONFIG['processed_dir'])
        print(f"Loaded dataset: {len(df)} samples")
        print(f"Features: {DATA_CONFIG['feature_columns']}")
        print(f"Target: {DATA_CONFIG['target_column']}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Train sklearn models
    print("\n" + "=" * 50)
    print("Training sklearn models...")
    sklearn_results = train_all_models_enhanced(DATA_CONFIG['processed_dir'])
    
    # Plot sklearn results and print RMSE
    for result in sklearn_results:
        if result['success']:
            plot_actual_vs_estimated(
                result['y_test'],
                result['y_pred'],
                model_name=result['name'],
                save_dir="results/plots"
            )
            rmse = print_rmse(result['name'], result['y_test'], result['y_pred'])
            rmse_results[result['name']] = rmse
    
    # 3. Train LSTM model
    print("\n" + "=" * 50)
    print("Training LSTM model...")
    try:
        lstm_results = train_lstm_on_all(DATA_CONFIG['processed_dir'])
        plot_actual_vs_estimated(
            lstm_results['r_actual'],
            lstm_results['r_pred'],
            model_name="LSTM",
            save_dir="results/plots"
        )
        rmse = print_rmse("LSTM", lstm_results['r_actual'], lstm_results['r_pred'])
        rmse_results["LSTM"] = rmse
    except Exception as e:
        print(f"LSTM training failed: {e}")
    
    # 4. Train GRU model
    print("\n" + "=" * 50)
    print("Training GRU model...")
    try:
        gru_results = train_gru_on_all(DATA_CONFIG['processed_dir'])
        plot_actual_vs_estimated(
            gru_results['r_actual'],
            gru_results['r_pred'],
            model_name="GRU",
            save_dir="results/plots"
        )
        rmse = print_rmse("GRU", gru_results['r_actual'], gru_results['r_pred'])
        rmse_results["GRU"] = rmse
    except Exception as e:
        print(f"GRU training failed: {e}")
    
    # 5. Train RNN model
    print("\n" + "=" * 50)
    print("Training RNN model...")
    try:
        rnn_results = train_rnn_on_all(DATA_CONFIG['processed_dir'])
        plot_actual_vs_estimated(
            rnn_results['r_actual'],
            rnn_results['r_pred'],
            model_name="RNN",
            save_dir="results/plots"
        )
        rmse = print_rmse("RNN", rnn_results['r_actual'], rnn_results['r_pred'])
        rmse_results["RNN"] = rmse
    except Exception as e:
        print(f"RNN training failed: {e}")
    
    # 6. Create RMSE comparison plot
    print("\n" + "=" * 50)
    print("Creating RMSE comparison plot...")
    if rmse_results:
        plot_rmse_comparison(rmse_results, save_dir="results/plots")
        print("RMSE comparison plot saved as results/plots/rmse_comparison.png")
        
        # Print summary of results
        print("\nRMSE Summary:")
        print("-" * 30)
        for model, rmse in sorted(rmse_results.items(), key=lambda x: x[1]):
            print(f"{model}: {rmse:.4f}")
        
        best_model = min(rmse_results, key=rmse_results.get)
        print(f"\nBest performing model: {best_model} (RMSE: {rmse_results[best_model]:.4f})")
    
    print("\n" + "=" * 50)
    print("Analysis complete. Check results/plots/ for all plots including rmse_comparison.png")

if __name__ == "__main__":
    run_analysis()