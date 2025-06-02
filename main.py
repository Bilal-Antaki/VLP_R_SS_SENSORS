from src.data.feature_engineering import create_engineered_features, select_features
from src.config import DATA_CONFIG, ANALYSIS_CONFIG, TRAINING_OPTIONS
from src.training.train_sklearn import train_all_models_enhanced
from src.training.train_lstm import train_lstm_on_all
from src.training.train_gru import train_gru_on_all
from src.data.loader import load_cir_data
from src.utils.visualizations import plot_actual_vs_estimated
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import torch
import time
import os

warnings.filterwarnings('ignore')

def create_predictions_dataframe(model_results):
    """Create a consolidated DataFrame of actual values and predictions from all models"""
    predictions_dict = {}
    
    # First, find the LSTM result to get the base length and actual values
    lstm_result = None
    for result in model_results:
        if result['name'].lower() == 'lstm' and result.get('predictions'):
            lstm_result = result
            break
    
    if lstm_result is None:
        return pd.DataFrame()
    
    # Use LSTM's actual values and predictions
    if 'y_test' in lstm_result['predictions']:
        predictions_dict['r_actual'] = lstm_result['predictions']['y_test']
        predictions_dict['r_lstm'] = lstm_result['predictions']['y_pred']
    else:
        return pd.DataFrame()
    
    # Only include other models if their predictions match LSTM length
    base_length = len(predictions_dict['r_actual'])
    
    for result in model_results:
        if result['name'].lower() == 'lstm':
            continue
            
        if result.get('predictions') and len(result['predictions']['y_test']) == base_length:
            model_name = result['name'].lower()
            predictions_dict[f'r_{model_name}'] = result['predictions']['y_pred']
    
    return pd.DataFrame(predictions_dict)


def run_analysis():
    """Run analysis with Linear, SVR, and LSTM models"""
    print("=" * 80)
    print(" " * 20 + "COMPREHENSIVE POSITION ESTIMATION ANALYSIS")
    print("=" * 80)
    
    # Create results directory if it doesn't exist
    os.makedirs('results/models', exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. Load and explore data
    print("\n1. Loading and exploring data...")
    df_list = []
    
    # Load all available datasets
    for keyword in DATA_CONFIG['datasets']:
        try:
            df_temp = load_cir_data(DATA_CONFIG['processed_dir'], filter_keyword=keyword)
            print(f"  Loaded {keyword}: {len(df_temp)} samples")
            df_list.append(df_temp)
        except:
            print(f"  {keyword} not found")
    
    # Combine all data
    df_all = pd.concat(df_list, ignore_index=True) if df_list else None
    
    if df_all is None:
        print("No data found!")
        return
    
    # 2. Feature Engineering
    print("\n2. Feature Engineering...")
    df_engineered = create_engineered_features(df_all, include_categorical=True)
    
    # Select features - exclude any coordinate-based features
    feature_cols = [col for col in df_engineered.columns 
                   if col not in ANALYSIS_CONFIG['feature_selection']['excluded_features']]
    
    X = df_engineered[feature_cols]
    y = df_engineered[DATA_CONFIG['target_column']]

    # Select best features
    selected_features = select_features(
        X, y, 
        method='correlation', 
        threshold=ANALYSIS_CONFIG['feature_selection']['correlation_threshold']
    )
    print(f"  Selected {len(selected_features)} features from {len(feature_cols)} total")
    print(f"  Top features: {selected_features[:10]}")
    
    # 3. Train all models and collect results
    print("\n3. Training all models...")
    all_model_results = []
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train and save LSTM results
    print("\nTraining LSTM model...")
    lstm_results = train_lstm_on_all(DATA_CONFIG['processed_dir'])
    
    # Plot LSTM actual vs estimated
    if TRAINING_OPTIONS['save_predictions']:
        print("\nPlotting LSTM actual vs estimated values...")
        plot_actual_vs_estimated(
            np.array(lstm_results['r_actual']),
            np.array(lstm_results['r_pred']),
            model_name="LSTM"
        )
    
    # Save LSTM model
    lstm_save_path = f'results/models/lstm_model_{timestamp}_rmse_{lstm_results["rmse"]:.4f}.pth'
    torch.save({
        'model_type': 'lstm',
        'timestamp': timestamp,
        'rmse': lstm_results['rmse'],
        'train_loss': lstm_results['train_loss'],
        'val_loss': lstm_results['val_loss'],
        'predictions': {
            'actual': lstm_results['r_actual'],
            'predicted': lstm_results['r_pred']
        }
    }, lstm_save_path)
    print(f"Saved LSTM model to {lstm_save_path}")
    
    # Clear GPU memory after LSTM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    all_model_results.append({
        'name': 'lstm',
        'type': 'RNN',
        'metrics': {
            'rmse': lstm_results['rmse']
        },
        'predictions': {
            'y_test': lstm_results['r_actual'],
            'y_pred': lstm_results['r_pred']
        } if TRAINING_OPTIONS['save_predictions'] else None,
        'training_history': {
            'train_loss': lstm_results['train_loss'],
            'val_loss': lstm_results['val_loss']
        } if TRAINING_OPTIONS['plot_training_history'] else None
    })
    
    # Train and save GRU model
    print("\nTraining GRU model...")
    gru_results = train_gru_on_all(DATA_CONFIG['processed_dir'])
    
    # Plot GRU actual vs estimated
    if TRAINING_OPTIONS['save_predictions']:
        print("\nPlotting GRU actual vs estimated values...")
        plot_actual_vs_estimated(
            np.array(gru_results['r_actual']),
            np.array(gru_results['r_pred']),
            model_name="GRU"
        )
    
    # Save GRU model
    gru_save_path = f'results/models/gru_model_{timestamp}_rmse_{gru_results["rmse"]:.4f}.pth'
    torch.save({
        'model_type': 'gru',
        'timestamp': timestamp,
        'rmse': gru_results['rmse'],
        'train_loss': gru_results['train_loss'],
        'val_loss': gru_results['val_loss'],
        'predictions': {
            'actual': gru_results['r_actual'],
            'predicted': gru_results['r_pred']
        }
    }, gru_save_path)
    print(f"Saved GRU model to {gru_save_path}")
    
    # Clear GPU memory after GRU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    all_model_results.append({
        'name': 'gru',
        'type': 'RNN',
        'metrics': {
            'rmse': gru_results['rmse']
        },
        'predictions': {
            'y_test': gru_results['r_actual'],
            'y_pred': gru_results['r_pred']
        } if TRAINING_OPTIONS['save_predictions'] else None,
        'training_history': {
            'train_loss': gru_results['train_loss'],
            'val_loss': gru_results['val_loss']
        } if TRAINING_OPTIONS['plot_training_history'] else None
    })
    
    # Train sklearn models
    print("\nTraining traditional ML models...")
    sklearn_results = train_all_models_enhanced(
        DATA_CONFIG['processed_dir'],
        include_slow_models=TRAINING_OPTIONS['include_slow_models']
    )
    
    # Save sklearn models and plot actual vs estimated
    for result in sklearn_results:
        if result['success']:
            model_save_path = f'results/models/{result["name"]}_model_{timestamp}_rmse_{result["metrics"]["rmse"]:.4f}.pkl'
            pd.to_pickle({
                'model_type': result['name'],
                'timestamp': timestamp,
                'metrics': result['metrics'],
                'predictions': {
                    'actual': result['y_test'],
                    'predicted': result['y_pred']
                }
            }, model_save_path)
            print(f"Saved {result['name']} model to {model_save_path}")
            
            # Plot actual vs estimated for sklearn models
            if TRAINING_OPTIONS['save_predictions']:
                print(f"\nPlotting {result['name']} actual vs estimated values...")
                plot_actual_vs_estimated(
                    result['y_test'],
                    result['y_pred'],
                    model_name=result['name']
                )
            
            all_model_results.append({
                'name': result['name'],
                'type': 'sklearn',
                'metrics': result['metrics'],
                'predictions': {
                    'y_test': result['y_test'],
                    'y_pred': result['y_pred']
                } if TRAINING_OPTIONS['save_predictions'] else None
            })
    
    # 4. Create visualization figures
    if TRAINING_OPTIONS['plot_training_history']:
        create_analysis_figures(all_model_results, df_all)
    
    # 5. Statistical Analysis and Results
    print("\n Results")
    print("=" * 80)
    perform_statistical_analysis(all_model_results)
    
    # 6. Save results
    if TRAINING_OPTIONS['save_predictions']:
        save_analysis_results(all_model_results)
    
    print("\nAnalysis complete. All models saved in results/models/")

def create_analysis_figures(model_results, df_raw):
    # Figure 1: Model Performance Comparison
    fig1, axes = plt.subplots(
        1, 2, 
        figsize=ANALYSIS_CONFIG['visualization']['figure_sizes']['model_comparison']
    )
    fig1.suptitle('Model Performance Comparison', fontsize=16)
    
    # Sort models by RMSE for better visualization
    model_results.sort(key=lambda x: x['metrics']['rmse'])
    
    # Model Performance Comparison (RMSE)
    model_names = [r['name'] for r in model_results]
    rmse_values = [r['metrics']['rmse'] for r in model_results]
    
    bars = axes[0].barh(model_names, rmse_values)
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('Model Performance by RMSE')
    axes[0].grid(True, alpha=ANALYSIS_CONFIG['visualization']['grid_alpha'])
    
    # Color code bars by model type
    for i, result in enumerate(model_results):
        if result['type'] == 'RNN':
            bars[i].set_color('red')
        else:
            bars[i].set_color('blue')
    
    # Training History for models that have it
    axes[1].set_title('Training History')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=ANALYSIS_CONFIG['visualization']['grid_alpha'])
    axes[1].set_yscale('log')
    
    for result in model_results:
        if result.get('training_history') and result['type'] == 'RNN':
            if 'train_loss' in result['training_history']:
                axes[1].plot(
                    result['training_history']['train_loss'],
                    label=f"{result['name']} (Train)",
                    linewidth=2
                )
            if 'val_loss' in result['training_history']:
                axes[1].plot(
                    result['training_history']['val_loss'],
                    label=f"{result['name']} (Val)",
                    linewidth=2,
                    linestyle='--'
                )
    
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def perform_statistical_analysis(model_results):
    
    # Sort results by RMSE for better readability
    sorted_results = sorted(model_results, key=lambda x: x['metrics']['rmse'])
    best_rmse = min(r['metrics']['rmse'] for r in model_results)
    
    # Individual model results
    print("\nDetailed Model Results:")
    print("-" * 40)
    
    for result in sorted_results:
        print(f"\n{result['name']}:")
        rmse = result['metrics']['rmse']
        
        # Calculate mean and std dev from predictions
        predictions = None
        if result.get('predictions'):
            if 'y_pred' in result['predictions']:
                predictions = result['predictions']['y_pred']
            elif 'r_pred' in result['predictions']:  # For LSTM results
                predictions = result['predictions']['r_pred']
        
        mean = std_dev = None
        if predictions is not None and len(predictions) > 0:
            predictions = np.array(predictions)
            mean = np.mean(predictions)
            std_dev = np.std(predictions)
        
        print(f"  RMSE: {rmse:.4f}")
        if mean is not None:
            print(f"  Mean: {mean:.4f}")
        if std_dev is not None:
            print(f"  Std: {std_dev:.4f}")
        
        # Show if this was the best model
        if rmse == best_rmse:
            print("  → Best performing model")

def save_analysis_results(model_results):
    """Save analysis results to files"""
    
    # Create results directory if it doesn't exist
    os.makedirs(ANALYSIS_CONFIG['output']['results_dir'], exist_ok=True)
    
    # Save detailed report
    report_path = os.path.join(
        ANALYSIS_CONFIG['output']['results_dir'], 
        ANALYSIS_CONFIG['output']['report_file']
    )
    
    with open(report_path, 'w') as f:
        f.write("POSITION ESTIMATION ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Sort models by RMSE
        sorted_results = sorted(model_results, key=lambda x: x['metrics']['rmse'])
        
        f.write("Model Performance Summary:\n")
        f.write("-"*40 + "\n")
        for result in sorted_results:
            f.write(f"\nModel: {result['name']} ({result['type']})\n")
            f.write(f"  RMSE: {result['metrics']['rmse']:.4f}\n")
            if result['metrics'].get('mae'):
                f.write(f"  MAE: {result['metrics']['mae']:.4f}\n")
            if result['metrics'].get('r2'):
                f.write(f"  R2: {result['metrics']['r2']:.4f}\n")


if __name__ == "__main__":
    run_analysis()