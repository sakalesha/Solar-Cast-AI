import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(csv_file='results.csv', output_dir='plots'):
    """
    Generates comparison plots for Baseline vs Attention models across all cities.
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run training first.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_file)
    
    # 1. RMSE Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='city', y='rmse', hue='model', palette='viridis')
    plt.title('RMSE Comparison: Baseline vs Temporal Attention', fontsize=14)
    plt.ylabel('Root Mean Squared Error (W/m²)')
    plt.xlabel('City')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'))
    plt.show()

    # 2. MAE Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='city', y='mae', hue='model', palette='magma')
    plt.title('MAE Comparison: Baseline vs Temporal Attention', fontsize=14)
    plt.ylabel('Mean Absolute Error (W/m²)')
    plt.xlabel('City')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'))
    plt.show()

    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    plot_results()
