import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('deep')
sns.set_context("paper", font_scale=1.2)


def plot_training_results(model='CoCoVin', dataset='Cora', seed=0, ema_alpha=0.2):
    """
    Read the metrics file and create plots showing training progress with EMA trend lines.

    Args:
        model (str): Model name (default: 'CoCoVin')
        dataset (str): Dataset name (default: 'Cora')
        seed (int): Random seed used for training (default: 0)
        ema_alpha (float): Smoothing factor for exponential moving average (default: 0.2)
    """
    # Define file path
    file_path = os.path.join('exp', 'metrics', f"{model}_{dataset}_{seed}_metrics.xlsx")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False

    # Read Excel file
    print(f"Reading metrics from {file_path}")
    df = pd.read_excel(file_path)

    # Calculate EMAs for all metrics
    ema_columns = {}
    for col in ['Train Accuracy', 'Val Accuracy', 'Test Accuracy', 'Train Loss', 'Val Loss', 'Test Loss']:
        ema_columns[f"{col}_EMA"] = df[col].ewm(alpha=ema_alpha, adjust=False).mean()

    # Add EMA columns to dataframe
    for col_name, values in ema_columns.items():
        df[col_name] = values

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join('exp', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot 1: Accuracies with EMA
    plt.figure(figsize=(10, 6))
    # Original accuracy lines (thinner)
    plt.plot(df['Epoch'], df['Train Accuracy'], alpha=0.5, linewidth=1, label='Train Accuracy')
    plt.plot(df['Epoch'], df['Val Accuracy'], alpha=0.5, linewidth=1, label='Val Accuracy')
    plt.plot(df['Epoch'], df['Test Accuracy'], alpha=0.5, linewidth=1, label='Test Accuracy')

    # EMA trend lines (thicker)
    plt.plot(df['Epoch'], df['Train Accuracy_EMA'], linewidth=2, label='Train Accuracy (EMA)')
    plt.plot(df['Epoch'], df['Val Accuracy_EMA'], linewidth=2, label='Val Accuracy (EMA)')
    plt.plot(df['Epoch'], df['Test Accuracy_EMA'], linewidth=2, label='Test Accuracy (EMA)')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model} Accuracy on {dataset} (seed={seed})')
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(plots_dir, f"{model}_{dataset}_{seed}_accuracy.png")
    plt.savefig(acc_path)
    plt.show()
    print(f"Accuracy plot saved to {acc_path}")

    # Plot 2: Losses with EMA
    plt.figure(figsize=(10, 6))
    # Original loss lines (thinner)
    plt.plot(df['Epoch'], df['Train Loss'], alpha=0.5, linewidth=1, label='Train Loss')
    plt.plot(df['Epoch'], df['Val Loss'], alpha=0.5, linewidth=1, label='Val Loss')
    plt.plot(df['Epoch'], df['Test Loss'], alpha=0.5, linewidth=1, label='Test Loss')

    # EMA trend lines (thicker)
    plt.plot(df['Epoch'], df['Train Loss_EMA'], linewidth=2, label='Train Loss (EMA)')
    plt.plot(df['Epoch'], df['Val Loss_EMA'], linewidth=2, label='Val Loss (EMA)')
    plt.plot(df['Epoch'], df['Test Loss_EMA'], linewidth=2, label='Test Loss (EMA)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model} Loss on {dataset} (seed={seed})')
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(plots_dir, f"{model}_{dataset}_{seed}_loss.png")
    plt.savefig(loss_path)
    plt.show()
    print(f"Loss plot saved to {loss_path}")

    # Plot 3: Combined plot with EMA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot accuracies with EMA on the left subplot
    # Original data (thinner lines)
    ax1.plot(df['Epoch'], df['Train Accuracy'], alpha=0.5, linewidth=1, label='Train')
    ax1.plot(df['Epoch'], df['Val Accuracy'], alpha=0.5, linewidth=1, label='Val')
    ax1.plot(df['Epoch'], df['Test Accuracy'], alpha=0.5, linewidth=1, label='Test')

    # EMA trend lines (thicker)
    ax1.plot(df['Epoch'], df['Train Accuracy_EMA'], linewidth=2, label='Train (EMA)')
    ax1.plot(df['Epoch'], df['Val Accuracy_EMA'], linewidth=2, label='Val (EMA)')
    ax1.plot(df['Epoch'], df['Test Accuracy_EMA'], linewidth=2, label='Test (EMA)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Epochs')
    ax1.legend()

    # Plot losses with EMA on the right subplot
    # Original data (thinner lines)
    ax2.plot(df['Epoch'], df['Train Loss'], alpha=0.5, linewidth=1, label='Train')
    ax2.plot(df['Epoch'], df['Val Loss'], alpha=0.5, linewidth=1, label='Val')
    ax2.plot(df['Epoch'], df['Test Loss'], alpha=0.5, linewidth=1, label='Test')

    # EMA trend lines (thicker)
    ax2.plot(df['Epoch'], df['Train Loss_EMA'], linewidth=2, label='Train (EMA)')
    ax2.plot(df['Epoch'], df['Val Loss_EMA'], linewidth=2, label='Val (EMA)')
    ax2.plot(df['Epoch'], df['Test Loss_EMA'], linewidth=2, label='Test (EMA)')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss over Epochs')
    ax2.legend()

    plt.suptitle(f'{model} Training on {dataset} (seed={seed})')
    plt.tight_layout()
    combined_path = os.path.join(plots_dir, f"{model}_{dataset}_{seed}_combined.png")
    plt.savefig(combined_path)
    plt.show()
    print(f"Combined plot saved to {combined_path}")

    return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Plot training results with EMA trend lines')
    parser.add_argument('--model', type=str, default='CoCoVin', help='Model name')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Starting seed value')
    parser.add_argument('--num_round', type=int, default=1, help='Number of rounds/seeds to plot')
    parser.add_argument('--ema_alpha', type=float, default=0.2, help='Smoothing factor for EMA (0-1)')

    return parser.parse_args()


def plot_multiple_seeds(model, dataset, seed, num_round, ema_alpha):
    """Plot results for multiple seeds"""
    print(f"Plotting results for {model} on {dataset}")
    print(f"Processing {num_round} seeds starting from seed={seed}")

    success_count = 0

    for i in range(num_round):
        current_seed = seed + i
        print(f"\nProcessing seed {current_seed} ({i+1}/{num_round}):")
        success = plot_training_results(model, dataset, current_seed, ema_alpha)
        if success:
            success_count += 1

    print(f"\nPlotting complete. Successfully processed {success_count}/{num_round} seeds.")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Plot results for multiple seeds
    plot_multiple_seeds(
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        num_round=args.num_round,
        ema_alpha=args.ema_alpha
    )