# _ver1/tune_hyperparams.py
import os
import itertools
import subprocess
import pandas as pd

def run_experiment(params):
    """Runs a single training experiment with the given parameters."""
    command = [
        'python', 'main.py',
        '--model', 'CoCoVinGCN',
        '--dataset', 'Cora',
        '--round', '1',  # Use 1 round for tuning, maybe more for final eval
        '--gpu', '0',
        '--n_epochs', '300' # Ensure consistent number of epochs
    ]

    # Add parameters to the command
    for key, value in params.items():
        command.append(f'--{key}')
        command.append(str(value))

    print(f"Running command: {' '.join(command)}")

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Error running experiment:")
        print(result.stderr)
        return None, None

    # Extract the best validation and test accuracy from the output
    best_val_acc = None
    best_test_acc = None
    for line in result.stdout.split('\n'):
        if 'new best validation accuracy' in line:
            parts = line.split()
            # Example line: epoch 035 | new best validation accuracy 0.8140 - test accuracy 0.8050
            # Corrected indices from 6 and 9 to 7 and 11
            val_acc_str = parts[7]
            test_acc_str = parts[11]
            # Keep updating to get the final best accuracy
            best_val_acc = float(val_acc_str)
            best_test_acc = float(test_acc_str)

    return best_val_acc, best_test_acc


def main():
    # Define the hyperparameter grid
    param_grid = {
        'alpha': [0.5, 0.8, 1.0],
        'gamma': [0.3, 0.6, 0.9],
        'beta': [0.1, 0.3, 0.6, 0.9]
    }

    # Create a list of all parameter combinations
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    print(f"Starting hyperparameter tuning for {len(experiments)} experiments...")

    for i, params in enumerate(experiments):
        print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
        print(f"Parameters: {params}")

        val_acc, test_acc = run_experiment(params)

        if val_acc is not None and test_acc is not None:
            results.append({
                **params,
                'val_acc': val_acc,
                'test_acc': test_acc
            })
            print(f"Result: Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")
        else:
            print("Experiment failed.")

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='val_acc', ascending=False)
    results_filename = 'tuning_results_cocovin.csv'
    results_df.to_csv(results_filename, index=False)

    print("\n--- Tuning Complete ---")
    print(f"Results saved to '{results_filename}'")
    print("Top 5 hyperparameter combinations:")
    print(results_df.head(5))


if __name__ == '__main__':
    main()