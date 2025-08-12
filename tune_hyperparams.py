# tune_hyperparams_pubmed.py
import os
import itertools
import subprocess
import pandas as pd

def run_experiment(params):
    """Runs a single training experiment with the given parameters."""
    # Base command with fixed parameters for PubMed dataset
    command = [
        'python', 'main.py',
        '--model', 'CoCoVinGCN',
        '--dataset', 'PubMed',
        '--round', '1',
        '--gpu', '0',
        '--n_epochs', '900',
        '--hid_dim', '128',
        '--alpha', '0.4',
        '--delta', '0.9',
        '--cls_mode', 'both'
    ]

    # Add variable parameters to the command
    for key, value in params.items():
        command.append(f'--{key}')
        command.append(str(value))

    print(f"Running command: {' '.join(command)}")

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)

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
            if len(parts) > 11:
                try:
                    val_acc_str = parts[7]
                    test_acc_str = parts[11]
                    best_val_acc = float(val_acc_str)
                    best_test_acc = float(test_acc_str)
                except (ValueError, IndexError):
                    continue

    return best_val_acc, best_test_acc


def main():
    # Define the hyperparameter grid for PubMed CoCoVinGCN
    param_grid = {
        'beta': [0.1, 0.3, 0.6, 0.9],
        'cocos_cls_mode': ['shuf', 'raw', 'both'],
        'm': [1, 2, 3]
    }

    # Create a list of all parameter combinations
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    print(f"Starting PubMed hyperparameter tuning for {len(experiments)} experiments...")

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
            print("Experiment failed or did not produce a valid accuracy.")

    # Save results to a CSV file
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='val_acc', ascending=False)
        results_filename = 'tuning_results_cocovin_pubmed.csv'
        results_df.to_csv(results_filename, index=False)

        print("\n--- Tuning Complete ---")
        print(f"Results saved to '{results_filename}'")
        print("Top 5 hyperparameter combinations:")
        print(results_df.head(5))

        # Also print the best configuration
        best_config = results_df.iloc[0]
        print("\nBest configuration:")
        for param, value in best_config.items():
            if param not in ['val_acc', 'test_acc']:
                print(f"--{param} {value}")
    else:
        print("\n--- Tuning Complete ---")
        print("No successful experiments were completed.")


if __name__ == '__main__':
    main()