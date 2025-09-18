import argparse
import yaml
import os
import json
import torch
import numpy as np
from collections import defaultdict

from .train import calibrate_acclimate
from .evaluate import run_benchmark_experiment, run_ablation_experiment, run_streaming_experiment

def main():
    parser = argparse.ArgumentParser(description="Run ACCLIMATE experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment suite.')
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config['global_settings']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    device = torch.device(config['global_settings']['device'])
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("Warning: CUDA device requested but not available. Exiting.")
        return

    # --- Phase 1: Smoke Test (if requested) ---
    if args.smoke_test:
        print("--- Running Smoke Test --- ")
        # In smoke test mode, we just run the configured experiments directly.
        pass # Fall through to the main experiment logic with the smoke test config

    # --- Phase 2: Full Experiment ---
    print(f"--- Loading configuration from {config_path} ---")

    # Calibrate ACCLIMATE hyper-parameters if needed
    calibrated_params = defaultdict(dict)
    needs_calibration = False
    for exp_key, exp_config in config.items():
        if 'experiment' in exp_key and exp_config['enabled']:
            methods = exp_config.get('methods', []) + exp_config.get('baselines', [])
            if 'ACCLIMATE' in methods:
                needs_calibration = True
                models_to_calibrate = exp_config.get('models', [exp_config.get('model')])
                for model_name in models_to_calibrate:
                    if model_name:
                        calibrated_params[model_name] = {}

    if needs_calibration:
        for model_name in calibrated_params.keys():
            params = calibrate_acclimate(model_name, config, device)
            calibrated_params[model_name] = params
        print("\n--- ACCLIMATE Calibration Complete ---")
        print(json.dumps(calibrated_params, indent=2))

    # Run enabled experiments
    all_results = {}

    if config.get('experiment_1', {}).get('enabled', False):
        results = run_benchmark_experiment(config, calibrated_params)
        all_results['experiment_1'] = results

    if config.get('experiment_2', {}).get('enabled', False):
        results = run_ablation_experiment(config, calibrated_params)
        all_results['experiment_2'] = results

    if config.get('experiment_3', {}).get('enabled', False):
        results = run_streaming_experiment(config, calibrated_params)
        all_results['experiment_3'] = results

    # Save and print final results
    for exp_name, results_log in all_results.items():
        if results_log:
            file_path = os.path.join(output_dir, f"{exp_name}_results.json")
            with open(file_path, 'w') as f:
                json.dump(results_log, f, indent=4)
            
            print(f"\n--- Final Results for {exp_name} ---")
            print(f"Results saved to {file_path}")
            print(json.dumps(results_log, indent=2))

    print("\n--- All experiments finished. ---")

if __name__ == '__main__':
    main()
