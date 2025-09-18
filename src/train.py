import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .preprocess import get_dataloader
from .evaluate import setup_model, add_adaptation_mechanism


def calculate_nll(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)


def calibrate_acclimate(model_name, config, device):
    """
    Performs hyper-parameter calibration for ACCLIMATE.
    1. Grid search for tau on the validation set (severity 3 corruption).
    2. Calibrate kappa_max to bound the change in NLL.
    """
    print(f"--- Calibrating ACCLIMATE for {model_name} ---")
    
    val_config = config['acclimate_hparams']['validation_setup'].copy()
    val_config['model'] = model_name
    val_config['batch_size'] = config['experiment_1']['batch_sizes'][-1] # Use largest batch size for speed

    best_tau = -1
    best_acc = -1

    # 1. Grid search for tau
    print(f"Searching for optimal tau in {config['acclimate_hparams']['tau_search_space']}")
    for tau in config['acclimate_hparams']['tau_search_space']:
        # Setup model and dataloader for this tau
        source_model = setup_model(model_name, val_config, device)
        acclimate_hparams = {'tau': tau, 'kappa_max': 999, 'k': config['acclimate_hparams']['k_search_space'][-1]} # Use high kappa_max to always adapt
        model = add_adaptation_mechanism(source_model, {'method': 'ACCLIMATE', 'hparams': acclimate_hparams, 'ablation': 'full'}, device)
        model.eval()

        val_loader = get_dataloader(val_config, split='validation')

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Calibrating tau={tau}", leave=False):
                if isinstance(batch, dict):
                    images, labels = batch['image'], batch['label']
                else:
                    images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"  tau={tau:.3f}, Validation Accuracy: {accuracy:.2f}%")
        if accuracy > best_acc:
            best_acc = accuracy
            best_tau = tau

    print(f"Best tau found: {best_tau} with accuracy {best_acc:.2f}%")

    # 2. Calibrate kappa_max using the best tau
    source_model = setup_model(model_name, val_config, device)
    acclimate_hparams = {'tau': best_tau, 'kappa_max': 999, 'k': config['acclimate_hparams']['k_search_space'][-1]}
    model = add_adaptation_mechanism(source_model, {'method': 'ACCLIMATE', 'hparams': acclimate_hparams, 'ablation': 'full'}, device)
    model.eval()
    
    # Re-create source model for NLL comparison
    source_model_for_nll = setup_model(model_name, val_config, device)
    source_model_for_nll.eval()

    val_loader = get_dataloader(val_config, split='validation')
    
    delta_nlls = []
    kappas = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Calibrating kappa_max with tau={best_tau}", leave=False):
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            # NLL before adaptation
            source_logits = source_model_for_nll(images)
            nll_before = calculate_nll(source_logits, labels).item()

            # Adapt and get NLL after
            adapted_logits = model(images)
            nll_after = calculate_nll(adapted_logits, labels).item()

            # Record delta NLL and the kappa value that was used for the update
            delta_nlls.append(nll_after - nll_before)
            # Get average kappa from all NPM layers
            avg_kappa = torch.mean(torch.stack([m.kappa for m in model.modules() if hasattr(m, 'is_npm_layer')])).item()
            kappas.append(avg_kappa)
    
    delta_nlls = np.array(delta_nlls)
    kappas = np.array(kappas)

    # Find the kappa value at the percentile where delta_nll exceeds the bound
    # We set kappa_max to be just below this value to prevent such increases.
    # Sort by kappa to find the threshold
    sorted_indices = np.argsort(kappas)
    sorted_kappas = kappas[sorted_indices]
    sorted_delta_nlls = delta_nlls[sorted_indices]

    nll_bound = config['acclimate_hparams']['kappa_max_nll_bound']
    
    # Find the first kappa where the delta NLL goes above the bound
    violation_indices = np.where(sorted_delta_nlls > nll_bound)[0]
    if len(violation_indices) > 0:
        first_violation_idx = violation_indices[0]
        calibrated_kappa_max = sorted_kappas[first_violation_idx]
    else:
        # If NLL never increases badly, we can be more lenient
        calibrated_kappa_max = np.max(kappas) * 1.1 # Allow all observed kappas plus a margin

    print(f"Calibrated kappa_max: {calibrated_kappa_max:.4f} (to keep ΔNLL <= {nll_bound})")

    # For the -CBU ablation, there is nothing to calibrate here.
    # The hypernetwork is trained online in evaluate.py
    
    return {'tau': best_tau, 'kappa_max': calibrated_kappa_max}
