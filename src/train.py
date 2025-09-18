import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .preprocess import get_dataloader
from .evaluate import setup_model, add_adaptation_mechanism


def _unpack_batch(batch):
    """Utility to extract (images, labels) from either a tuple or a dict returned
    by a HuggingFace dataset DataLoader."""
    if isinstance(batch, dict):
        # HF datasets -> dict with keys `image`, `label`
        return batch["image"], batch["label"]
    # torchvision style -> tuple
    return batch


def calculate_nll(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)


def calibrate_acclimate(model_name, config, device):
    """
    Performs hyper-parameter calibration for ACCLIMATE.
    1. Grid search for tau on the validation set (severity 3 corruption).
    2. Calibrate kappa_max to bound the change in NLL.
    """
    print(f"--- Calibrating ACCLIMATE for {model_name} ---")

    val_config = config["acclimate_hparams"]["validation_setup"].copy()
    val_config["model"] = model_name
    # Use the largest batch-size configured for speed
    val_config["batch_size"] = config["experiment_1"]["batch_sizes"][-1]

    best_tau = -1.0
    best_acc = -1.0

    print(
        "Searching for optimal tau in"
        f" {config['acclimate_hparams']['tau_search_space']}"
    )

    for tau in config["acclimate_hparams"]["tau_search_space"]:
        # 1. Build an ACCLIMATE-wrapped model with the current tau
        source_model = setup_model(model_name, val_config, device)
        acclimate_hparams = {
            "tau": tau,
            "kappa_max": 999.0,  # allow all updates during calibration
            "k": config["acclimate_hparams"]["k_search_space"][-1],
        }
        model_wrapper = add_adaptation_mechanism(
            source_model,
            {"method": "ACCLIMATE", "hparams": acclimate_hparams, "ablation": "full"},
            device,
        )
        model_wrapper.eval()  # use the new TTAMethod.eval()

        val_loader = get_dataloader(val_config, split="validation")
        if val_loader is None:
            raise RuntimeError("Validation dataloader could not be created – aborting calibration.")

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Calibrating tau={tau}", leave=False):
                images, labels = _unpack_batch(batch)
                images, labels = images.to(device), labels.to(device)
                outputs = model_wrapper(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"  tau={tau:.3f}, Validation Accuracy: {accuracy:.2f}%")
        if accuracy > best_acc:
            best_acc = accuracy
            best_tau = tau

    print(f"Best tau found: {best_tau} with accuracy {best_acc:.2f}%")

    # 2. Calibrate kappa_max with the best tau
    source_model = setup_model(model_name, val_config, device)
    acclimate_hparams = {
        "tau": best_tau,
        "kappa_max": 999.0,
        "k": config["acclimate_hparams"]["k_search_space"][-1],
    }
    model_wrapper = add_adaptation_mechanism(
        source_model,
        {"method": "ACCLIMATE", "hparams": acclimate_hparams, "ablation": "full"},
        device,
    )
    model_wrapper.eval()

    # Source (non-adapted) model for ΔNLL comparison
    src_for_nll = setup_model(model_name, val_config, device)
    src_for_nll.eval()

    val_loader = get_dataloader(val_config, split="validation")

    delta_nlls = []
    kappas = []
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc=f"Calibrating kappa_max with tau={best_tau}",
            leave=False,
        ):
            images, labels = _unpack_batch(batch)
            images, labels = images.to(device), labels.to(device)

            # Before adaptation
            nll_before = calculate_nll(src_for_nll(images), labels).item()
            # After (ACCLIMATE) adaptation
            nll_after = calculate_nll(model_wrapper(images), labels).item()

            delta_nlls.append(nll_after - nll_before)
            # collect average κ used in this forward
            avg_kappa = torch.mean(
                torch.stack(
                    [
                        m.kappa
                        for m in model_wrapper.model.modules()
                        if hasattr(m, "is_npm_layer")
                    ]
                )
            ).item()
            kappas.append(avg_kappa)

    delta_nlls = np.array(delta_nlls)
    kappas = np.array(kappas)

    nll_bound = config["acclimate_hparams"]["kappa_max_nll_bound"]

    # Sort κ and find first violation of ΔNLL bound
    order = np.argsort(kappas)
    sorted_kappa = kappas[order]
    sorted_delta = delta_nlls[order]

    viol = np.where(sorted_delta > nll_bound)[0]
    if len(viol) > 0:
        calibrated_kappa_max = sorted_kappa[viol[0]]
    else:
        calibrated_kappa_max = float(np.max(kappas) * 1.1)  # 10 % head-room

    print(
        f"Calibrated kappa_max: {calibrated_kappa_max:.4f} "
        f"(to keep ΔNLL ≤ {nll_bound})"
    )

    return {"tau": best_tau, "kappa_max": calibrated_kappa_max}
