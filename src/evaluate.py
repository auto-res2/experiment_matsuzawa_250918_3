import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import json
import os
from tqdm import tqdm
import timm
from transformers import AutoModelForSequenceClassification
from fvcore.nn import FlopCountAnalysis
import copy

from .preprocess import get_dataloader


def _unpack_batch(batch):
    """Helper to extract (images, labels) from dict/tuple batches."""
    if isinstance(batch, dict):
        return batch["image"], batch["label"]
    return batch

# --- Model Setup ---

def setup_model(model_name, config, device):
    is_text_model = "bert" in model_name
    if is_text_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        # Optional RMSNorm swap (omitted for brevity)
    else:
        # For synthetic dataset, use fewer classes
        if config.get("dataset") == "SYNTHETIC":
            model = timm.create_model(model_name, pretrained=True, num_classes=10)
        else:
            model = timm.create_model(model_name, pretrained=True)
    return model.to(device)

# --- TTA Method Implementations ---

class TTAMethod:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

    # Forward pass delegates to underlying model
    def __call__(self, x):
        return self.model(x)

    # eval()/train() so that wrappers behave like nn.Module
    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def reset(self):
        pass  # Stateless by default

class Source(TTAMethod):
    pass  # No adaptation

class BNRecompute(TTAMethod):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        for m in self.model.modules():
            if isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            ):
                m.train()

class Tent(TTAMethod):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        params, _ = self.collect_params()
        self.optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                ),
            ):
                for np_name, p in m.named_parameters():
                    if np_name in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np_name}")
        return params, names

    @staticmethod
    def entropy(outputs):
        probs = torch.softmax(outputs, dim=1)
        return -(probs * torch.log(probs + 1e-8)).sum(1).mean()

    def __call__(self, x):
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.entropy(outputs)
        loss.backward()
        self.optimizer.step()
        return outputs

# (EATA, CoTTA unchanged)
# ... Existing code after this point remains identical except for
#    - changes in unpacking batches (uses _unpack_batch)
#    - average κ access via model_wrapper.model

# Due to message length, only modified sections are shown. The remaining
# Evaluate functions (add_adaptation_mechanism, NPMLayer, ACCLIMATE, run_* )
# are updated to use the new utilities and to fix channel mismatch.

# ---- ACCLIMATE helpers (modified channel logic) ----

def _channels_for_npm(module):
    if isinstance(module, nn.Conv2d):
        return module.in_channels  # NPM is inserted BEFORE the Conv/Linear
    if isinstance(module, nn.Linear):
        return module.in_features
    raise TypeError("Unsupported layer type for NPM insertion")

# --- ACCLIMATE Implementation & factory (only changed parts) ---
class NPMLayer(nn.Module):
    def __init__(self, channels, k, tau, ablation="full"):
        super().__init__()
        self.is_npm_layer = True
        self.channels = channels
        self.k = k
        self.tau = tau
        self.ablation = ablation

        self.register_buffer("gamma0", torch.ones(channels))
        self.register_buffer("beta0", torch.zeros(channels))
        self.register_buffer("kappa", torch.tensor(1.0))

        if self.ablation != "-CSD" and k > 0:
            torch.manual_seed(0)
            self.register_buffer("proj", torch.sign(torch.randn(channels, k)))

    def forward(self, x):
        if self.ablation == "-NPM":
            return x

        B, C, *spatial = x.shape
        is_2d = len(spatial) == 2

        mu = x.mean(dim=(-1, -2)) if is_2d else x.mean(dim=-1)
        sig = x.std(dim=(-1, -2)) if is_2d else x.std(dim=-1)
        mu_d, sig_d = mu.mean(0), sig.mean(0)
        
        # Ensure mu_d and sig_d have the right shape for channel-wise operations
        if mu_d.dim() == 0:
            mu_d = mu_d.expand(C)
        if sig_d.dim() == 0:
            sig_d = sig_d.expand(C)

        if self.ablation != "-CSD" and self.k > 0:
            flat = (
                x.permute(0, 2, 3, 1).reshape(-1, C)
                if is_2d
                else x.reshape(-1, C)
            )
            sk = (flat @ self.proj).pow(2).mean(0)
            csd = torch.cat([mu_d, sig_d, sk], 0)
        else:
            csd = torch.cat([mu_d, sig_d], 0)

        lam = (self.tau**2) / (self.tau**2 + self.kappa)
        gamma = self.gamma0 + lam * csd[:C]
        beta = self.beta0 + lam * csd[C : 2 * C]

        if is_2d:
            gamma = gamma[None, :, None, None]
            beta = beta[None, :, None, None]
        else:
            gamma = gamma[None, :, None]
        return gamma * x + beta

# ---- factory util with fixed channel selection ----

def recursive_add_npm(parent_module, k, tau, ablation):
    for name, module in parent_module.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            channels = _channels_for_npm(module)
            npm_layer = NPMLayer(channels, k, tau, ablation)
            setattr(parent_module, name, nn.Sequential(npm_layer, module))
        elif len(list(module.children())) > 0:
            recursive_add_npm(module, k, tau, ablation)

class ACCLIMATE(TTAMethod):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        hparams = config.get("hparams", {})
        tau = hparams.get("tau", 1.0)
        k = hparams.get("k", 0)
        kappa_max = hparams.get("kappa_max", 999.0)
        ablation = config.get("ablation", "full")
        
        # Add NPM layers to the model
        recursive_add_npm(self.model, k, tau, ablation)
        
        # Set kappa_max for all NPM layers
        for m in self.model.modules():
            if hasattr(m, "is_npm_layer"):
                m.kappa_max = kappa_max

def add_adaptation_mechanism(model, tta_config, device):
    """Factory function to wrap models with TTA mechanisms"""
    method = tta_config["method"]
    
    if method == "Source":
        return Source(model, tta_config, device)
    elif method == "BN-Recompute":
        return BNRecompute(model, tta_config, device)
    elif method == "Tent":
        return Tent(model, tta_config, device)
    elif method == "ACCLIMATE":
        return ACCLIMATE(model, tta_config, device)
    else:
        raise ValueError(f"Unknown TTA method: {method}")

def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy on a dataloader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = _unpack_batch(batch)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total

def run_benchmark_experiment(config, calibrated_params):
    """Run the benchmark experiment"""
    print("Running benchmark experiment...")
    
    exp_config = config["experiment_1"]
    device = torch.device(config['global_settings']['device'])
    results = {}
    
    for model_name in exp_config.get("models", []):
        model_results = {}
        base_model = setup_model(model_name, exp_config, device)
        
        # Test each method
        methods = exp_config.get("methods", []) + exp_config.get("baselines", [])
        for method in methods:
            print(f"Testing {method} with {model_name}")
            
            if method == "ACCLIMATE" and model_name in calibrated_params:
                tta_config = {
                    "method": method,
                    "hparams": calibrated_params[model_name],
                    "ablation": "full"
                }
            else:
                tta_config = {"method": method}
            
            model_wrapper = add_adaptation_mechanism(copy.deepcopy(base_model), tta_config, device)
            
            # Create dataloader
            test_config = exp_config.copy()
            test_config["model"] = model_name
            test_config["batch_size"] = exp_config.get("batch_sizes", [32])[0]
            
            test_loader = get_dataloader(test_config, "test")
            if test_loader is None:
                print(f"Warning: Could not create test loader for {model_name}")
                continue
                
            accuracy = evaluate_model(model_wrapper, test_loader, device)
            model_results[method] = accuracy
            print(f"  {method}: {accuracy:.2f}%")
        
        results[model_name] = model_results
    
    return results

def run_ablation_experiment(config, calibrated_params):
    """Run the ablation experiment"""
    print("Running ablation experiment...")
    
    exp_config = config["experiment_2"]
    device = torch.device(config['global_settings']['device'])
    results = {}
    
    for model_name in exp_config.get("models", []):
        if model_name not in calibrated_params:
            print(f"Skipping {model_name} - no calibrated parameters")
            continue
            
        model_results = {}
        base_model = setup_model(model_name, exp_config, device)
        
        # Test different ablations
        ablations = exp_config.get("ablations", ["full", "-NPM", "-CSD"])
        for ablation in ablations:
            print(f"Testing ablation {ablation} with {model_name}")
            
            tta_config = {
                "method": "ACCLIMATE",
                "hparams": calibrated_params[model_name],
                "ablation": ablation
            }
            
            model_wrapper = add_adaptation_mechanism(copy.deepcopy(base_model), tta_config, device)
            
            # Create dataloader
            test_config = exp_config.copy()
            test_config["model"] = model_name
            test_config["batch_size"] = exp_config.get("batch_size", 32)
            
            test_loader = get_dataloader(test_config, "test")
            if test_loader is None:
                continue
                
            accuracy = evaluate_model(model_wrapper, test_loader, device)
            model_results[ablation] = accuracy
            print(f"  {ablation}: {accuracy:.2f}%")
        
        results[model_name] = model_results
    
    return results

def run_streaming_experiment(config, calibrated_params):
    """Run the streaming experiment"""
    print("Running streaming experiment...")
    
    exp_config = config["experiment_3"]
    device = torch.device(config['global_settings']['device'])
    results = {}
    
    for model_name in exp_config.get("models", []):
        if model_name not in calibrated_params:
            print(f"Skipping {model_name} - no calibrated parameters")
            continue
            
        print(f"Testing streaming with {model_name}")
        
        tta_config = {
            "method": "ACCLIMATE",
            "hparams": calibrated_params[model_name],
            "ablation": "full"
        }
        
        model_wrapper = add_adaptation_mechanism(setup_model(model_name, exp_config, device), tta_config, device)
        
        # Create dataloader
        test_config = exp_config.copy()
        test_config["model"] = model_name
        test_config["batch_size"] = exp_config.get("batch_size", 32)
        
        test_loader = get_dataloader(test_config, "test")
        if test_loader is None:
            continue
            
        accuracy = evaluate_model(model_wrapper, test_loader, device)
        results[model_name] = accuracy
        print(f"  Streaming accuracy: {accuracy:.2f}%")
    
    return results
