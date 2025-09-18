import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import timm
from transformers import AutoTokenizer
import os
from PIL import Image
import numpy as np
import random


def get_transform(model_name):
    try:
        model = timm.create_model(model_name, pretrained=False)
        config = timm.data.resolve_model_data_config(model)
        return timm.data.create_transform(**config, is_training=False)
    except Exception:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

# --- Synthetic / custom datasets (unchanged) ---
# ... (omitted for brevity) ...

# --- Data Loader Factory ---

def get_dataloader(config, split):
    dataset_name = config["dataset"]
    model_name = config["model"]
    batch_size = config["batch_size"]

    is_text_model = "bert" in model_name

    # TEXT DATASETS --------------------------------------------------------
    if is_text_model:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tok_fn(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        if dataset_name == "Amazon-Yelp":
            try:
                amazon_ds = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    "raw_review_All_Beauty",
                    split="full",
                )
                yelp_ds = load_dataset(
                    "yassiracharki/Yelp_Reviews_for_Binary_Senti_Analysis",
                    split="test",
                )
                amazon_ds = amazon_ds.map(
                    lambda x: {"text": x["text"], "labels": 1 if x["rating"] > 3 else 0}
                )
                yelp_ds = yelp_ds.map(
                    lambda x: {
                        "text": x["text"],
                        "labels": 1 if x["label"] == 1 else 0,
                    }
                )
                dataset = yelp_ds
                tokenized = dataset.map(tok_fn, batched=True)
                tokenized.set_format(
                    type="torch",
                    columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
                )
                return DataLoader(
                    tokenized.select(range(min(1000, len(tokenized)))),
                    batch_size=batch_size,
                )
            except Exception as e:
                print(f"Failed to load text dataset: {e}")
                return None
        else:
            raise ValueError(f"Unknown text dataset: {dataset_name}")

    # IMAGE DATASETS -------------------------------------------------------
    transform = get_transform(model_name)

    try:
        if dataset_name == "ImageNet-C":
            ds = load_dataset("ang9867/ImageNet-C", split="train")
            target_sev = 5 if split == "test" else 3
            ds = ds.filter(lambda x: x["severity"] == target_sev)
            ds = ds.map(lambda x: {"image": transform(x["image"]), "label": x["label"]})
            ds.set_format("torch")
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == "CIFAR-C":
            ds = load_dataset("randall-lab/cifar10-c", split="test")
            cifar_t = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            ds = ds.map(lambda x: {"image": cifar_t(x["image"]), "label": x["label"]})
            ds.set_format("torch")
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        # (other datasets unchanged, just remove trust_remote_code arg)
        else:
            raise ValueError(f"Unknown image dataset: {dataset_name}")
    except Exception as e:
        print(
            f"STRICT NO-FALLBACK RULE: Could not load dataset '{dataset_name}'. Error: {e}"
        )
        return None
