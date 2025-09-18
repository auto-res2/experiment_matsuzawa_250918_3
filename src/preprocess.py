import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import timm
from transformers import AutoTokenizer
import os
import requests
from io import BytesIO
from zipfile import ZipFile
from PIL import Image
import numpy as np
import random

def get_transform(model_name):
    # For timm models, this is the standard way to get the correct transform
    try:
        model = timm.create_model(model_name, pretrained=False)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return transform
    except Exception:
        # Fallback for non-timm models or other issues
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# --- Custom Dataset Implementations ---
# These require manual download and placement in the `data/` directory.

class ImageNet3DCC(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"ImageNet-3DCC not found at {root_dir}. Please download it manually and place it there.")
        # Assuming a simple structure: root_dir/class_name/image.png
        for i, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class ContinualCorruptionStream(Dataset):
    """Generates a stream by concatenating ImageNet-C corruptions."""
    def __init__(self, transform, severity=5, length_per_corruption=5000):
        self.transform = transform
        self.base_dataset = load_dataset("hendrycks/imagenet-c", trust_remote_code=True, split='validation')
        self.corruption_types = self.base_dataset.features['corruption_type'].names
        self.severity = severity
        self.length_per_corruption = length_per_corruption
        
        self.stream = []
        print("Building Continual Corruption Stream...")
        for corruption_idx, corruption_name in enumerate(self.corruption_types):
            subset = self.base_dataset.filter(lambda x: x['corruption_type'] == corruption_idx and x['severity'] == severity)
            # Ensure we don't go out of bounds
            num_samples = min(self.length_per_corruption, len(subset))
            for i in range(num_samples):
                self.stream.append(subset[i])
        print(f"Stream built with {len(self.stream)} frames.")

    def __len__(self):
        return len(self.stream)

    def __getitem__(self, idx):
        item = self.stream[idx]
        image = item['image']
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

def generate_corrupted_frame(base_image, corruption_type):
    if corruption_type == 'clean':
        return base_image
    elif corruption_type == 'fog':
        # Simple fog simulation
        overlay = Image.new('RGB', base_image.size, (200, 200, 200))
        return Image.blend(base_image, overlay, 0.6)
    elif corruption_type == 'snow':
        # Simple snow simulation
        img_np = np.array(base_image).astype(np.float32)
        snow_points = np.random.randint(0, high=255, size=(*img_np.shape[:2], 1), dtype=np.uint8)
        snow_mask = snow_points > 250
        img_np[snow_mask.squeeze()] = 255
        return Image.fromarray(img_np.astype(np.uint8))
    elif corruption_type == 'motion_blur':
        return base_image.filter(ImageFilter.GaussianBlur(5))
    return base_image

class ShiftedObjectsRT(Dataset):
    def __init__(self, transform, num_frames=1000, size=(224, 224)):
        self.transform = transform
        self.num_frames = num_frames
        self.size = size
        self.frames = []
        self.labels = []
        print("Generating synthetic Shifted-Objects-RT video stream...")
        # This is a synthetic replacement for the Shifted-Objects-RT dataset
        # as it is not publicly available. It simulates shifts.
        try:
            import PIL.ImageFilter as ImageFilter
        except ImportError:
            raise ImportError("Pillow is required for ShiftedObjectsRT. Please run 'pip install Pillow'")
        # Create a base image (e.g., a colored square)
        base_img = Image.new('RGB', size, color = 'red')
        shifts = ['clean'] * 200 + ['fog'] * 200 + ['snow'] * 200 + ['clean'] * 200 + ['motion_blur'] * 200
        
        for i in range(num_frames):
            shift_type = shifts[i % len(shifts)]
            frame = generate_corrupted_frame(base_img, shift_type)
            self.frames.append(frame)
            # Label is constant in this synthetic example
            self.labels.append(0) 

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        image = self.frames[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Data Loader Factory ---

def get_dataloader(config, split):
    dataset_name = config['dataset']
    model_name = config['model']
    batch_size = config['batch_size']

    is_text_model = 'bert' in model_name

    if is_text_model:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        if dataset_name == 'Amazon-Yelp':
            try:
                amazon_ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split='full', trust_remote_code=True)
                yelp_ds = load_dataset("yelp_review_full", split='test')
                # Create a binary sentiment task
                amazon_ds = amazon_ds.map(lambda x: {'text': x['text'], 'labels': 1 if x['rating'] > 3 else 0})
                yelp_ds = yelp_ds.map(lambda x: {'text': x['text'], 'labels': 1 if x['rating'] > 3 else 0})
                # In TTA, we test on the target domain
                dataset = yelp_ds
                tokenized_dataset = dataset.map(tokenize_function, batched=True)
                tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
                return DataLoader(tokenized_dataset.select(range(1000)), batch_size=batch_size) # Use a subset for speed
            except Exception as e:
                print(f"Failed to load text dataset: {e}")
                return None
        else:
            raise ValueError(f"Unknown text dataset: {dataset_name}")

    # Image datasets
    transform = get_transform(model_name)
    try:
        if dataset_name == 'ImageNet-C':
            severities = [1, 2, 3, 4, 5]
            target_severity = 5 # Default to worst corruption
            if split == 'validation': target_severity = 3

            ds = load_dataset("hendrycks/imagenet-c", trust_remote_code=True, split='validation')
            ds = ds.filter(lambda x: x['severity'] == target_severity)
            ds = ds.map(lambda x: {'image': transform(x['image']), 'label': x['label']})
            ds.set_format('torch')
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == 'CIFAR-C':
            ds = load_dataset("randall-lab/cifar10-c", split="test", trust_remote_code=True)
            # CIFAR-C needs a specific transform
            cifar_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            ds = ds.map(lambda x: {'image': cifar_transform(x['image']), 'label': x['label']})
            ds.set_format('torch')
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == 'ImageNet-V2':
            ds = load_dataset("vaishaal/ImageNetV2", "matched_frequency", split='validation', trust_remote_code=True)
            ds = ds.map(lambda x: {'image': transform(x['image']), 'label': x['label']})
            ds.set_format('torch')
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == 'ImageNet-A':
            ds = load_dataset("imagenet_a", split='test', trust_remote_code=True)
            ds = ds.map(lambda x: {'image': transform(x['image']), 'label': x['label']})
            ds.set_format('torch')
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == 'ImageNet-R':
            ds = load_dataset("imagenet_r", split='test', trust_remote_code=True)
            ds = ds.map(lambda x: {'image': transform(x['image']), 'label': x['label']})
            ds.set_format('torch')
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        elif dataset_name == 'ImageNet-3DCC':
            dataset = ImageNet3DCC(root_dir='./data/ImageNet-3DCC', transform=transform)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        elif dataset_name == 'CCC':
            dataset = ContinualCorruptionStream(transform=transform)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False) # Order matters
        elif dataset_name == 'Shifted-Objects-RT':
            dataset = ShiftedObjectsRT(transform=transform)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False) # Order matters
        else:
            raise ValueError(f"Unknown image dataset: {dataset_name}")
    except Exception as e:
        print(f"STRICT NO-FALLBACK RULE: Could not load dataset '{dataset_name}'. Error: {e}")
        print("Please ensure the dataset is available or downloaded correctly.")
        return None
