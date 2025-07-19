import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import re




class CaptchaDataset(Dataset):
    
    def __init__(self, images, labels, texts, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.texts = texts
        self.transform = transform
        self.augment = augment
        

        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        img_array = self.images[idx]
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        

        if self.transform:
            img = self.transform(img)
        
        # Apply augmentation if enabled
        if self.augment:
            img = self.augment_transform(img)
        
        # Convert to tensor and normalize
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label


class CaptchaImageLoader:
    
    
    def __init__(self, data_dir="captcha_data/output", img_size=(200, 80), 
                 chars="0123456789abcdefghijklmnopqrstuvwxyz"):
        """
        Initialize the captcha image loader.
        
        Args:
            data_dir: Path to the output directory containing train/test folders
            img_size: Tuple of (width, height) for resizing images
            chars: String of valid characters for the captcha
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.chars = chars
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(chars)
        self.max_length = 8  # Maximum captcha length
        
        # Validate directories exist
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
    
    def extract_label_from_filename(self, filename):
        """
        Extract the text label from filename.
        Expected format: {label}_{text}_{index}.jpg
        """
        name = Path(filename).stem
        
        parts = name.split('_')
        
        
        if len(parts) >= 2:
            return parts[1]
        else:
            text_match = re.search(r'[a-z0-9]+', name)
            if text_match:
                return text_match.group()
            else:
                return ""
    
    def preprocess_image(self, image_path):
        
        img = Image.open(image_path).convert('RGB')
        
        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
    
    def text_to_sequence(self, text):
        
        sequence = []
        for char in text:
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(0)  
        
        if len(sequence) < self.max_length:
            sequence.extend([0] * (self.max_length - len(sequence)))
        else:
            sequence = sequence[:self.max_length]
        
        return sequence
    
    def sequence_to_text(self, sequence):
        
        text = ""
        for idx in sequence:
            if idx in self.idx_to_char:
                text += self.idx_to_char[idx]
        return text
    
    def load_dataset(self, split='train'):
        
        if split == 'train':
            data_dir = self.train_dir
        elif split == 'test':
            data_dir = self.test_dir
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        images = []
        labels = []
        texts = []
        
        image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        
        print(f"Loading {len(image_files)} images from {split} split...")
        
        for img_path in image_files:
            try:

                text = self.extract_label_from_filename(img_path.name)
                
                if text:  
                    
                    img_array = self.preprocess_image(img_path)
                    
                    label_sequence = self.text_to_sequence(text)
                    
                    images.append(img_array)
                    labels.append(label_sequence)
                    texts.append(text)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images with shape {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample texts: {texts[:5]}")
        
        return images, labels, texts
    
    def create_data_loader(self, split='train', batch_size=32, augment=True, shuffle=True, num_workers=0):
        
        images, labels, texts = self.load_dataset(split)
        
        # Create dataset
        dataset = CaptchaDataset(images, labels, texts, augment=augment)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return data_loader
    
    def visualize_samples(self, num_samples=5, split='train'):
        
        images, labels, texts = self.load_dataset(split)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            if i < len(images):
                axes[i].imshow(images[i])
                axes[i].set_title(f"Text: {texts[i]}")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_info(self):
        """
        Print information about the dataset.
        """
        print("=== Dataset Information ===")
        
        for split in ['train', 'test']:
            try:
                images, labels, texts = self.load_dataset(split)
                print(f"\n{split.upper()} Split:")
                print(f"  Number of images: {len(images)}")
                print(f"  Image shape: {images.shape}")
                print(f"  Label shape: {labels.shape}")
                print(f"  Unique texts: {len(set(texts))}")
                print(f"  Text length range: {min(len(t) for t in texts)}-{max(len(t) for t in texts)}")
                print(f"  Sample texts: {texts[:3]}")
            except Exception as e:
                print(f"Error loading {split} split: {e}")


def main():
    """
    Example usage of the CaptchaImageLoader.
    """
    # Initialize the loader
    loader = CaptchaImageLoader()
    
    # Get dataset information
    loader.get_dataset_info()
    
    # Visualize some samples
    loader.visualize_samples(num_samples=5, split='train')
    
    # Create data loaders
    train_loader = loader.create_data_loader(split='train', batch_size=32, augment=True, shuffle=True)
    test_loader = loader.create_data_loader(split='test', batch_size=32, augment=False, shuffle=False)
    
    print("\n=== Data Loaders Created ===")
    print(f"Train loader: {train_loader}")
    print(f"Test loader: {test_loader}")
    
    # Test the loaders
    for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx} shapes:")
        print(f"  Images: {batch_images.shape}")
        print(f"  Labels: {batch_labels.shape}")
        print(f"  Image dtype: {batch_images.dtype}")
        print(f"  Label dtype: {batch_labels.dtype}")
        break


if __name__ == "__main__":
    main()
