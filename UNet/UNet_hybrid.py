#!/usr/bin/env python3
"""
Hybrid Attention Mechanism-Based U-Net for OCT Image Segmentation

This implementation follows the research paper for segmenting three specific 
sub-retinal layers (ILM, PR1, BM) in Optical Coherence Tomography (OCT) images.

Author: Implementation based on the hybrid U-Net research paper
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations import Compose
import uuid

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class EdgeAttentionBlock(nn.Module):
    """Edge Attention Block for early layers focusing on edge information."""
    
    def __init__(self, in_channels):
        super(EdgeAttentionBlock, self).__init__()
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.attention_conv = nn.Conv2d(in_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Canny-like operation
        edge_features = self.relu(self.edge_conv(x))
        
        # Generate attention weights
        attention_weights = self.sigmoid(self.attention_conv(edge_features))
        
        # Apply attention
        attended_features = x * attention_weights
        
        return attended_features


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention Block for deeper layers focusing on spatial features."""
    
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.attention_conv = nn.Conv2d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global max and average pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # Global max along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)     # Global average along channel dimension
        
        # Concatenate pooling results
        concat_pool = torch.cat([max_pool, avg_pool], dim=1)
        
        # Generate attention weights
        attention_weights = self.sigmoid(self.attention_conv(concat_pool))
        
        # Apply attention
        attended_features = x * attention_weights
        
        return attended_features


class EncoderBlock(nn.Module):
    """Encoder block with optional attention mechanism."""
    
    def __init__(self, in_channels, out_channels, use_edge_attention=False):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.use_edge_attention = use_edge_attention
        
        if use_edge_attention:
            self.attention = EdgeAttentionBlock(out_channels)
        else:
            self.attention = SpatialAttentionBlock()
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Convolution
        conv = self.relu(self.conv(x))
        
        # Apply attention
        conv = self.attention(conv)
        
        # Max pooling
        pooled = self.pool(conv)
        
        return pooled, conv


class DecoderBlock(nn.Module):
    """Decoder block with skip connections."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # After concatenation: out_channels + skip_channels
        self.conv = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip_features):
        # Transpose convolution
        up = self.upconv(x)
        
        # Skip connection
        concat = torch.cat([up, skip_features], dim=1)
        
        # Convolution
        conv = self.relu(self.conv(concat))
        
        return conv


class HybridUNet(nn.Module):
    """
    Hybrid U-Net model with Edge and Spatial Attention mechanisms
    for OCT image segmentation of sub-retinal layers.
    """
    
    def __init__(self, input_channels=1, num_classes=4):
        super(HybridUNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Encoder blocks
        self.enc1 = EncoderBlock(input_channels, 64, use_edge_attention=True)
        self.enc2 = EncoderBlock(64, 128, use_edge_attention=True)
        self.enc3 = EncoderBlock(128, 256, use_edge_attention=False)
        self.enc4 = EncoderBlock(256, 512, use_edge_attention=False)
        self.enc5 = EncoderBlock(512, 1024, use_edge_attention=False)
        
        # Base layer
        self.base_conv = nn.Conv2d(1024, 1024, 3, padding=1)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_attention = SpatialAttentionBlock()
        
        # Decoder blocks
        self.dec4 = DecoderBlock(1024, 1024, 512)  # in_channels=1024, skip_channels=1024, out_channels=512
        self.dec3 = DecoderBlock(512, 512, 256)    # in_channels=512, skip_channels=512, out_channels=256
        self.dec2 = DecoderBlock(256, 256, 128)    # in_channels=256, skip_channels=256, out_channels=128
        self.dec1 = DecoderBlock(128, 128, 64)     # in_channels=128, skip_channels=128, out_channels=64
        
        # Final layers
        self.final_upconv = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.final_conv = nn.Conv2d(128, 64, 3, padding=1)  # 64 + 64 from skip1
        self.final_relu = nn.ReLU(inplace=True)
        
        # Output layer
        self.output_conv = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        # Encoder path
        enc1, skip1 = self.enc1(x)
        enc2, skip2 = self.enc2(enc1)
        enc3, skip3 = self.enc3(enc2)
        enc4, skip4 = self.enc4(enc3)
        enc5, skip5 = self.enc5(enc4)
        
        # Base layer
        base = self.base_relu(self.base_conv(enc5))
        base = self.base_attention(base)
        
        # Decoder path
        dec4 = self.dec4(base, skip5)
        dec3 = self.dec3(dec4, skip4)
        dec2 = self.dec2(dec3, skip3)
        dec1 = self.dec1(dec2, skip2)
        
        # Final decoder
        final = self.final_upconv(dec1)
        final = torch.cat([final, skip1], dim=1)
        final = self.final_relu(self.final_conv(final))
        
        # Output
        output = self.output_conv(final)
        
        return output


class OCTDataset(Dataset):
    """Dataset class for OCT images and masks."""
    
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            # Convert to PIL or numpy format for albumentations if needed
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to torch tensors
        if len(image.shape) == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice coefficient.
    
    Args:
        y_pred: Predicted tensor
        y_true: Ground truth tensor
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_metric(y_pred, y_true, smooth=1e-6):
    """
    Calculate IoU metric.
    
    Args:
        y_pred: Predicted tensor
        y_true: Ground truth tensor
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def load_dataset(file_path):
    """
    Load dataset from HDF5 file.
    
    Args:
        file_path: Path to the .h5 file
        
    Returns:
        Tuple of (images, annotations)
    """
    print(f"Loading dataset from {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Load images
        images = np.array(f['images'])
        print(f"Images shape: {images.shape}")
        
        # Load annotations for specific layers
        layers_group = f['layers']
        annotations = {}
        target_layers = ['ILM', 'PR1', 'BM']
        
        for layer in target_layers:
            if layer in layers_group:
                annotations[layer] = np.array(layers_group[layer])
                print(f"{layer} annotations shape: {annotations[layer].shape}")
            else:
                print(f"Warning: {layer} not found in dataset")
    
    return images, annotations


def create_augmentation_pipeline():
    """
    Create augmentation pipeline using Albumentations.
    
    Returns:
        Albumentations Compose object
    """
    return Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=1.0),
        A.GaussianBlur(blur_limit=(1, 3), p=0.5),
        A.InvertImg(p=0.1),
        A.Equalize(p=0.1)
    ])


def preprocess_data(images, annotations, target_size=(512, 256)):
    """
    Preprocess images and annotations.
    
    Args:
        images: Input images array
        annotations: Dictionary of annotations
        target_size: Target image size (height, width)
        
    Returns:
        Tuple of (preprocessed_images, preprocessed_masks)
    """
    print("Preprocessing data...")
    
    batch_size, original_height, original_width = images.shape
    target_height, target_width = target_size
    
    # Initialize preprocessed arrays
    preprocessed_images = np.zeros((batch_size, target_height, target_width, 1))
    preprocessed_masks = np.zeros((batch_size, target_height, target_width))
    
    # Create augmentation pipeline
    augment = create_augmentation_pipeline()
    
    # Scaling factor for annotations
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    
    for i in range(batch_size):
        # Convert to uint8 and normalize to 0-255 range
        img = images[i].astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Convert grayscale to RGB for albumentations compatibility
        img_rgb = np.stack([img, img, img], axis=-1)
        
        # Resize image using albumentations
        augmented = A.Resize(target_height, target_width, interpolation=1)(image=img_rgb)
        img_resized = augmented['image']
        
        # Apply augmentations
        if np.random.random() > 0.5:  # Apply augmentation to 50% of images
            augmented = augment(image=img_resized)
            img_resized = augmented['image']
        
        # Convert back to grayscale and normalize to 0-1
        img_grayscale = np.mean(img_resized, axis=2)
        img_grayscale = img_grayscale.astype(np.float32) / 255.0
        preprocessed_images[i, :, :, 0] = img_grayscale
        
        # Create combined mask for three layers
        mask = np.zeros((target_height, target_width))
        
        # Process each layer annotation
        layer_labels = {'ILM': 1, 'PR1': 2, 'BM': 3}
        
        for layer_name, label in layer_labels.items():
            if layer_name in annotations:
                layer_coords = annotations[layer_name][i]
                
                # Scale and create mask
                for x in range(len(layer_coords)):
                    y_coord_raw = layer_coords[x]
                    
                    # Skip NaN values
                    if np.isnan(y_coord_raw):
                        continue
                        
                    y_coord = int(y_coord_raw * height_scale)
                    x_coord = int(x * width_scale)
                    
                    if 0 <= y_coord < target_height and 0 <= x_coord < target_width:
                        mask[y_coord, x_coord] = label
        
        preprocessed_masks[i] = mask
    
    print(f"Preprocessed images shape: {preprocessed_images.shape}")
    print(f"Preprocessed masks shape: {preprocessed_masks.shape}")
    
    return preprocessed_images, preprocessed_masks


def test_preprocessing_visualization(images, annotations, preprocessed_images, preprocessed_masks, save_path="preprocessing_test.png"):
    """
    Visualize preprocessing results for testing.
    
    Args:
        images: Original images
        annotations: Original annotations
        preprocessed_images: Preprocessed images
        preprocessed_masks: Preprocessed masks
        save_path: Path to save the visualization
    """
    print("Creating preprocessing test visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image with annotations
    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Original Image with Annotations')
    
    # Plot original annotations
    colors = ['red', 'blue', 'green']
    layer_names = ['ILM', 'PR1', 'BM']
    
    for i, (layer_name, color) in enumerate(zip(layer_names, colors)):
        if layer_name in annotations:
            y_coords = annotations[layer_name][0]
            x_coords = range(len(y_coords))
            
            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(y_coords)
            valid_x = [x for x, valid in zip(x_coords, valid_indices) if valid]
            valid_y = y_coords[valid_indices]
            
            if len(valid_x) > 0:
                axes[0].plot(valid_x, valid_y, color=color, label=layer_name, linewidth=2)
    
    axes[0].legend()
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    
    # Preprocessed image with mask overlay
    axes[1].imshow(preprocessed_images[0, :, :, 0], cmap='gray')
    axes[1].imshow(preprocessed_masks[0], alpha=0.3, cmap='jet')
    axes[1].set_title('Preprocessed Image with Mask Overlay')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Preprocessing test visualization saved as {save_path}")


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    
    # Calculate Dice coefficient
    intersection = np.sum(y_true_flat == y_pred_flat)
    dice = (2.0 * intersection + 1.0) / (len(y_true_flat) + len(y_pred_flat) + 1.0)
    
    # Calculate IoU
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice': dice,
        'iou': iou
    }


def plot_training_metrics(history, save_path="training_metrics.png"):
    """
    Plot training metrics.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    print("Plotting training metrics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Dice and IoU
    axes[0, 1].plot(epochs, history['train_dice'], 'g-', label='Training Dice')
    axes[0, 1].plot(epochs, history['val_dice'], 'orange', label='Validation Dice')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: IoU
    axes[1, 0].plot(epochs, history['train_iou'], 'm-', label='Training IoU')
    axes[1, 0].plot(epochs, history['val_iou'], 'c-', label='Validation IoU')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Combined metrics
    axes[1, 1].plot(epochs, history['val_dice'], 'g-', label='Val Dice')
    axes[1, 1].plot(epochs, history['val_iou'], 'orange', label='Val IoU')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training metrics plot saved as {save_path}")


def train_model(model, train_loader, val_loader, epochs=300, learning_rate=0.001, device='cuda'):
    """
    Train the Hybrid U-Net model using PyTorch.
    
    Args:
        model: HybridUNet model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Training history dictionary
    """
    print("Starting model training...")
    
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coefficient(output, target).item()
            train_iou += iou_metric(output, target).item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_dice += dice_coefficient(output, target).item()
                val_iou += iou_metric(output, target).item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_train_dice = train_dice / train_batches
        avg_val_dice = val_dice / val_batches
        avg_train_iou = train_iou / train_batches
        avg_val_iou = val_iou / val_batches
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'  Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}')
            print(f'  Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model_weights.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return history


def main():
    """
    Main function to execute the complete pipeline.
    """
    print("=== Hybrid U-Net for OCT Image Segmentation ===")
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configuration
    dataset_path = "/home/suraj/Git/SCR-Progression/Nemours_Jing_RL_Annotated.h5"
    batch_size = 4
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Step 1: Load dataset
    try:
        images, annotations = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Step 2: Preprocess data
    try:
        preprocessed_images, preprocessed_masks = preprocess_data(images, annotations)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return
    
    # Step 3: Test preprocessing visualization
    try:
        test_preprocessing_visualization(images, annotations, preprocessed_images, preprocessed_masks)
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_images, preprocessed_masks,
        test_size=0.2,
        random_state=42,
        stratify=None
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Create data loaders
    train_dataset = OCTDataset(X_train.squeeze(-1), y_train)  # Remove channel dimension for dataset
    val_dataset = OCTDataset(X_test.squeeze(-1), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Step 6: Initialize model
    model = HybridUNet(input_channels=1, num_classes=4)  # Background + 3 layers
    
    print("Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 7: Train model
    try:
        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=30,
            learning_rate=0.001,
            device=device
        )
        
        # Step 8: Plot training metrics
        plot_training_metrics(history)
        
        # Step 9: Evaluate model
        print("\nEvaluating model on test set...")
        model.eval()
        test_loss = 0.0
        test_dice = 0.0
        test_iou = 0.0
        test_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                test_dice += dice_coefficient(output, target).item()
                test_iou += iou_metric(output, target).item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        avg_test_dice = test_dice / test_batches
        avg_test_iou = test_iou / test_batches
        
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Dice: {avg_test_dice:.4f}")
        print(f"Test IoU: {avg_test_iou:.4f}")
        
        # Make predictions for additional metrics
        all_predictions = []
        all_targets = []
        
        
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                output = model(data)
                pred = torch.softmax(output, dim=1)
                pred = torch.argmax(pred, dim=1)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.numpy().flatten())
        
        # Calculate additional metrics
        metrics = calculate_metrics(np.array(all_targets), np.array(all_predictions))
        print(f"\nAdditional Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    # Check PyTorch and GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    main()
