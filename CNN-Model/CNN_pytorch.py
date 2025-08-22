''' baseline CNN'''

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from skimage.transform import resize

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class LayerAnnotationDataset(Dataset):
    def __init__(self, images, layer_maps):
        self.images = images.astype(np.float32)
        self.layer_maps = layer_maps.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        layers = self.layer_maps[idx]
        # Transpose image to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(layers)

# Model definition
class LayerAnnotationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 224 * 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        x = x.view(-1, 224, 2)
        return x

def denormalize_layers(layers, layer_min=0, layer_max=224):
    """Denormalize layer coordinates from [0,1] to pixel coordinates"""
    return layers * (layer_max - layer_min) + layer_min

def combine_datasets(datasets_info):
    """
    Combine multiple datasets into a single training dataset.
    
    Args:
        datasets_info (list): List of tuples (images, layer_maps, dataset_name)
    
    Returns:
        tuple: (combined_images, combined_layer_maps)
    """
    all_images = []
    all_layer_maps = []
    
    for images, layer_maps, name in datasets_info:
        print(f"Adding {name} dataset: {len(images)} samples")
        all_images.append(images)
        all_layer_maps.append(layer_maps)
    
    combined_images = np.concatenate(all_images, axis=0)
    combined_layer_maps = np.concatenate(all_layer_maps, axis=0)
    
    print(f"Combined dataset: {len(combined_images)} total samples")
    return combined_images, combined_layer_maps

def plot_sample_predictions(model, images, layer_maps, num_samples=3, save_dir=None):
    """Plot 3 sample images: original, with true annotations, with predicted annotations"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    
    for idx in range(min(num_samples, len(images))):
        img = images[idx]
        true_layers = denormalize_layers(layer_maps[idx], 0, 224)
        
        with torch.no_grad():
            inp = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
            pred_layers = model(inp).cpu().numpy()[0]
            pred_layers = denormalize_layers(pred_layers, 0, 224)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img[:, :, 0], cmap='gray')
        axes[0].set_title(f'Sample {idx}: Original Image')
        axes[0].axis('off')
        
        # Image with true annotations
        axes[1].imshow(img[:, :, 0], cmap='gray')
        axes[1].plot(range(224), true_layers[:, 0], 'g-', label='True ILM', linewidth=2)
        axes[1].plot(range(224), true_layers[:, 1], 'b-', label='True BM', linewidth=2)
        axes[1].set_title(f'Sample {idx}: True Annotations')
        axes[1].legend()
        axes[1].axis('off')
        
        # Image with predicted annotations
        axes[2].imshow(img[:, :, 0], cmap='gray')
        axes[2].plot(range(224), pred_layers[:, 0], 'r--', label='Pred ILM', linewidth=2)
        axes[2].plot(range(224), pred_layers[:, 1], 'm--', label='Pred BM', linewidth=2)
        axes[2].set_title(f'Sample {idx}: Predicted Annotations')
        axes[2].legend()
        axes[2].axis('off')
        
        plt.tight_layout()
        if save_dir:
            filename = f"sample_predictions_{idx}.png"
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
        plt.close()

def mae_metric(pred, target)-> float:
    '''
    Mean Absolute Error
    args: 
        pred (Tensor)
        target (Tensor)
    returns:
        Mean Absolute Error as a float
    '''
    return torch.mean(torch.abs(pred - target)).item()

def lines_to_mask(lines, height=224, width=224, thickness=3):
    """
    Convert line coordinates to binary masks with thickness for better overlap.
    
    Args:
        lines: (batch, width, 2) - for each x, y1 and y2
        height: image height
        width: image width  
        thickness: thickness of the mask lines (default 3 for better overlap)
    
    Returns: 
        (batch, 2, height, width) binary masks for ILM and BM
    """
    batch = lines.shape[0]
    mask = torch.zeros((batch, 2, height, width), device=lines.device)
    
    for b in range(batch):
        for i in range(2):  # ILM and BM
            y_coords = torch.clamp(lines[b, :, i].round().long(), 0, height-1)
            
            # Create thick lines by adding neighboring pixels
            for offset in range(-thickness//2, thickness//2 + 1):
                y_thick = torch.clamp(y_coords + offset, 0, height-1)
                mask[b, i, y_thick, torch.arange(width)] = 1
    
    return mask

def dice_coefficient(pred_mask, target_mask, eps=1e-6):
    # pred_mask, target_mask: (batch, 2, H, W)
    intersection = (pred_mask * target_mask).sum(dim=(2,3))
    union = pred_mask.sum(dim=(2,3)) + target_mask.sum(dim=(2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_metric(pred_mask, target_mask, eps=1e-6):
    # pred_mask, target_mask: (batch, 2, H, W)
    intersection = (pred_mask * target_mask).sum(dim=(2,3))
    union = pred_mask.sum(dim=(2,3)) + target_mask.sum(dim=(2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def precision_recall_f1(pred_mask, target_mask, eps=1e-6):
    # pred_mask, target_mask: (batch, 2, H, W)
    tp = (pred_mask * target_mask).sum(dim=(2,3))
    fp = (pred_mask * (1 - target_mask)).sum(dim=(2,3))
    fn = ((1 - pred_mask) * target_mask).sum(dim=(2,3))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return precision.mean().item(), recall.mean().item(), f1.mean().item()

def coordinate_mae(pred_coords, target_coords):
    """
    Calculate Mean Absolute Error for coordinate predictions (better for line-based tasks).
    
    Args:
        pred_coords: (batch, width, 2) predicted coordinates
        target_coords: (batch, width, 2) target coordinates
    
    Returns:
        float: Mean absolute error in pixels
    """
    return torch.mean(torch.abs(pred_coords - target_coords)).item()

def coordinate_rmse(pred_coords, target_coords):
    """
    Calculate Root Mean Square Error for coordinate predictions.
    
    Args:
        pred_coords: (batch, width, 2) predicted coordinates  
        target_coords: (batch, width, 2) target coordinates
    
    Returns:
        float: Root mean square error in pixels
    """
    return torch.sqrt(torch.mean((pred_coords - target_coords) ** 2)).item()

def boundary_distance_metric(pred_coords, target_coords, threshold=5.0):
    """
    Calculate percentage of predictions within threshold distance of ground truth.
    
    Args:
        pred_coords: (batch, width, 2) predicted coordinates
        target_coords: (batch, width, 2) target coordinates  
        threshold: maximum distance in pixels to consider as correct
    
    Returns:
        float: Percentage of predictions within threshold (0-1)
    """
    distances = torch.abs(pred_coords - target_coords)
    within_threshold = (distances <= threshold).float()
    return within_threshold.mean().item()

def check_for_nan_gradients(model):
    """Check if gradients contain NaN values"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient found in {name}")
                return True
    return False

def plot_training_metrics_comprehensive(train_losses, val_losses, val_f1s, val_precisions, val_recalls, 
                                       val_dice, val_iou, val_coord_maes, val_coord_rmses, val_boundary_accuracies, 
                                       n_epochs, save_dir=None):
    """Plot comprehensive training metrics including coordinate-based metrics"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot 1: Loss vs Epoch
    axes[0, 0].plot(range(1, n_epochs+1), train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(range(1, n_epochs+1), val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Coordinate-based Metrics (Better for line prediction)
    axes[0, 1].plot(range(1, n_epochs+1), val_coord_maes, label='MAE (pixels)', color='green')
    axes[0, 1].plot(range(1, n_epochs+1), val_coord_rmses, label='RMSE (pixels)', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error (pixels)')
    axes[0, 1].set_title('Coordinate Prediction Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Boundary Accuracy (% within 5 pixels)
    axes[1, 0].plot(range(1, n_epochs+1), val_boundary_accuracies, label='Boundary Acc (5px)', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Boundary Accuracy (within 5 pixels)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: Dice and IoU vs Epoch (with thick masks)
    axes[1, 1].plot(range(1, n_epochs+1), val_dice, label='Dice Score', color='green')
    axes[1, 1].plot(range(1, n_epochs+1), val_iou, label='IoU Score', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Dice and IoU Scores (Thick Masks)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: F1 Score vs Epoch
    axes[2, 0].plot(range(1, n_epochs+1), val_f1s, label='F1 Score', color='purple')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('F1 Score')
    axes[2, 0].set_title('F1 Score (Thick Masks)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Precision and Recall vs Epoch
    axes[2, 1].plot(range(1, n_epochs+1), val_precisions, label='Precision', color='cyan')
    axes[2, 1].plot(range(1, n_epochs+1), val_recalls, label='Recall', color='magenta')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Score')
    axes[2, 1].set_title('Precision and Recall (Thick Masks)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        filename = "comprehensive_training_metrics.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
    plt.close()

def load_nemours_data(hdf5_path, target_size=(224, 224), normalize=True):
    """
    Load data from Nemours_Jing_RL_Annotated.h5 file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        target_size (tuple): Target size for images (height, width)
        normalize (bool): Whether to normalize layer annotations to [0, 1]
    
    Returns:
        tuple: (images, layer_maps) where images have shape (N, H, W, 1) 
               and layer_maps have shape (N, W, 2) for ILM and BM layers
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load images: shape (310, 496, 768)
        images = f['images'][:]
        
        # Load layer annotations - only ILM and BM
        layers = f['layers']
        ilm_annotations = layers['ILM'][:]  # shape (310, 768)
        bm_annotations = layers['BM'][:]    # shape (310, 768)
        
        # Resize images and scale layer annotations
        resized_images = []
        resized_layer_maps = []
        
        original_height, original_width = images.shape[1], images.shape[2]
        target_height, target_width = target_size
        
        # Calculate scaling factors
        height_scale = target_height / original_height
        width_scale = target_width / original_width
        
        for i in range(images.shape[0]):
            # Resize image
            img = images[i]
            resized_img = resize(img, target_size, preserve_range=True, anti_aliasing=True)
            
            # Normalize image to [0, 1] range
            if resized_img.max() > 1.0:
                resized_img = resized_img / 255.0
            
            resized_images.append(resized_img)
            
            # Scale layer annotations
            # Create new x-coordinates for target width
            new_x = np.linspace(0, original_width - 1, target_width)
            original_x = np.arange(original_width)
            
            # Interpolate ILM and BM y-coordinates for new x-coordinates
            ilm_y_resized = np.interp(new_x, original_x, ilm_annotations[i])
            bm_y_resized = np.interp(new_x, original_x, bm_annotations[i])
            
            # Scale y-coordinates to new height
            ilm_y_resized = ilm_y_resized * height_scale
            bm_y_resized = bm_y_resized * height_scale
            
            # Normalize layer annotations to [0, 1] if requested
            if normalize:
                # Check for invalid values before normalization
                if target_height == 0:
                    print(f"Warning: target_height is 0, skipping normalization")
                else:
                    ilm_y_resized = ilm_y_resized / target_height
                    bm_y_resized = bm_y_resized / target_height
                    
                    # Check for NaN or infinite values after normalization
                    if np.isnan(ilm_y_resized).any() or np.isnan(bm_y_resized).any():
                        print(f"Warning: NaN values generated during normalization for sample {i}")
                        print(f"  ILM range before norm: [{ilm_y_resized.min():.4f}, {ilm_y_resized.max():.4f}]")
                        print(f"  BM range before norm: [{bm_y_resized.min():.4f}, {bm_y_resized.max():.4f}]")
                    
                    if np.isinf(ilm_y_resized).any() or np.isinf(bm_y_resized).any():
                        print(f"Warning: Infinite values generated during normalization for sample {i}")
            
            # Replace any NaN or infinite values with valid defaults
            ilm_y_resized = np.nan_to_num(ilm_y_resized, nan=0.5, posinf=1.0, neginf=0.0)
            bm_y_resized = np.nan_to_num(bm_y_resized, nan=0.7, posinf=1.0, neginf=0.0)
            
            # Combine ILM and BM into layer_map format (W, 2)
            layer_map = np.column_stack([ilm_y_resized, bm_y_resized])
            resized_layer_maps.append(layer_map)
        
        # Convert to numpy arrays
        images = np.array(resized_images)
        layer_maps = np.array(resized_layer_maps)
        
        # Add channel dimension to images if needed
        if images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
        
        return images, layer_maps

def load_duke_data(hdf5_path):
    """
    Load data from Duke dataset HDF5 file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
    
    Returns:
        tuple: (images, layer_maps) in normalized format
    """
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]  # (N, 224, 224)
        layer_maps = f['layer_maps'][:]  # (N, 224, 3)
        layer_names = [name.decode() for name in f['layer_names'][:]]
    
    # Find indices for ILM and BM
    ilm_idx = layer_names.index('ILM') if 'ILM' in layer_names else 0
    bm_idx = next((i for i, name in enumerate(layer_names) if 'Bruch' in name or 'BM' in name), 2)
    
    # Normalize images to [0, 1] if needed
    if images.max() > 1.0:
        images = images / 255.0
    
    # Extract ILM and BM layers only
    layer_maps = layer_maps[:, :, [ilm_idx, bm_idx]]
    
    # Check for and handle NaN values
    if np.isnan(layer_maps).any():
        print(f"Warning: NaN values found in Duke layer maps")
        nan_samples = np.isnan(layer_maps).any(axis=(1, 2))
        print(f"Samples with NaN: {np.sum(nan_samples)}")
        layer_maps = np.nan_to_num(layer_maps, nan=0.5)
    
    # Add channel dimension to images if needed
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    
    return images, layer_maps

def test_lines_to_mask_visualization(model, images, layer_maps, num_samples=3, save_dir="mask_test"):
    """Test the lines_to_mask function by visualizing masks overlaid on images."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(images))):
            img = images[idx]
            true_layers_norm = layer_maps[idx]  # Normalized [0,1]
            true_layers = denormalize_layers(true_layers_norm, 0, 224)  # Denormalized for visualization
            
            # Get model prediction (normalized output)
            inp = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
            pred_layers_norm = model(inp).cpu().numpy()[0]  # Normalized output
            pred_layers = denormalize_layers(pred_layers_norm, 0, 224)  # Denormalized for visualization
            
            # Convert to torch tensors for mask generation (use denormalized for mask generation)
            true_layers_torch = torch.from_numpy(true_layers).unsqueeze(0)
            pred_layers_torch = torch.from_numpy(pred_layers).unsqueeze(0)
            
            # Generate masks with thickness=1 and thickness=3 for comparison
            true_mask_thin = lines_to_mask(true_layers_torch, thickness=1)
            pred_mask_thin = lines_to_mask(pred_layers_torch, thickness=1)
            true_mask_thick = lines_to_mask(true_layers_torch, thickness=3)
            pred_mask_thick = lines_to_mask(pred_layers_torch, thickness=3)
            
            # Calculate metrics for both thin and thick masks
            dice_thin = dice_coefficient(pred_mask_thin, true_mask_thin)
            dice_thick = dice_coefficient(pred_mask_thick, true_mask_thick)
            iou_thin = iou_metric(pred_mask_thin, true_mask_thin)
            iou_thick = iou_metric(pred_mask_thick, true_mask_thick)
            
            # Create visualization
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            
            # Row 1: Original and line annotations
            axes[0, 0].imshow(img[:, :, 0], cmap='gray')
            axes[0, 0].set_title(f'Sample {idx}: Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(img[:, :, 0], cmap='gray')
            axes[0, 1].plot(range(224), true_layers[:, 0], 'g-', label='True ILM', linewidth=2)
            axes[0, 1].plot(range(224), true_layers[:, 1], 'b-', label='True BM', linewidth=2)
            axes[0, 1].plot(range(224), pred_layers[:, 0], 'r--', label='Pred ILM', linewidth=2)
            axes[0, 1].plot(range(224), pred_layers[:, 1], 'm--', label='Pred BM', linewidth=2)
            axes[0, 1].set_title('Line Annotations Comparison')
            axes[0, 1].legend()
            axes[0, 1].axis('off')
            
            # Coordinate accuracy analysis
            coord_mae = np.mean(np.abs(pred_layers - true_layers))
            coord_rmse = np.sqrt(np.mean((pred_layers - true_layers) ** 2))
            axes[0, 2].text(0.1, 0.8, f'Coordinate Metrics:', fontsize=12, fontweight='bold')
            axes[0, 2].text(0.1, 0.7, f'MAE: {coord_mae:.2f} pixels', fontsize=10)
            axes[0, 2].text(0.1, 0.6, f'RMSE: {coord_rmse:.2f} pixels', fontsize=10)
            axes[0, 2].text(0.1, 0.4, f'Mask Metrics (thin):', fontsize=12, fontweight='bold')
            axes[0, 2].text(0.1, 0.3, f'Dice: {dice_thin:.4f}', fontsize=10)
            axes[0, 2].text(0.1, 0.2, f'IoU: {iou_thin:.4f}', fontsize=10)
            axes[0, 2].text(0.1, 0.05, f'Mask Metrics (thick):', fontsize=12, fontweight='bold')
            axes[0, 2].text(0.1, -0.05, f'Dice: {dice_thick:.4f}', fontsize=10)
            axes[0, 2].text(0.1, -0.15, f'IoU: {iou_thick:.4f}', fontsize=10)
            axes[0, 2].set_xlim(0, 1)
            axes[0, 2].set_ylim(-0.2, 1)
            axes[0, 2].set_title('Metrics Summary')
            axes[0, 2].axis('off')
            
            # Row 2: Thin masks
            axes[1, 0].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[1, 0].imshow(true_mask_thin[0, 0].numpy(), cmap='Reds', alpha=0.6)
            axes[1, 0].imshow(true_mask_thin[0, 1].numpy(), cmap='Blues', alpha=0.6)
            axes[1, 0].set_title('True Masks (Thin, 1px)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[1, 1].imshow(pred_mask_thin[0, 0].numpy(), cmap='Reds', alpha=0.6)
            axes[1, 1].imshow(pred_mask_thin[0, 1].numpy(), cmap='Blues', alpha=0.6)
            axes[1, 1].set_title('Pred Masks (Thin, 1px)')
            axes[1, 1].axis('off')
            
            # Overlap visualization for thin masks
            overlap_thin = true_mask_thin[0, 0].numpy() * pred_mask_thin[0, 0].numpy() + \
                          true_mask_thin[0, 1].numpy() * pred_mask_thin[0, 1].numpy()
            axes[1, 2].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[1, 2].imshow(overlap_thin, cmap='Greens', alpha=0.8)
            axes[1, 2].set_title('Overlap (Thin masks)')
            axes[1, 2].axis('off')
            
            # Row 3: Thick masks
            axes[2, 0].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[2, 0].imshow(true_mask_thick[0, 0].numpy(), cmap='Reds', alpha=0.6)
            axes[2, 0].imshow(true_mask_thick[0, 1].numpy(), cmap='Blues', alpha=0.6)
            axes[2, 0].set_title('True Masks (Thick, 3px)')
            axes[2, 0].axis('off')
            
            axes[2, 1].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[2, 1].imshow(pred_mask_thick[0, 0].numpy(), cmap='Reds', alpha=0.6)
            axes[2, 1].imshow(pred_mask_thick[0, 1].numpy(), cmap='Blues', alpha=0.6)
            axes[2, 1].set_title('Pred Masks (Thick, 3px)')
            axes[2, 1].axis('off')
            
            # Overlap visualization for thick masks
            overlap_thick = true_mask_thick[0, 0].numpy() * pred_mask_thick[0, 0].numpy() + \
                           true_mask_thick[0, 1].numpy() * pred_mask_thick[0, 1].numpy()
            axes[2, 2].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[2, 2].imshow(overlap_thick, cmap='Greens', alpha=0.8)
            axes[2, 2].set_title('Overlap (Thick masks)')
            axes[2, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'mask_test_sample_{idx}.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()

def validate_data(images, layer_maps):
    """Validate data integrity and fix common issues"""
    print("Validating data integrity...")
    
    # Check for NaN values
    images_nan = np.isnan(images).any()
    layers_nan = np.isnan(layer_maps).any()
    
    if images_nan:
        print("⚠️  NaN values found in images - replacing with zeros")
        images = np.nan_to_num(images, nan=0.0)
    
    if layers_nan:
        print("⚠️  NaN values found in layer maps - replacing with defaults")
        layer_maps = np.nan_to_num(layer_maps, nan=0.5)
    
    # Check for infinite values
    images_inf = np.isinf(images).any()
    layers_inf = np.isinf(layer_maps).any()
    
    if images_inf:
        print("⚠️  Infinite values found in images - clipping to valid range")
        images = np.clip(images, 0.0, 1.0)
    
    if layers_inf:
        print("⚠️  Infinite values found in layer maps - clipping to valid range")
        layer_maps = np.clip(layer_maps, 0.0, 1.0)
    
    # Check value ranges
    if images.min() < 0 or images.max() > 1:
        print(f"⚠️  Images out of expected range [0,1]: [{images.min():.3f}, {images.max():.3f}]")
        images = np.clip(images, 0.0, 1.0)
    
    if layer_maps.min() < 0 or layer_maps.max() > 1:
        print(f"⚠️  Layer maps out of expected range [0,1]: [{layer_maps.min():.3f}, {layer_maps.max():.3f}]")
        layer_maps = np.clip(layer_maps, 0.0, 1.0)
    
    print("✅ Data validation complete")
    return images, layer_maps

if __name__ == "__main__":
    # Configuration
    use_nemours_data = True
    use_duke_data = True
    max_samples_per_dataset = 700  # Set to number to limit samples, None for all

    # Load datasets
    datasets = []
    
    if use_nemours_data:
        nemours_path = '/home/suraj/Git/SCR-Progression/Nemours_Jing_RL_Annotated.h5'
        if os.path.exists(nemours_path):
            print("Loading Nemours dataset...")
            nemours_images, nemours_layers = load_nemours_data(nemours_path, normalize=True)
            if max_samples_per_dataset:
                nemours_images = nemours_images[:max_samples_per_dataset]
                nemours_layers = nemours_layers[:max_samples_per_dataset]
            datasets.append((nemours_images, nemours_layers, "Nemours"))
        else:
            print(f"Nemours dataset not found at {nemours_path}")
    
    if use_duke_data:
        duke_path = '/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control_normalized_corrected.h5'
        if os.path.exists(duke_path):
            print("Loading Duke dataset...")
            duke_images, duke_layers = load_duke_data(duke_path)
            if max_samples_per_dataset:
                duke_images = duke_images[:max_samples_per_dataset]
                duke_layers = duke_layers[:max_samples_per_dataset]
            datasets.append((duke_images, duke_layers, "Duke"))
        else:
            print(f"Duke dataset not found at {duke_path}")
    
    if not datasets:
        print("No datasets found. Please ensure at least one dataset is available.")
        exit(1)
    
    # Combine datasets
    if len(datasets) > 1:
        images, layer_maps = combine_datasets(datasets)
    else:
        images, layer_maps = datasets[0][0], datasets[0][1]
        print(f"Using single dataset: {datasets[0][2]}")
    
    print(f"Final dataset: {len(images)} samples")
    print(f"Image shape: {images.shape}")
    print(f"Layer maps shape: {layer_maps.shape}")
    print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Layer maps range: [{layer_maps.min():.3f}, {layer_maps.max():.3f}]")
    
    # Check for NaN values in the raw data
    print(f"Images contain NaN: {np.isnan(images).any()}")
    print(f"Layer maps contain NaN: {np.isnan(layer_maps).any()}")
    
    if np.isnan(images).any():
        nan_count = np.isnan(images).sum()
        print(f"Number of NaN values in images: {nan_count}")
        
    if np.isnan(layer_maps).any():
        nan_count = np.isnan(layer_maps).sum()
        print(f"Number of NaN values in layer_maps: {nan_count}")
        # Find which samples have NaN values
        nan_samples = np.isnan(layer_maps).any(axis=(1, 2))
        print(f"Samples with NaN layer maps: {np.where(nan_samples)[0]}")
        
        # Remove samples with NaN values
        valid_samples = ~nan_samples
        images = images[valid_samples]
        layer_maps = layer_maps[valid_samples]
        print(f"After removing NaN samples: {len(images)} samples remaining")
    
    # Shuffle the dataset
    indices = np.random.permutation(len(images))
    images = images[indices]
    layer_maps = layer_maps[indices]
    
    # Validate data integrity
    images, layer_maps = validate_data(images, layer_maps)
    # Split data
    dataset = LayerAnnotationDataset(images, layer_maps)
    n_total = len(dataset)
    n_test = int(0.2 * n_total)
    n_train = n_total - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    # Model, loss, optimizer
    model = LayerAnnotationCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    train_losses = []
    val_losses = []
    val_f1s = []
    val_precisions = []
    val_recalls = []
    val_dice = []
    val_iou = []
    val_coord_maes = []
    val_coord_rmses = []
    val_boundary_accuracies = []

    n_epochs = 10
    print(f"\nStarting training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            # Check for NaN in inputs
            if torch.isnan(imgs).any() or torch.isnan(targets).any():
                print(f"NaN detected in inputs at epoch {epoch+1}")
                print(f"  Images NaN: {torch.isnan(imgs).any()}")
                print(f"  Targets NaN: {torch.isnan(targets).any()}")
                print(f"  Images range: [{imgs.min():.4f}, {imgs.max():.4f}]")
                print(f"  Targets range: [{targets.min():.4f}, {targets.max():.4f}]")
                continue
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print(f"NaN detected in model outputs at epoch {epoch+1}")
                continue
                
            loss = criterion(outputs, targets)
            
            # Check for NaN in loss
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at epoch {epoch+1}")
                continue
                
            loss.backward()
            
            # Check for NaN gradients
            if check_for_nan_gradients(model):
                print(f"NaN gradients detected at epoch {epoch+1}")
                continue
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            train_loss /= num_batches
        else:
            train_loss = float('nan')
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                
                # Check for NaN in validation inputs
                if torch.isnan(imgs).any() or torch.isnan(targets).any():
                    continue
                    
                outputs = model(imgs)
                
                # Check for NaN in validation outputs
                if torch.isnan(outputs).any():
                    continue
                    
                loss = criterion(outputs, targets)
                
                if not torch.isnan(loss).any():
                    val_loss += loss.item()
                    val_batches += 1
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())
        
        if val_batches > 0:
            val_loss /= val_batches
        else:
            val_loss = float('nan')
        val_losses.append(val_loss)

        # Calculate metrics if we have valid outputs
        if len(all_outputs) > 0:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Denormalize for mask generation and coordinate metrics
            pred_denorm = denormalize_layers(all_outputs.numpy(), 0, 224)
            target_denorm = denormalize_layers(all_targets.numpy(), 0, 224)
            
            # Coordinate-based metrics (better for line predictions)
            coord_mae = coordinate_mae(torch.from_numpy(pred_denorm), torch.from_numpy(target_denorm))
            coord_rmse = coordinate_rmse(torch.from_numpy(pred_denorm), torch.from_numpy(target_denorm))
            boundary_acc = boundary_distance_metric(torch.from_numpy(pred_denorm), torch.from_numpy(target_denorm), threshold=5.0)
            
            # Mask-based metrics (with thicker lines for better overlap)
            pred_mask = lines_to_mask(torch.from_numpy(pred_denorm), thickness=3)
            target_mask = lines_to_mask(torch.from_numpy(target_denorm), thickness=3)
            
            dice = dice_coefficient(pred_mask, target_mask)
            iou = iou_metric(pred_mask, target_mask)
            precision, recall, f1 = precision_recall_f1(pred_mask, target_mask)
            
            val_f1s.append(f1)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_dice.append(dice)
            val_iou.append(iou)
            val_coord_maes.append(coord_mae)
            val_coord_rmses.append(coord_rmse)
            val_boundary_accuracies.append(boundary_acc)
        else:
            # Use default values if no valid outputs
            val_f1s.append(0.0)
            val_precisions.append(0.0)
            val_recalls.append(0.0)
            val_dice.append(0.0)
            val_iou.append(0.0)
            val_coord_maes.append(224.0)  # Worst case: full image height error
            val_coord_rmses.append(224.0)
            val_boundary_accuracies.append(0.0)
            dice, iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
            coord_mae, coord_rmse, boundary_acc = 224.0, 224.0, 0.0

        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Coord MAE: {coord_mae:.2f}px, RMSE: {coord_rmse:.2f}px, Boundary Acc: {boundary_acc:.3f}")
        print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}")    # Save model and generate visualizations
    torch.save(model.state_dict(), "CNN_regression_model.pth")
    print("\nGenerating visualizations...")
    
    X_test = np.array([test_set[i][0].numpy().transpose(1, 2, 0) for i in range(min(3, n_test))])
    y_test = np.array([test_set[i][1].numpy() for i in range(min(3, n_test))])
    
    plot_sample_predictions(model, X_test, y_test, num_samples=3, save_dir="pytorch_logs")
    plot_training_metrics_comprehensive(train_losses, val_losses, val_f1s, val_precisions, val_recalls, 
                                      val_dice, val_iou, val_coord_maes, val_coord_rmses, val_boundary_accuracies, 
                                      n_epochs, save_dir="pytorch_logs")
    test_lines_to_mask_visualization(model, X_test, y_test, num_samples=3, save_dir="pytorch_logs/mask_tests")
    
    print("Training complete. Model saved as 'CNN_regression_model.pth'.")
    print("Visualizations saved in 'pytorch_logs/' directory.")
    print(f"\nFinal Performance Summary:")
    print(f"  Final Coordinate MAE: {val_coord_maes[-1]:.2f} pixels")
    print(f"  Final Coordinate RMSE: {val_coord_rmses[-1]:.2f} pixels") 
    print(f"  Final Boundary Accuracy (5px): {val_boundary_accuracies[-1]:.3f}")
    print(f"  Final Dice Score: {val_dice[-1]:.4f}")
    print(f"  Final IoU Score: {val_iou[-1]:.4f}")