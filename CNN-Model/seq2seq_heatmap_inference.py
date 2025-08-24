"""
Heatmap Inference Script for CNN Layer Regression Model

This script:
1. Loads the Nemours dataset
2. Uses a trained CNN regression model to predict layer coordinates (ILM and BM)
3. Creates heatmaps for predicted annotations similar to ground truth
4. Compares predicted vs ground truth heatmaps for each volume

Usage:
    python seq2seq_heatmap_inference.py
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.transform import resize
from datetime import datetime
from tqdm import tqdm


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LayerAnnotationCNN(nn.Module):
    """Regression model architecture from CNN_model_inference.py"""
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
            nn.Linear(256, 224 * 2)  # Output: (batch_size, 224*2) -> reshaped to (224, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        x = x.view(-1, 224, 2)  # Output shape: (batch_size, 224, 2) for [ILM, BM] coords
        return x


class InferenceDataset(Dataset):
    """Dataset for inference"""
    def __init__(self, images):
        self.images = images.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Add channel dimension and transpose to (C, H, W) for PyTorch
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)


def load_nemours_dataset(hdf5_path, target_size=(224, 224)):
    """
    Load Nemours dataset and return images, ground truth layers, and names
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load original data
        images_orig = f['images'][:]  # Shape: (N, 496, 768)
        layers = f['layers']
        names = f['names'][:]
        
        # Extract layer data
        layer_data = {
            'ILM': layers['ILM'][:],
            'BM': layers['BM'][:]  # Only need ILM and BM for heatmaps
        }
    
    print(f"Original image shape: {images_orig.shape}")
    print(f"Available layers: {list(layer_data.keys())}")
    
    # Resize images to target size for model inference
    resized_images = []
    for i in range(images_orig.shape[0]):
        img = images_orig[i]
        resized_img = resize(img, target_size, preserve_range=True, anti_aliasing=True)
        # Normalize to [0, 1]
        if resized_img.max() > 1.0:
            resized_img = resized_img / 255.0
        resized_images.append(resized_img)
    
    images = np.array(resized_images)
    
    # Decode names
    volume_names = [name.decode('utf-8') for name in names]
    
    return images, layer_data, volume_names, images_orig


def convert_regression_to_coordinates(predictions, original_size=(496, 768)):
    """
    Convert regression predictions directly to layer coordinates
    
    Args:
        predictions: Model predictions of shape (N, 224, 2) where 2 = [ILM, BM] coordinates
        original_size: Target size to scale coordinates to
    
    Returns:
        dict: Layer coordinates scaled to original image size
    """
    num_images = predictions.shape[0]
    orig_height, orig_width = original_size
    
    # Convert predictions to numpy if needed
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    
    # DEBUG: Check prediction ranges
    print(f"\n🔍 DEBUG: Prediction Analysis")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Min prediction: {np.min(predictions):.4f}")
    print(f"Max prediction: {np.max(predictions):.4f}")
    print(f"Mean prediction: {np.mean(predictions):.4f}")
    print(f"Sample predictions for first image:")
    print(f"  First 5 ILM coords: {predictions[0, :5, 0]}")
    print(f"  First 5 BM coords: {predictions[0, :5, 1]}")
    
    # CRITICAL FIX: The model seems to output coordinates in the range [0, 0.5] roughly
    # but they represent pixel coordinates for 224x224 images that need scaling
    # Let's check if predictions are already pixel coordinates for 224x224 images
    
    pred_max = np.max(predictions)
    pred_min = np.min(predictions)
    
    print(f"\n🔧 COORDINATE ANALYSIS:")
    if pred_max <= 1.0 and pred_min >= 0.0:
        print(f"✓ Predictions appear to be normalized [0,1]")
        scaling_mode = "normalized"
    elif pred_max <= 224.0 and pred_min >= 0.0:
        print(f"✓ Predictions appear to be pixel coordinates for 224x224")
        scaling_mode = "pixel_224"
    else:
        print(f"⚠️ Predictions are in unexpected range [{pred_min:.3f}, {pred_max:.3f}]")
        print(f"⚠️ Assuming they're normalized but need adjustment")
        scaling_mode = "adjusted_normalized"
    
    # Initialize coordinate arrays
    ilm_coords = np.full((num_images, orig_width), np.nan)
    bm_coords = np.full((num_images, orig_width), np.nan)
    
    for i in range(num_images):
        pred_coords = predictions[i]  # Shape: (224, 2)
        
        # Scale from model output width (224) to original width (768)
        for model_x in range(pred_coords.shape[0]):  # 224 points
            # Map model x-coordinate to original x-coordinate
            orig_x = int((model_x / 224.0) * orig_width)
            
            if orig_x < orig_width:
                # Get coordinates
                ilm_y_raw = pred_coords[model_x, 0]
                bm_y_raw = pred_coords[model_x, 1]
                
                # Skip if coordinates are invalid
                if not (np.isnan(ilm_y_raw) or np.isnan(bm_y_raw)):
                    
                    if scaling_mode == "normalized":
                        # Standard normalized scaling
                        ilm_y = ilm_y_raw * orig_height
                        bm_y = bm_y_raw * orig_height
                    elif scaling_mode == "pixel_224":
                        # Pixel coordinates for 224x224, scale to orig_height
                        ilm_y = ilm_y_raw * (orig_height / 224.0)
                        bm_y = bm_y_raw * (orig_height / 224.0)
                    else:  # adjusted_normalized
                        # The predictions might be normalized but shifted/scaled incorrectly
                        # Let's try treating them as relative positions within the image
                        # Scale from [pred_min, pred_max] to [0, orig_height]
                        ilm_y_norm = (ilm_y_raw - pred_min) / (pred_max - pred_min)
                        bm_y_norm = (bm_y_raw - pred_min) / (pred_max - pred_min)
                        ilm_y = ilm_y_norm * orig_height
                        bm_y = bm_y_norm * orig_height
                    
                    # Ensure ILM is above BM (anatomically correct)
                    if ilm_y > bm_y:
                        ilm_y, bm_y = bm_y, ilm_y
                    
                    # Clip to valid range
                    ilm_y = np.clip(ilm_y, 0, orig_height - 1)
                    bm_y = np.clip(bm_y, 0, orig_height - 1)
                    
                    ilm_coords[i, orig_x] = ilm_y
                    bm_coords[i, orig_x] = bm_y
        
        # Interpolate missing values for smoother curves
        for coord_name, coord_array in [('ILM', ilm_coords[i]), ('BM', bm_coords[i])]:
            valid_indices = ~np.isnan(coord_array)
            if np.sum(valid_indices) > 1:
                # Interpolate NaN values
                x_valid = np.where(valid_indices)[0]
                y_valid = coord_array[valid_indices]
                x_all = np.arange(orig_width)
                coord_array[:] = np.interp(x_all, x_valid, y_valid, 
                                         left=np.nan, right=np.nan)
                
                # Debug: Check interpolated results for first image
                if i == 0:
                    print(f"{coord_name} final range: [{np.nanmin(coord_array):.1f}, {np.nanmax(coord_array):.1f}]")
    
    print(f"✓ Used scaling mode: {scaling_mode}")
    return {
        'ILM': ilm_coords,
        'BM': bm_coords
    }


def create_predicted_heatmap(pred_layers, names, volume_name, figsize=(12, 8), save_path=None):
    """
    Create heatmap from predicted layer coordinates (similar to create_ilm_bm_heatmap)
    """
    # Find all B-scans with the same name (same volume)
    volume_indices = []
    for i, name in enumerate(names):
        if name == volume_name:
            volume_indices.append(i)
    
    print(f"Volume '{volume_name}': Found {len(volume_indices)} B-scans")
    
    if len(volume_indices) == 0:
        print(f"No B-scans found for volume '{volume_name}'")
        return None
    
    # Calculate ILM to BM distance for each B-scan
    num_bscans = len(volume_indices)
    width = pred_layers['ILM'].shape[1]  # Should be 768 for original size
    
    # Initialize distance matrix with NaN
    distance_matrix = np.full((num_bscans, width), np.nan)
    
    for bscan_idx, data_idx in enumerate(volume_indices):
        ilm_coords = pred_layers['ILM'][data_idx]
        bm_coords = pred_layers['BM'][data_idx]
        
        # Calculate distance for each x coordinate
        for x_idx in range(width):
            ilm_y = ilm_coords[x_idx]
            bm_y = bm_coords[x_idx]
            
            # Skip if either coordinate is NaN
            if not (np.isnan(ilm_y) or np.isnan(bm_y)):
                distance = abs(bm_y - ilm_y)
                distance_matrix[bscan_idx, x_idx] = distance
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap - mask NaN values for better visualization
    masked_matrix = np.ma.masked_invalid(distance_matrix)
    
    # Find min and max distance values for color mapping (excluding NaN)
    valid_distances = distance_matrix[~np.isnan(distance_matrix)]
    
    if len(valid_distances) == 0:
        print(f"No valid distances found for volume '{volume_name}'")
        return None
    
    min_distance = np.min(valid_distances)
    max_distance = np.max(valid_distances)
    
    # Create heatmap
    im = ax.imshow(masked_matrix, cmap='viridis', aspect='auto', 
                   vmin=min_distance, vmax=max_distance, origin='lower')
    
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('B-scan index')
    ax.set_title(f'Predicted ILM to BM Distance Heatmap for Volume: {volume_name}')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Depth (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print(f"\nPredicted Distance Statistics:")
    print(f"Valid measurements: {len(valid_distances):,}")
    print(f"Min distance: {min_distance:.2f} pixels")
    print(f"Max distance: {max_distance:.2f} pixels")
    print(f"Mean distance: {np.mean(valid_distances):.2f} ± {np.std(valid_distances):.2f} pixels")
    
    return distance_matrix


def create_ground_truth_heatmap(gt_layers, names, volume_name, figsize=(12, 8), save_path=None):
    """
    Create heatmap from ground truth layer coordinates (from data_ops.ipynb)
    """
    # Find all B-scans with the same name (same volume)
    volume_indices = []
    for i, name in enumerate(names):
        if name == volume_name:
            volume_indices.append(i)
    
    print(f"Volume '{volume_name}': Found {len(volume_indices)} B-scans")
    
    if len(volume_indices) == 0:
        print(f"No B-scans found for volume '{volume_name}'")
        return None
    
    # Calculate ILM to BM distance for each B-scan
    num_bscans = len(volume_indices)
    width = 768
    
    # Initialize distance matrix with NaN
    distance_matrix = np.full((num_bscans, width), np.nan)
    
    for bscan_idx, data_idx in enumerate(volume_indices):
        ilm_coords = gt_layers['ILM'][data_idx]
        bm_coords = gt_layers['BM'][data_idx]
        
        # Calculate distance for each x coordinate
        for x_idx in range(width):
            ilm_y = ilm_coords[x_idx]
            bm_y = bm_coords[x_idx]
            
            # Skip if either coordinate is NaN
            if not (np.isnan(ilm_y) or np.isnan(bm_y)):
                distance = abs(bm_y - ilm_y)
                distance_matrix[bscan_idx, x_idx] = distance
    
    # Replace depth value with np.nan if x-coordinate is 0 and distance is 0
    zero_x0_mask = distance_matrix[:, 0] == 0
    distance_matrix[zero_x0_mask, 0] = np.nan
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap - mask NaN values for better visualization
    masked_matrix = np.ma.masked_invalid(distance_matrix)
    
    # Find min and max distance values for color mapping (excluding NaN)
    valid_distances = distance_matrix[~np.isnan(distance_matrix)]
    min_distance = np.min(valid_distances)
    max_distance = np.max(valid_distances)
    
    # Create heatmap
    im = ax.imshow(masked_matrix, cmap='viridis', aspect='auto', 
                   vmin=min_distance, vmax=max_distance, origin='lower')
    
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('B-scan index')
    ax.set_title(f'Ground Truth ILM to BM Distance Heatmap for Volume: {volume_name}')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Depth (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print(f"\nGround Truth Distance Statistics:")
    print(f"Valid measurements: {len(valid_distances):,}")
    print(f"Min distance: {min_distance:.2f} pixels")
    print(f"Max distance: {max_distance:.2f} pixels")
    print(f"Mean distance: {np.mean(valid_distances):.2f} ± {np.std(valid_distances):.2f} pixels")
    
    return distance_matrix


def list_unique_volumes(names, max_display=10):
    """List unique volume names in the dataset"""
    unique_names = {}
    for i, name in enumerate(names):
        if name not in unique_names:
            unique_names[name] = []
        unique_names[name].append(i)
    
    print(f"Unique volumes ({len(unique_names)} total):")
    displayed = 0
    for vol_name, indices in unique_names.items():
        if displayed < max_display:
            print(f"  '{vol_name}': {len(indices)} B-scans")
            displayed += 1
        elif displayed == max_display:
            print(f"  ... and {len(unique_names) - max_display} more volumes")
            break
    
    return list(unique_names.keys())


def run_inference_and_create_heatmaps(model_path, dataset_path, output_dir="depth-map"):
    """
    Main function to run inference and create heatmaps
    """
    # Create output directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    # results should be saved as depth-map/<timestamp>/
    results_dir = f"{output_dir}/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Load model
    print("Loading trained regression model...")
    model = LayerAnnotationCNN().to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load dataset
    print("Loading Nemours dataset...")
    images, gt_layers, volume_names, original_images = load_nemours_dataset(dataset_path)
    print(f"✓ Loaded {len(images)} images")
    
    # List available volumes
    unique_volumes = list_unique_volumes(volume_names)
    
    # Create inference dataset and dataloader
    dataset = InferenceDataset(images)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Run inference to get predicted coordinates
    print("Running inference to get predicted layer coordinates...")
    all_predictions = []
    
    with torch.no_grad():
        for batch_images in tqdm(dataloader, desc="Inference"):
            batch_images = batch_images.to(device)
            
            # Get model predictions - direct coordinate output
            outputs = model(batch_images)  # Shape: (batch_size, 224, 2)
            
            # Move to CPU and store
            all_predictions.append(outputs.cpu())
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"✓ Generated predictions for {len(all_predictions)} images")
    print(f"Prediction shape: {all_predictions.shape}")
    
    # Convert regression predictions to layer coordinates
    print("Converting predictions to layer coordinates...")
    pred_layers = convert_regression_to_coordinates(all_predictions, original_size=(496, 768))
    print(f"✓ Converted to coordinate format")
    
    # DEBUG: Check ground truth coordinate ranges for comparison
    print(f"\n🔍 DEBUG: Ground Truth Analysis")
    gt_ilm_valid = gt_layers['ILM'][~np.isnan(gt_layers['ILM'])]
    gt_bm_valid = gt_layers['BM'][~np.isnan(gt_layers['BM'])]
    print(f"GT ILM range: [{np.min(gt_ilm_valid):.1f}, {np.max(gt_ilm_valid):.1f}]")
    print(f"GT BM range: [{np.min(gt_bm_valid):.1f}, {np.max(gt_bm_valid):.1f}]")
    print(f"Sample GT distances: {np.abs(gt_bm_valid[:10] - gt_ilm_valid[:10])}")
    
    # Create heatmaps for a few sample volumes
    sample_volumes = unique_volumes[:3]  # First 3 volumes
    
    for vol_name in sample_volumes:
        print(f"\n{'='*60}")
        print(f"Creating heatmaps for volume: {vol_name}")
        print(f"{'='*60}")
        
        # Create ground truth heatmap
        gt_save_path = os.path.join(results_dir, f"gt_heatmap_{vol_name}.png")
        gt_matrix = create_ground_truth_heatmap(
            gt_layers, volume_names, vol_name, 
            figsize=(12, 6), save_path=gt_save_path
        )
        
        # Create predicted heatmap
        pred_save_path = os.path.join(results_dir, f"pred_heatmap_{vol_name}.png")
        pred_matrix = create_predicted_heatmap(
            pred_layers, volume_names, vol_name, 
            figsize=(12, 6), save_path=pred_save_path
        )
        
        # Create comparison plot
        if gt_matrix is not None and pred_matrix is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            
            # Ground truth
            gt_masked = np.ma.masked_invalid(gt_matrix)
            gt_valid = gt_matrix[~np.isnan(gt_matrix)]
            im1 = ax1.imshow(gt_masked, cmap='viridis', aspect='auto', 
                           vmin=np.min(gt_valid), vmax=np.max(gt_valid), origin='lower')
            ax1.set_title(f'Ground Truth: {vol_name}')
            ax1.set_xlabel('X coordinate (pixels)')
            ax1.set_ylabel('B-scan index')
            plt.colorbar(im1, ax=ax1, label='Depth (pixels)')
            
            # Predicted
            pred_masked = np.ma.masked_invalid(pred_matrix)
            pred_valid = pred_matrix[~np.isnan(pred_matrix)]
            if len(pred_valid) > 0:
                im2 = ax2.imshow(pred_masked, cmap='viridis', aspect='auto', 
                               vmin=np.min(pred_valid), vmax=np.max(pred_valid), origin='lower')
                ax2.set_title(f'Predicted: {vol_name}')
                ax2.set_xlabel('X coordinate (pixels)')
                ax2.set_ylabel('B-scan index')
                plt.colorbar(im2, ax=ax2, label='Depth (pixels)')
            else:
                ax2.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=16)
                ax2.set_title(f'Predicted: {vol_name} (No valid data)')
            
            plt.tight_layout()
            comparison_path = os.path.join(results_dir, f"comparison_{vol_name}.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    print(f"\n✅ Heatmap generation completed!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"📊 Generated heatmaps for {len(sample_volumes)} volumes")


def main():
    """Main execution function"""
    # Configuration
    MODEL_PATH = "/home/suraj/Git/SCR-Progression/CNN-Model/CNN_pytorch_model.pth"
    DATASET_PATH = "/home/suraj/Models/SCR-Progression/e2e/Nemours_Jing_0805.h5"
    
    print("🔬 CNN Regression Heatmap Inference Script")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Run inference and create heatmaps
    run_inference_and_create_heatmaps(MODEL_PATH, DATASET_PATH)


if __name__ == "__main__":
    main()
