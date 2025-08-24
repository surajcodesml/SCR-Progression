'''
This script is used to perform inference using a pre-trained CNN_pytorch_model.pth model 
which was trained on LayerAnnotationCNN architecture from CNN_pytorch.py file.
Using to test the model performance on a large set of test data.
'''

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
from datetime import datetime
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model architecture
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
        x = x.view(-1, 224, 2)  # Output shape: (batch_size, 224, 2)
        return x

# Custom Dataset for inference
class InferenceDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images.astype(np.float32)
        self.masks = masks.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        # Transpose image to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(mask), idx

def generate_annotation_masks(images, layers, height=None, width=None):
    """
    Generate segmentation masks from ILM, PR1, and BM layer annotations.
    Class 0: Background
    Class 1: ILM to PR1 region
    Class 2: PR1 to BM region
    """
    if height is None:
        height = images.shape[1]
    if width is None:
        width = images.shape[2]

    batch_size = images.shape[0]
    masks = np.zeros((batch_size, height, width), dtype=np.uint8)

    # Check if coordinates are normalized
    ilm_sample = layers['ILM'][0] if len(layers['ILM']) > 0 else []
    pr1_sample = layers['PR1'][0] if len(layers['PR1']) > 0 else []
    bm_sample = layers['BM'][0] if len(layers['BM']) > 0 else []
    
    all_coords = []
    if len(ilm_sample) > 0:
        all_coords.extend(ilm_sample[~np.isnan(ilm_sample)])
    if len(pr1_sample) > 0:
        all_coords.extend(pr1_sample[~np.isnan(pr1_sample)])
    if len(bm_sample) > 0:
        all_coords.extend(bm_sample[~np.isnan(bm_sample)])
    
    is_normalized = len(all_coords) > 0 and np.max(all_coords) <= 1.0

    for b in range(batch_size):
        ilm_line = layers['ILM'][b]
        pr1_line = layers['PR1'][b]
        bm_line = layers['BM'][b]

        for x in range(width):
            ilm_y = ilm_line[x]
            pr1_y = pr1_line[x]
            bm_y = bm_line[x]

            if np.isnan(ilm_y) or np.isnan(pr1_y) or np.isnan(bm_y):
                continue

            # Scale coordinates if they are normalized
            if is_normalized:
                ilm_y = ilm_y * height
                pr1_y = pr1_y * height
                bm_y = bm_y * height

            ilm_y = int(np.clip(round(ilm_y), 0, height-1))
            pr1_y = int(np.clip(round(pr1_y), 0, height-1))
            bm_y = int(np.clip(round(bm_y), 0, height-1))

            top, mid, bottom = sorted([ilm_y, pr1_y, bm_y])

            if top < mid:
                masks[b, top:mid, x] = 1
            if mid < bottom:
                masks[b, mid:bottom, x] = 2

    return masks

def load_duke_data_with_masks(hdf5_path, target_size=(224, 224), start_idx=None, num_samples=None):
    """
    Load data from Duke dataset HDF5 file and generate segmentation masks.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        target_size (tuple): Target size for images and masks (height, width)
        start_idx (int): Starting index for data slice (if None, starts from beginning)
        num_samples (int): Number of samples to load (if None, loads all from start_idx)
    
    Returns:
        tuple: (images, masks) where images have shape (N, H, W, 1) 
               and masks have shape (N, H, W) with classes 0,1,2
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Get total dataset size first
        total_samples = f['images'].shape[0]
        print(f"Total samples in dataset: {total_samples}")
        
        # Determine slice indices
        if start_idx is None:
            start_idx = 0
        if num_samples is None:
            end_idx = total_samples
        else:
            end_idx = min(start_idx + num_samples, total_samples)
        
        print(f"Loading samples {start_idx} to {end_idx-1} ({end_idx-start_idx} samples)")
        
        # Load data slice
        images_orig = f['images'][start_idx:end_idx]
        layer_maps = f['layer_maps'][start_idx:end_idx]
        layer_names = [name.decode() for name in f['layer_names'][:]]
    
    print(f"Loaded image shape: {images_orig.shape}")
    print(f"Layer names: {layer_names}")
    
    # Find layer indices
    ilm_idx = None
    pr1_idx = None
    bm_idx = None
    
    for i, name in enumerate(layer_names):
        name_upper = name.upper()
        if 'ILM' in name_upper:
            ilm_idx = i
        elif any(layer in name_upper for layer in ['PR1', 'PRE', 'RPEDC', 'PHOTORECEPTOR']):
            pr1_idx = i
        elif any(layer in name_upper for layer in ['BM', 'BRUCH', 'MEMBRANE']):
            bm_idx = i
    
    if ilm_idx is None or pr1_idx is None or bm_idx is None:
        raise ValueError(f"Could not find required layers. Available: {layer_names}")
    
    print(f"Using layers: ILM (idx {ilm_idx}), PR1 (idx {pr1_idx}), BM (idx {bm_idx})")
    
    layer_data = {
        'ILM': layer_maps[:, :, ilm_idx],
        'PR1': layer_maps[:, :, pr1_idx], 
        'BM': layer_maps[:, :, bm_idx]
    }
    
    # Print layer coordinate ranges
    print("Layer coordinate ranges:")
    for layer_name, coords in layer_data.items():
        valid_coords = coords[~np.isnan(coords)]
        if len(valid_coords) > 0:
            print(f"  {layer_name}: [{valid_coords.min():.3f}, {valid_coords.max():.3f}] (mean: {valid_coords.mean():.3f})")
        else:
            print(f"  {layer_name}: No valid coordinates found")
    
    # Normalize images if needed
    if images_orig.max() > 1.0:
        images_orig = images_orig.astype(np.float32) / 255.0
    
    # Generate masks
    print("Generating segmentation masks...")
    masks_orig = generate_annotation_masks(images_orig, layer_data)
    print(f"Generated masks shape: {masks_orig.shape}")
    print(f"Mask classes: {np.unique(masks_orig)}")
    
    # Images are already at target size (224x224), so no resizing needed
    images = images_orig
    masks = masks_orig
    
    # Add channel dimension to images if needed
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    
    print(f"Final image shape: {images.shape}")
    print(f"Final mask shape: {masks.shape}")
    print(f"Final mask classes: {np.unique(masks)}")
    
    return images, masks

def predictions_to_masks(predictions, height=224, width=224):
    """
    Convert regression predictions (layer coordinates) to segmentation masks.
    
    Args:
        predictions: Tensor of shape (batch_size, 224, 2) where 2 represents [ILM, BM] coordinates
        height: Target mask height
        width: Target mask width
    
    Returns:
        masks: Numpy array of shape (batch_size, height, width) with class labels 0,1,2
    """
    batch_size = predictions.shape[0]
    masks = np.zeros((batch_size, height, width), dtype=np.uint8)
    
    # Convert to numpy if tensor
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    
    for b in range(batch_size):
        pred_coords = predictions[b]  # Shape: (224, 2)
        
        for x in range(width):
            if x < pred_coords.shape[0]:
                # Predictions are normalized coordinates [0, 1], need to scale to pixel coordinates
                ilm_y = pred_coords[x, 0] * height  # Scale to pixel coordinates
                bm_y = pred_coords[x, 1] * height   # Scale to pixel coordinates
                
                # Clip coordinates to valid range
                ilm_y = int(np.clip(round(ilm_y), 0, height-1))
                bm_y = int(np.clip(round(bm_y), 0, height-1))
                
                # Ensure proper order (ILM should be above BM)
                if ilm_y > bm_y:
                    ilm_y, bm_y = bm_y, ilm_y
                
                # Create a reasonable middle layer approximation
                # Divide the region between ILM and BM into two equal parts
                total_thickness = bm_y - ilm_y
                if total_thickness > 2:  # Only create layers if there's enough space
                    middle_y = ilm_y + total_thickness // 2
                    
                    # Class 1: ILM to middle region (upper retinal layers)
                    if ilm_y < middle_y:
                        masks[b, ilm_y:middle_y, x] = 1
                    # Class 2: middle to BM region (lower retinal layers)
                    if middle_y < bm_y:
                        masks[b, middle_y:bm_y, x] = 2
                elif total_thickness > 0:  # Very thin region, assign to class 1
                    masks[b, ilm_y:bm_y, x] = 1
    
    return masks

def calculate_segmentation_metrics(pred, target, num_classes=3):
    """Calculate comprehensive segmentation metrics per class and overall"""
    pred_classes = torch.argmax(pred, dim=1) if pred.dim() > 3 else pred
    
    per_class_metrics = {}
    
    for class_id in range(num_classes):
        pred_mask = (pred_classes == class_id).float()
        target_mask = (target == class_id).float()
        
        tp = (pred_mask * target_mask).sum()
        fp = (pred_mask * (1 - target_mask)).sum()
        fn = ((1 - pred_mask) * target_mask).sum()
        
        # Dice and IoU
        intersection = tp
        union = pred_mask.sum() + target_mask.sum()
        
        if union > 0:
            dice = (2 * intersection) / union
            iou = intersection / (union - intersection)
        else:
            dice = 1.0
            iou = 1.0
        
        # Precision, Recall, F1
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 1.0 if tp + fn == 0 else 0.0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 1.0 if tp + fp == 0 else 0.0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        per_class_metrics[f'class_{class_id}'] = {
            'dice': dice.item() if hasattr(dice, 'item') else float(dice),
            'iou': iou.item() if hasattr(iou, 'item') else float(iou),
            'precision': precision.item() if hasattr(precision, 'item') else float(precision),
            'recall': recall.item() if hasattr(recall, 'item') else float(recall),
            'f1': f1.item() if hasattr(f1, 'item') else float(f1)
        }
    
    # Calculate overall averages
    overall_metrics = {}
    for metric in ['dice', 'iou', 'precision', 'recall', 'f1']:
        overall_metrics[metric] = np.mean([per_class_metrics[f'class_{i}'][metric] for i in range(num_classes)])
    
    return overall_metrics, per_class_metrics

def calculate_regression_metrics(pred_masks, target_masks, num_classes=3):
    """Calculate segmentation metrics from numpy mask arrays"""
    overall_metrics = {}
    per_class_metrics = {}
    
    for class_id in range(num_classes):
        pred_mask = (pred_masks == class_id).astype(float)
        target_mask = (target_masks == class_id).astype(float)
        
        tp = (pred_mask * target_mask).sum()
        fp = (pred_mask * (1 - target_mask)).sum()
        fn = ((1 - pred_mask) * target_mask).sum()
        
        # Dice and IoU
        intersection = tp
        union = pred_mask.sum() + target_mask.sum()
        
        if union > 0:
            dice = (2 * intersection) / union
            iou = intersection / (union - intersection)
        else:
            dice = 1.0
            iou = 1.0
        
        # Precision, Recall, F1
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 1.0 if tp + fn == 0 else 0.0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 1.0 if tp + fp == 0 else 0.0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        per_class_metrics[f'class_{class_id}'] = {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    # Calculate overall averages
    for metric in ['dice', 'iou', 'precision', 'recall', 'f1']:
        overall_metrics[metric] = np.mean([per_class_metrics[f'class_{i}'][metric] for i in range(num_classes)])
    
    return overall_metrics, per_class_metrics

def save_sample_predictions(model, images, masks, indices, save_dir, num_samples=8):
    """Save sample prediction visualizations"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a grid of samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    class_names = ['Background', 'ILM-Middle Region', 'Middle-BM Region']
    
    with torch.no_grad():
        for i, idx in enumerate(indices[:num_samples]):
            img = images[idx]
            true_mask = masks[idx]
            
            # Get prediction (regression output)
            inp = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
            pred_coords = model(inp)  # Shape: (1, 224, 2)
            
            # Convert to mask
            pred_mask = predictions_to_masks(pred_coords.cpu(), height=224, width=224)[0]
            
            # Original image
            axes[i, 0].imshow(img[:, :, 0], cmap='gray')
            axes[i, 0].set_title(f'Sample {idx}: Original')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            masked_true = np.ma.masked_where(true_mask == 0, true_mask)
            axes[i, 1].imshow(masked_true, cmap='viridis', alpha=0.8, vmin=0, vmax=2)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
            axes[i, 2].imshow(masked_pred, cmap='viridis', alpha=0.8, vmin=0, vmax=2)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_predictions.png'), 
               bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # Configuration
    model_path = "CNN_regression_model.pth"
    duke_path = '/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control_normalized_corrected.h5'
    batch_size = 4  # Conservative for GTX 1650 Ti 4GB
    num_test_samples = 2000  # Test on last 2000 images
    target_size = (224, 224)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"inference_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Load model
    print("Loading trained model...")
    model = LayerAnnotationCNN().to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load Duke dataset (last 2000 samples)
    print(f"Loading last {num_test_samples} samples from Duke dataset...")
    try:
        # First, get total number of samples to calculate start index
        with h5py.File(duke_path, 'r') as f:
            total_samples = f['images'].shape[0]
        
        start_idx = max(0, total_samples - num_test_samples)
        print(f"Dataset has {total_samples} total samples")
        print(f"Testing on samples {start_idx} to {total_samples-1}")
        
        images, masks = load_duke_data_with_masks(
            duke_path, 
            target_size=target_size, 
            start_idx=start_idx, 
            num_samples=num_test_samples
        )
        
        print(f"✓ Loaded {len(images)} samples for testing")
        print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
        print(f"Mask classes: {np.unique(masks)}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Create dataset and dataloader
    dataset = InferenceDataset(images, masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Run inference
    print(f"Running inference with batch size {batch_size}...")
    model.eval()
    
    all_overall_metrics = []
    all_per_class_metrics = []
    sample_indices = []
    
    # Initialize running totals for metrics
    total_dice_per_class = np.zeros(3)
    total_iou_per_class = np.zeros(3)
    total_precision_per_class = np.zeros(3)
    total_recall_per_class = np.zeros(3)
    total_f1_per_class = np.zeros(3)
    total_samples = 0
    
    with torch.no_grad():
        for batch_imgs, batch_masks, batch_indices in tqdm(dataloader, desc="Processing batches"):
            batch_imgs = batch_imgs.to(device)
            batch_masks = batch_masks.to(device).long()
            
            # Get predictions for entire batch (regression output: layer coordinates)
            outputs = model(batch_imgs)  # Shape: (batch_size, 224, 2)
            
            # Convert regression predictions to segmentation masks
            pred_masks = predictions_to_masks(outputs.cpu(), height=224, width=224)
            
            # Calculate metrics for each sample in batch
            for i in range(len(batch_imgs)):
                sample_pred_mask = pred_masks[i]  # Numpy array
                sample_target_mask = batch_masks[i].cpu().numpy()  # Convert to numpy
                sample_idx = batch_indices[i].item()
                
                overall_metrics, per_class_metrics = calculate_regression_metrics(
                    sample_pred_mask, sample_target_mask
                )
                
                all_overall_metrics.append(overall_metrics)
                all_per_class_metrics.append(per_class_metrics)
                sample_indices.append(sample_idx)
                
                # Accumulate per-class metrics
                for class_id in range(3):
                    total_dice_per_class[class_id] += per_class_metrics[f'class_{class_id}']['dice']
                    total_iou_per_class[class_id] += per_class_metrics[f'class_{class_id}']['iou']
                    total_precision_per_class[class_id] += per_class_metrics[f'class_{class_id}']['precision']
                    total_recall_per_class[class_id] += per_class_metrics[f'class_{class_id}']['recall']
                    total_f1_per_class[class_id] += per_class_metrics[f'class_{class_id}']['f1']
                
                total_samples += 1
    
    # Calculate final averages
    print("Calculating final metrics...")
    
    # Overall averages
    overall_dice = np.mean([m['dice'] for m in all_overall_metrics])
    overall_iou = np.mean([m['iou'] for m in all_overall_metrics])
    overall_precision = np.mean([m['precision'] for m in all_overall_metrics])
    overall_recall = np.mean([m['recall'] for m in all_overall_metrics])
    overall_f1 = np.mean([m['f1'] for m in all_overall_metrics])
    
    # Per-class averages
    mean_dice_per_class = total_dice_per_class / total_samples
    mean_iou_per_class = total_iou_per_class / total_samples
    mean_precision_per_class = total_precision_per_class / total_samples
    mean_recall_per_class = total_recall_per_class / total_samples
    mean_f1_per_class = total_f1_per_class / total_samples
    
    # Print results
    print("\n" + "="*90)
    print("INFERENCE RESULTS ON DUKE DATASET (LAST 2000 SAMPLES)")
    print("="*90)
    print(f"Model: {model_path}")
    print(f"Dataset: {duke_path}")
    print(f"Samples tested: {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Average Dice Score:  {overall_dice:.4f}")
    print(f"  Average IoU Score:   {overall_iou:.4f}")
    print(f"  Average Precision:   {overall_precision:.4f}")
    print(f"  Average Recall:      {overall_recall:.4f}")
    print(f"  Average F1 Score:    {overall_f1:.4f}")
    
    print(f"\n📋 PER-CLASS METRICS:")
    class_names = ['Background (Class 0)', 'ILM-Middle Region (Class 1)', 'Middle-BM Region (Class 2)']
    
    for i, class_name in enumerate(class_names):
        print(f"\n  🔸 {class_name}:")
        print(f"    Dice Score:  {mean_dice_per_class[i]:.4f}")
        print(f"    IoU Score:   {mean_iou_per_class[i]:.4f}")
        print(f"    Precision:   {mean_precision_per_class[i]:.4f}")
        print(f"    Recall:      {mean_recall_per_class[i]:.4f}")
        print(f"    F1 Score:    {mean_f1_per_class[i]:.4f}")
    
    # Save detailed results
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "dataset_path": duke_path,
            "samples_tested": total_samples,
            "start_index": start_idx,
            "end_index": start_idx + total_samples - 1,
            "batch_size": batch_size,
            "target_size": list(target_size),
            "device": str(device)
        },
        "overall_metrics": {
            "dice_score": float(overall_dice),
            "iou_score": float(overall_iou),
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1_score": float(overall_f1)
        },
        "per_class_metrics": {
            "class_0_background": {
                "dice_score": float(mean_dice_per_class[0]),
                "iou_score": float(mean_iou_per_class[0]),
                "precision": float(mean_precision_per_class[0]),
                "recall": float(mean_recall_per_class[0]),
                "f1_score": float(mean_f1_per_class[0])
            },
            "class_1_ilm_pr1": {
                "dice_score": float(mean_dice_per_class[1]),
                "iou_score": float(mean_iou_per_class[1]),
                "precision": float(mean_precision_per_class[1]),
                "recall": float(mean_recall_per_class[1]),
                "f1_score": float(mean_f1_per_class[1])
            },
            "class_2_pr1_bm": {
                "dice_score": float(mean_dice_per_class[2]),
                "iou_score": float(mean_iou_per_class[2]),
                "precision": float(mean_precision_per_class[2]),
                "recall": float(mean_recall_per_class[2]),
                "f1_score": float(mean_f1_per_class[2])
            }
        },
        "detailed_per_sample_metrics": []
    }
    
    # Add per-sample metrics (optional, for detailed analysis)
    for i, (overall, per_class, idx) in enumerate(zip(all_overall_metrics, all_per_class_metrics, sample_indices)):
        sample_result = {
            "sample_index": idx,
            "overall": overall,
            "per_class": per_class
        }
        results["detailed_per_sample_metrics"].append(sample_result)
    
    # Save results to JSON
    results_file = os.path.join(results_dir, "inference_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save sample predictions
    print("\n📸 Generating sample predictions...")
    save_sample_predictions(
        model, images, masks, sample_indices, 
        os.path.join(results_dir, "sample_predictions"), 
        num_samples=8
    )
    
    # Print summary
    print(f"\n✅ Inference completed successfully!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"📄 Detailed metrics: {results_file}")
    print(f"🖼️  Sample predictions: {os.path.join(results_dir, 'sample_predictions')}")
    print("="*90)

if __name__ == "__main__":
    main()
