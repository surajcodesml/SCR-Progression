import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

def load_nemours_data_sample(hdf5_path, target_size=(224, 224), normalize=True, num_samples=2):
    """Load a few samples from Nemours dataset for testing"""
    with h5py.File(hdf5_path, 'r') as f:
        # Load first few images
        images = f['images'][:num_samples]
        
        # Load layer annotations - only ILM and BM
        layers = f['layers']
        ilm_annotations = layers['ILM'][:num_samples]
        bm_annotations = layers['BM'][:num_samples]
        
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
            new_x = np.linspace(0, original_width - 1, target_width)
            original_x = np.arange(original_width)
            
            # Interpolate ILM and BM y-coordinates
            ilm_y_resized = np.interp(new_x, original_x, ilm_annotations[i])
            bm_y_resized = np.interp(new_x, original_x, bm_annotations[i])
            
            # Scale y-coordinates to new height
            ilm_y_resized = ilm_y_resized * height_scale
            bm_y_resized = bm_y_resized * height_scale
            
            # Store denormalized coordinates for visualization
            layer_map_denorm = np.column_stack([ilm_y_resized, bm_y_resized])
            
            # Also create normalized version
            if normalize:
                ilm_y_norm = ilm_y_resized / target_height
                bm_y_norm = bm_y_resized / target_height
                # Replace NaN values
                ilm_y_norm = np.nan_to_num(ilm_y_norm, nan=0.3, posinf=1.0, neginf=0.0)
                bm_y_norm = np.nan_to_num(bm_y_norm, nan=0.7, posinf=1.0, neginf=0.0)
                layer_map_norm = np.column_stack([ilm_y_norm, bm_y_norm])
            else:
                layer_map_norm = layer_map_denorm
            
            resized_layer_maps.append((layer_map_denorm, layer_map_norm))
        
        # Convert to numpy arrays
        images = np.array(resized_images)
        if images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
        
        return images, resized_layer_maps

def lines_to_mask(lines, height=224, width=224):
    """Convert line coordinates to binary masks"""
    # lines: (batch, width, 2) - for each x, y1 and y2
    # Returns: (batch, 2, height, width) binary masks for ILM and BM
    batch = lines.shape[0]
    mask = torch.zeros((batch, 2, height, width), device=lines.device)
    for b in range(batch):
        for i in range(2):  # ILM and BM
            y_coords = torch.clamp(lines[b, :, i].round().long(), 0, height-1)
            mask[b, i, y_coords, torch.arange(width)] = 1
    return mask

def test_mask_function_before_training():
    """Test lines_to_mask function with sample data before training"""
    
    # Load sample data
    nemours_path = '/home/suraj/Git/SCR-Progression/Nemours_Jing_RL_Annotated.h5'
    if not os.path.exists(nemours_path):
        print(f"Dataset not found at {nemours_path}")
        return
    
    print("Loading sample data for mask testing...")
    images, layer_maps_list = load_nemours_data_sample(nemours_path, num_samples=2)
    
    # Create output directory
    os.makedirs("mask_test_before_training", exist_ok=True)
    
    for idx in range(len(images)):
        img = images[idx]
        layer_map_denorm, layer_map_norm = layer_maps_list[idx]
        
        print(f"\nSample {idx}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Layer map denorm shape: {layer_map_denorm.shape}")
        print(f"  Layer map norm shape: {layer_map_norm.shape}")
        print(f"  Denorm ILM range: [{layer_map_denorm[:, 0].min():.1f}, {layer_map_denorm[:, 0].max():.1f}]")
        print(f"  Denorm BM range: [{layer_map_denorm[:, 1].min():.1f}, {layer_map_denorm[:, 1].max():.1f}]")
        print(f"  Norm ILM range: [{layer_map_norm[:, 0].min():.3f}, {layer_map_norm[:, 0].max():.3f}]")
        print(f"  Norm BM range: [{layer_map_norm[:, 1].min():.3f}, {layer_map_norm[:, 1].max():.3f}]")
        
        # Convert to torch tensor for mask generation (use denormalized coordinates)
        layers_torch = torch.from_numpy(layer_map_denorm).unsqueeze(0).float()
        
        # Generate mask using lines_to_mask function
        mask = lines_to_mask(layers_torch)
        
        print(f"  Generated mask shape: {mask.shape}")
        print(f"  ILM mask sum: {mask[0, 0].sum().item()}")
        print(f"  BM mask sum: {mask[0, 1].sum().item()}")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Original image, Line annotations, Mask overlay
        axes[0, 0].imshow(img[:, :, 0], cmap='gray')
        axes[0, 0].set_title(f'Sample {idx}: Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img[:, :, 0], cmap='gray')
        axes[0, 1].plot(range(224), layer_map_denorm[:, 0], 'g-', label='ILM', linewidth=2)
        axes[0, 1].plot(range(224), layer_map_denorm[:, 1], 'b-', label='BM', linewidth=2)
        axes[0, 1].set_title('Line Annotations')
        axes[0, 1].legend()
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
        axes[0, 2].imshow(mask[0, 0].numpy(), cmap='Reds', alpha=0.6)
        axes[0, 2].imshow(mask[0, 1].numpy(), cmap='Blues', alpha=0.6)
        axes[0, 2].set_title('Masks Overlay (Red: ILM, Blue: BM)')
        axes[0, 2].axis('off')
        
        # Row 2: Individual masks and mask combination
        axes[1, 0].imshow(mask[0, 0].numpy(), cmap='Reds')
        axes[1, 0].set_title('ILM Mask Only')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask[0, 1].numpy(), cmap='Blues')
        axes[1, 1].set_title('BM Mask Only')
        axes[1, 1].axis('off')
        
        # Combined mask visualization
        combined_mask = mask[0, 0].numpy() + mask[0, 1].numpy() * 2
        axes[1, 2].imshow(combined_mask, cmap='viridis')
        axes[1, 2].set_title('Combined Masks (1=ILM, 2=BM, 3=Overlap)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'mask_test_before_training/mask_test_sample_{idx}.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # Also create a detailed analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot y-coordinates vs x-coordinates
        axes[0, 0].plot(range(224), layer_map_denorm[:, 0], 'g-', label='ILM', linewidth=2)
        axes[0, 0].plot(range(224), layer_map_denorm[:, 1], 'b-', label='BM', linewidth=2)
        axes[0, 0].set_xlabel('X coordinate')
        axes[0, 0].set_ylabel('Y coordinate (pixels)')
        axes[0, 0].set_title('Layer Coordinates (Denormalized)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(224, 0)  # Invert y-axis to match image coordinates
        
        # Plot normalized coordinates
        axes[0, 1].plot(range(224), layer_map_norm[:, 0], 'g-', label='ILM', linewidth=2)
        axes[0, 1].plot(range(224), layer_map_norm[:, 1], 'b-', label='BM', linewidth=2)
        axes[0, 1].set_xlabel('X coordinate')
        axes[0, 1].set_ylabel('Y coordinate (normalized)')
        axes[0, 1].set_title('Layer Coordinates (Normalized)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(1, 0)  # Invert y-axis
        
        # Show mask statistics
        ilm_mask_coords = torch.nonzero(mask[0, 0])
        bm_mask_coords = torch.nonzero(mask[0, 1])
        
        if len(ilm_mask_coords) > 0:
            axes[1, 0].scatter(ilm_mask_coords[:, 1], ilm_mask_coords[:, 0], 
                             c='red', alpha=0.6, s=1, label='ILM mask pixels')
        if len(bm_mask_coords) > 0:
            axes[1, 0].scatter(bm_mask_coords[:, 1], bm_mask_coords[:, 0], 
                             c='blue', alpha=0.6, s=1, label='BM mask pixels')
        axes[1, 0].set_xlabel('X coordinate')
        axes[1, 0].set_ylabel('Y coordinate')
        axes[1, 0].set_title('Mask Pixel Locations')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(224, 0)  # Invert y-axis
        
        # Show histogram of y-coordinates
        axes[1, 1].hist(layer_map_denorm[:, 0], bins=20, alpha=0.7, color='green', label='ILM')
        axes[1, 1].hist(layer_map_denorm[:, 1], bins=20, alpha=0.7, color='blue', label='BM')
        axes[1, 1].set_xlabel('Y coordinate (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Y Coordinates')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'mask_test_before_training/mask_analysis_sample_{idx}.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()

def test_metric_calculations():
    """Test metric calculations with synthetic data to understand potential issues"""
    print("\n" + "="*50)
    print("TESTING METRIC CALCULATIONS")
    print("="*50)
    
    # Create synthetic perfect prediction case
    height, width = 224, 224
    batch_size = 1
    
    # Create a simple case: horizontal lines
    true_lines = torch.zeros(batch_size, width, 2)
    true_lines[0, :, 0] = 50  # ILM at y=50
    true_lines[0, :, 1] = 150  # BM at y=150
    
    # Test 1: Perfect prediction
    pred_lines = true_lines.clone()
    
    true_mask = lines_to_mask(true_lines, height, width)
    pred_mask = lines_to_mask(pred_lines, height, width)
    
    print("\nTest 1: Perfect Prediction")
    print(f"True mask shape: {true_mask.shape}")
    print(f"Pred mask shape: {pred_mask.shape}")
    print(f"True ILM mask sum: {true_mask[0, 0].sum().item()}")
    print(f"True BM mask sum: {true_mask[0, 1].sum().item()}")
    print(f"Pred ILM mask sum: {pred_mask[0, 0].sum().item()}")
    print(f"Pred BM mask sum: {pred_mask[0, 1].sum().item()}")
    
    # Calculate metrics manually
    def calculate_metrics_detailed(pred_mask, true_mask):
        eps = 1e-6
        
        # Calculate for each layer separately
        for layer_idx in range(2):
            layer_name = "ILM" if layer_idx == 0 else "BM"
            pred_layer = pred_mask[0, layer_idx]
            true_layer = true_mask[0, layer_idx]
            
            tp = (pred_layer * true_layer).sum().item()
            fp = (pred_layer * (1 - true_layer)).sum().item()
            fn = ((1 - pred_layer) * true_layer).sum().item()
            tn = ((1 - pred_layer) * (1 - true_layer)).sum().item()
            
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            f1 = (2 * precision * recall + eps) / (precision + recall + eps)
            
            intersection = (pred_layer * true_layer).sum().item()
            union = pred_layer.sum().item() + true_layer.sum().item()
            dice = (2 * intersection + eps) / (union + eps)
            
            iou_union = pred_layer.sum().item() + true_layer.sum().item() - intersection
            iou = (intersection + eps) / (iou_union + eps)
            
            print(f"\n{layer_name} Layer Metrics:")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Dice: {dice:.4f}")
            print(f"  IoU: {iou:.4f}")
            print(f"  Intersection: {intersection}")
            print(f"  Union (Dice): {union}")
            print(f"  Union (IoU): {iou_union}")
    
    calculate_metrics_detailed(pred_mask, true_mask)
    
    # Test 2: Slightly offset prediction
    print("\n" + "-"*30)
    print("Test 2: Slightly Offset Prediction (1 pixel)")
    
    pred_lines_offset = true_lines.clone()
    pred_lines_offset[0, :, 0] = 51  # ILM at y=51 (1 pixel off)
    pred_lines_offset[0, :, 1] = 151  # BM at y=151 (1 pixel off)
    
    pred_mask_offset = lines_to_mask(pred_lines_offset, height, width)
    calculate_metrics_detailed(pred_mask_offset, true_mask)
    
    # Test 3: More realistic wavy lines
    print("\n" + "-"*30)
    print("Test 3: Wavy Lines (More Realistic)")
    
    x_coords = torch.arange(width, dtype=torch.float32)
    true_lines_wavy = torch.zeros(batch_size, width, 2)
    true_lines_wavy[0, :, 0] = 50 + 10 * torch.sin(x_coords * 2 * 3.14159 / width)  # Wavy ILM
    true_lines_wavy[0, :, 1] = 150 + 15 * torch.cos(x_coords * 2 * 3.14159 / width)  # Wavy BM
    
    # Prediction with small error
    pred_lines_wavy = true_lines_wavy.clone()
    pred_lines_wavy[0, :, 0] += torch.randn(width) * 2  # Add noise
    pred_lines_wavy[0, :, 1] += torch.randn(width) * 2  # Add noise
    
    true_mask_wavy = lines_to_mask(true_lines_wavy, height, width)
    pred_mask_wavy = lines_to_mask(pred_lines_wavy, height, width)
    
    calculate_metrics_detailed(pred_mask_wavy, true_mask_wavy)
    
    # Create visualization of this test case
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # True masks
    axes[0, 0].imshow(true_mask_wavy[0, 0].numpy(), cmap='Reds')
    axes[0, 0].set_title('True ILM Mask')
    
    axes[0, 1].imshow(true_mask_wavy[0, 1].numpy(), cmap='Blues')
    axes[0, 1].set_title('True BM Mask')
    
    axes[0, 2].imshow(true_mask_wavy[0, 0].numpy() + true_mask_wavy[0, 1].numpy(), cmap='viridis')
    axes[0, 2].set_title('True Combined Masks')
    
    # Predicted masks
    axes[1, 0].imshow(pred_mask_wavy[0, 0].numpy(), cmap='Reds')
    axes[1, 0].set_title('Pred ILM Mask')
    
    axes[1, 1].imshow(pred_mask_wavy[0, 1].numpy(), cmap='Blues')
    axes[1, 1].set_title('Pred BM Mask')
    
    axes[1, 2].imshow(pred_mask_wavy[0, 0].numpy() + pred_mask_wavy[0, 1].numpy(), cmap='viridis')
    axes[1, 2].set_title('Pred Combined Masks')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_test_before_training/metric_test_visualization.png', 
               bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    print("Testing lines_to_mask function before training...")
    test_mask_function_before_training()
    test_metric_calculations()
    print("\nMask testing complete! Check 'mask_test_before_training' directory for results.")
