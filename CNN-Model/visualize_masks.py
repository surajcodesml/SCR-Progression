import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

def visualize_sample_masks(sample_idx=0):
    """Visualize the original image with overlaid layer annotations and generated masks"""
    
    # Load original image
    h5_file_path = '/home/suraj/Git/SCR-Progression/e2e/Nemours_Jing_RL_Annotated.h5'
    with h5py.File(h5_file_path, 'r') as f:
        original_image = f['images'][sample_idx]  # Shape: (496, 768)
        
        # Load layer boundaries
        ilm_boundary = f['layers']['ILM'][sample_idx]
        pr1_boundary = f['layers']['PR1'][sample_idx] 
        bm_boundary = f['layers']['BM'][sample_idx]
    
    # Load generated masks
    multiclass_mask = cv2.imread(f'/home/suraj/Git/SCR-Progression/CNN-Model/masks/multiclass_mask_sample_{sample_idx}.png', cv2.IMREAD_GRAYSCALE)
    ilm_mask = cv2.imread(f'/home/suraj/Git/SCR-Progression/CNN-Model/masks/binary_mask_ILM_sample_{sample_idx}.png', cv2.IMREAD_GRAYSCALE)
    pr1_mask = cv2.imread(f'/home/suraj/Git/SCR-Progression/CNN-Model/masks/binary_mask_PR1_sample_{sample_idx}.png', cv2.IMREAD_GRAYSCALE)
    bm_mask = cv2.imread(f'/home/suraj/Git/SCR-Progression/CNN-Model/masks/binary_mask_BM_sample_{sample_idx}.png', cv2.IMREAD_GRAYSCALE)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image with layer boundaries overlaid
    axes[0, 0].imshow(original_image, cmap='gray')
    x_coords = np.arange(768)
    valid_x = ~np.isnan(ilm_boundary)
    if np.any(valid_x):
        axes[0, 0].plot(x_coords[valid_x], ilm_boundary[valid_x], 'r-', linewidth=2, label='ILM')
    valid_x = ~np.isnan(pr1_boundary)
    if np.any(valid_x):
        axes[0, 0].plot(x_coords[valid_x], pr1_boundary[valid_x], 'g-', linewidth=2, label='PR1')
    valid_x = ~np.isnan(bm_boundary)
    if np.any(valid_x):
        axes[0, 0].plot(x_coords[valid_x], bm_boundary[valid_x], 'b-', linewidth=2, label='BM')
    axes[0, 0].set_title(f'Original Image with Layer Boundaries (Sample {sample_idx})')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('X coordinate')
    axes[0, 0].set_ylabel('Y coordinate')
    
    # Multiclass mask
    axes[0, 1].imshow(multiclass_mask, cmap='viridis')
    axes[0, 1].set_title('Multiclass Mask\n(0=bg, 64=ILM-PR1, 128=PR1-BM, 192=below BM)')
    axes[0, 1].set_xlabel('X coordinate')
    axes[0, 1].set_ylabel('Y coordinate')
    
    # Original image
    axes[0, 2].imshow(original_image, cmap='gray')
    axes[0, 2].set_title('Original OCT Image')
    axes[0, 2].set_xlabel('X coordinate')
    axes[0, 2].set_ylabel('Y coordinate')
    
    # Binary masks
    axes[1, 0].imshow(ilm_mask, cmap='gray')
    axes[1, 0].set_title('ILM Binary Mask')
    axes[1, 0].set_xlabel('X coordinate')
    axes[1, 0].set_ylabel('Y coordinate')
    
    axes[1, 1].imshow(pr1_mask, cmap='gray')
    axes[1, 1].set_title('PR1 Binary Mask')
    axes[1, 1].set_xlabel('X coordinate')
    axes[1, 1].set_ylabel('Y coordinate')
    
    axes[1, 2].imshow(bm_mask, cmap='gray')
    axes[1, 2].set_title('BM Binary Mask')
    axes[1, 2].set_xlabel('X coordinate')
    axes[1, 2].set_ylabel('Y coordinate')
    
    plt.tight_layout()
    plt.savefig(f'/home/suraj/Git/SCR-Progression/CNN-Model/masks/visualization_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as: masks/visualization_sample_{sample_idx}.png")

if __name__ == "__main__":
    # Visualize all 3 samples
    for i in range(3):
        print(f"\nVisualizing sample {i}...")
        visualize_sample_masks(i)
