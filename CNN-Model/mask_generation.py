import h5py
import numpy as np
import cv2  # For saving masks (optional)
import os

# Step 1: Load the HDF5 file
def load_h5_data(file_path, target_layers=['ILM', 'PR1', 'BM'], num_samples=3):
    with h5py.File(file_path, 'r') as f:
        # Load only the first num_samples images
        images = f['images'][:num_samples]  # Shape: (num_samples, height, width)
        layer_group = f['layers']
        
        # Filter only the target layers we want
        layers = {}
        for layer_name in target_layers:
            if layer_name in layer_group:
                layers[layer_name] = layer_group[layer_name][:num_samples]  # Shape: (num_samples, 768)
            else:
                print(f"Warning: Layer '{layer_name}' not found in the data")
        
        layer_names = list(layers.keys())
    return images, layers, layer_names

# Step 2: Generate segmentation masks
def generate_masks(images, layers, layer_names, height=None, width=768):
    if height is None:
        height = images.shape[1]  # Get height from actual image dimensions
        
    batch_size = images.shape[0]
    num_layers = len(layer_names)
    
    # Initialize multi-class mask: batch_size x height x width
    # 0: background, 1: region between ILM and PR1, 2: region between PR1 and BM, 3: region below BM
    multiclass_masks = np.zeros((batch_size, height, width), dtype=np.uint8)
    
    # Initialize binary masks for each layer boundary
    binary_masks = {name: np.zeros((batch_size, height, width), dtype=np.uint8) for name in layer_names}
    
    # Process each image in the batch
    for b in range(batch_size):
        # Get y-coordinates for each layer's boundary
        boundaries = np.array([layers[name][b] for name in layer_names])  # Shape: (num_layers, 768)
        
        # Identify valid x-coordinates (where no layer has NaN)
        valid_x = ~np.any(np.isnan(boundaries), axis=0)  # Shape: (768,)
        
        # Sort boundaries by y-value (top to bottom: ILM topmost, then PR1, then BM bottommost)
        mean_y = np.nanmean(boundaries, axis=1)  # Use nanmean to ignore NaNs
        sorted_indices = np.argsort(mean_y)
        sorted_boundaries = boundaries[sorted_indices]
        sorted_layer_names = [layer_names[i] for i in sorted_indices]
        
        # Round y-coordinates to nearest integer (pixel values) for valid regions
        boundaries_int = np.round(sorted_boundaries).astype(int)
        
        # Clip y-coordinates to valid image range [0, height-1]
        boundaries_int = np.clip(boundaries_int, 0, height - 1)
        
        # Process only valid x-coordinates
        for x in range(width):
            if not valid_x[x]:
                # Skip columns with NaN in any layer (leave as 0 in masks)
                continue
                
            # Get y-coordinates for this x-column (should be 3 layers: ILM, PR1, BM)
            y_coords = boundaries_int[:, x]
            
            if len(y_coords) >= 3:  # Ensure we have all 3 layers
                ilm_y, pr1_y, bm_y = y_coords[0], y_coords[1], y_coords[2]
                
                # Define regions:
                # Region 0: Above ILM (background)
                # Region 1: Between ILM and PR1 (retinal tissue)
                # Region 2: Between PR1 and BM (photoreceptor region)
                # Region 3: Below BM (choroid/background)
                
                # Background above ILM
                if ilm_y > 0:
                    multiclass_masks[b, 0:ilm_y, x] = 0
                
                # Region between ILM and PR1
                if ilm_y < pr1_y:
                    multiclass_masks[b, ilm_y:pr1_y, x] = 1
                
                # Region between PR1 and BM  
                if pr1_y < bm_y:
                    multiclass_masks[b, pr1_y:bm_y, x] = 2
                
                # Region below BM
                if bm_y < height:
                    multiclass_masks[b, bm_y:height, x] = 3
                
                # Create binary masks for each layer boundary
                for i, layer_name in enumerate(sorted_layer_names):
                    y_boundary = y_coords[i]
                    if 0 <= y_boundary < height:
                        # Mark the boundary line in binary mask
                        binary_masks[layer_name][b, y_boundary, x] = 1
    
    return multiclass_masks, binary_masks

# Step 3: Post-process masks (optional)
def smooth_masks(masks, kernel_size=3):
    smoothed_masks = masks.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for b in range(masks.shape[0]):
        smoothed_masks[b] = cv2.morphologyEx(masks[b], cv2.MORPH_CLOSE, kernel)
    return smoothed_masks

# Step 4: Save or visualize masks
def save_masks(multiclass_masks, binary_masks, output_dir='masks'):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {multiclass_masks.shape[0]} sample masks to {output_dir}/")
    
    for b in range(multiclass_masks.shape[0]):
        # Save multi-class mask (scale values for better visibility)
        multiclass_filename = f'{output_dir}/multiclass_mask_sample_{b}.png'
        cv2.imwrite(multiclass_filename, multiclass_masks[b] * 64)  # Scale 0,1,2,3 -> 0,64,128,192
        
        # Save binary masks for each layer
        for layer_name, mask in binary_masks.items():
            binary_filename = f'{output_dir}/binary_mask_{layer_name}_sample_{b}.png'
            cv2.imwrite(binary_filename, mask[b] * 255)  # Binary: 0 or 255
    
    print(f"Saved masks for layers: {list(binary_masks.keys())}")
    print("Multiclass mask regions: 0=background, 1=ILM-PR1, 2=PR1-BM, 3=below_BM")

# Main function
def main(h5_file_path):
    print(f"Loading data from: {h5_file_path}")
    
    # Load data (only 3 samples, only ILM, PR1, BM layers)
    images, layers, layer_names = load_h5_data(h5_file_path)
    
    print(f"Loaded {images.shape[0]} images with shape {images.shape}")
    print(f"Processing layers: {layer_names}")
    
    # Generate masks
    multiclass_masks, binary_masks = generate_masks(images, layers, layer_names)
    
    print(f"Generated masks with shape: {multiclass_masks.shape}")
    
    # Optional: Smooth masks (apply only to non-NaN regions if needed)
    multiclass_masks = smooth_masks(multiclass_masks)
    for layer_name in binary_masks:
        binary_masks[layer_name] = smooth_masks(binary_masks[layer_name])
    
    # Save masks
    save_masks(multiclass_masks, binary_masks)
    
    return multiclass_masks, binary_masks

# Example usage
if __name__ == "__main__":
    h5_file_path = '/home/suraj/Git/SCR-Progression/e2e/Nemours_Jing_RL_Annotated.h5'
    multiclass_masks, binary_masks = main(h5_file_path)