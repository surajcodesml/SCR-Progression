import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.io as sio

def read_mat_file(file_path):
    """Read a .mat file and return the data."""
    try:
        data = sio.loadmat(file_path, squeeze_me=True)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def normalize_volume(volume):
    """Normalize entire volume to [0, 1] range"""
    v_min, v_max = volume.min(), volume.max()
    return (volume - v_min) / (v_max - v_min)

def check_annotation_validity(layer_maps, threshold=0.6):
    """Check if B-scan has enough valid annotations"""
    valid_points = ~np.isnan(layer_maps)
    valid_ratio = np.mean(valid_points)
    return valid_ratio >= threshold

def process_volume(images, layer_maps, start_idx=20, end_idx=80, validity_threshold=0.6):
    """Process one volume of B-scans and their annotations"""
    # Extract relevant B-scans
    volume = images[:, :, start_idx:end_idx]
    annotations = layer_maps[start_idx:end_idx]
    
    # Normalize volume
    volume_norm = normalize_volume(volume)
    
    # Transpose volume to (n_scans, height, width)
    volume_norm = np.transpose(volume_norm, (2, 0, 1))
    
    # Filter based on annotation validity
    valid_scans = []
    for i in range(len(annotations)):
        if check_annotation_validity(annotations[i], validity_threshold):
            valid_scans.append(i)
    
    if not valid_scans:
        return None, None
        
    return volume_norm[valid_scans], annotations[valid_scans]

def convert_to_h5(mat_files, output_path, train_ratio=0.7, val_ratio=0.15):
    """Convert .mat files to HDF5 optimized for ML training"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # First pass: calculate total number of valid scans
    total_scans = 0
    print("Phase 1: Counting valid scans...")
    for mat_file in tqdm(mat_files):
        data = read_mat_file(mat_file)
        if data is None:
            continue
        volume_norm, annotations = process_volume(
            data['images'], 
            data['layerMaps']
        )
        if volume_norm is not None:
            total_scans += len(volume_norm)
    
    print(f"\nTotal valid scans found: {total_scans}")
    
    # Create HDF5 file with pre-allocated space
    print("\nPhase 2: Creating HDF5 dataset...")
    with h5py.File(output_path, 'w') as h5f:
        # Create extensible datasets
        images_ds = h5f.create_dataset(
            'images', 
            shape=(0, 512, 1000),
            maxshape=(total_scans, 512, 1000),
            chunks=(1, 512, 1000),
            compression='gzip'
        )
        
        maps_ds = h5f.create_dataset(
            'layer_maps',
            shape=(0, 1000, 3),
            maxshape=(total_scans, 1000, 3),
            chunks=(1, 1000, 3),
            compression='gzip'
        )
        
        # Metadata
        patient_indices = []  # Track which scans belong to which patient
        current_idx = 0
        
        # Second pass: store data
        print("\nPhase 3: Processing and storing data...")
        for patient_idx, mat_file in enumerate(tqdm(mat_files)):
            data = read_mat_file(mat_file)
            if data is None:
                continue
                
            volume_norm, annotations = process_volume(
                data['images'],
                data['layerMaps']
            )
            
            if volume_norm is None:
                continue
            
            n_scans = len(volume_norm)
            
            # Resize datasets
            new_size = current_idx + n_scans
            images_ds.resize(new_size, axis=0)
            maps_ds.resize(new_size, axis=0)
            
            # Store data
            images_ds[current_idx:new_size] = volume_norm
            maps_ds[current_idx:new_size] = annotations
            
            # Track patient indices
            patient_indices.extend([patient_idx] * n_scans)
            current_idx = new_size
        
        # Store metadata
        h5f.create_dataset('patient_indices', data=np.array(patient_indices))
        h5f.create_dataset('layer_names', data=np.array(
            ["ILM", "RPEDC", "Bruch's Membrane"], dtype='S'))
        
        # Create train/val/test splits
        print("\nPhase 4: Creating data splits...")
        n_samples = len(patient_indices)
        indices = np.random.permutation(n_samples)
        train_idx = int(n_samples * train_ratio)
        val_idx = int(n_samples * (train_ratio + val_ratio))
        
        splits = {
            'train': indices[:train_idx],
            'val': indices[train_idx:val_idx],
            'test': indices[val_idx:]
        }
        
        split_group = h5f.create_group('splits')
        for split_name, split_indices in splits.items():
            split_group.create_dataset(split_name, data=split_indices)
            print(f"{split_name} set size: {len(split_indices)}")

if __name__ == "__main__":
    # Set paths
    data_dir = Path("/home/suraj/Data/Duke_WLOA_RL_Annotated/Control")
    output_file = data_dir / "oct_dataset.h5"
    
    # Get list of .mat files
    mat_files = [f for f in data_dir.glob("*.mat")]
    
    print(f"Found {len(mat_files)} .mat files")
    convert_to_h5(mat_files, output_file)
    print("\nConversion complete!")