#!/usr/bin/env python3
"""
E2E to HDF5 Converter
====================

This script converts .e2e files to HDF5 format with raw images and layer annotations.
Supports appending .e2e files to create a comprehensive dataset.

Author: Your Name
Date: July 2025
"""

import os
import h5py
import numpy as np
import eyepy as ep
from pathlib import Path
from typing import Dict, Tuple
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class E2EToHDF5Converter:
    """
    A class to convert .e2e files to HDF5 format with raw data.
    """
    
    def __init__(self, hdf5_path: str):
        """
        Initialize the converter with HDF5 file path.
        
        Args:
            hdf5_path (str): Path to the HDF5 file to create/append to
        """
        self.hdf5_path = hdf5_path
        self.supported_layers = ['ILM', 'BM', 'ELM', 'PR1', 'RPE', 'ONL', 'OPL', 'INL', 'IPL', 'GCL', 'NFL']
        
    def extract_e2e_data(self, e2e_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
        """
        Extract images and layer data from .e2e file.
        
        Args:
            e2e_path (str): Path to the .e2e file
            
        Returns:
            Tuple containing:
                - images (np.ndarray): B-scan images (raw data)
                - layers (Dict[str, np.ndarray]): Layer annotations (raw data)
                - filename (str): Name of the source file
        """
        logger.info(f"Loading .e2e file: {e2e_path}")
        
        # Load the .e2e file
        ev = ep.import_heyex_e2e(e2e_path)
        
        # Extract images (keep original data type)
        images = ev.data  # Shape: (n_bscans, height, width)
        
        # Extract layer data (keep original data type)
        layer_data = {}
        for layer_name in self.supported_layers:
            if layer_name in ev.layers:
                layer_data[layer_name] = ev.layers[layer_name].data
                
        # Get filename without extension
        filename = Path(e2e_path).stem
        
        logger.info(f"Extracted {images.shape[0]} B-scans with {len(layer_data)} layers")
        
        return images, layer_data, filename
    
    def _append_or_create_dataset(self, hdf5_group, dataset_name: str, data: np.ndarray, 
                             compression: str = 'gzip', compression_opts: int = 9) -> None:
        """
        Helper function to create a new dataset or append to existing one.
        
        Args:
            hdf5_group: HDF5 group or file object
            dataset_name: Name of the dataset
            data: Data to add/append
            compression: Compression type
            compression_opts: Compression level
        """
        if dataset_name not in hdf5_group:
            # Create new dataset
            hdf5_group.create_dataset(dataset_name, data=data,
                                    maxshape=(None, *data.shape[1:]),
                                    compression=compression, compression_opts=compression_opts)
            logger.info(f"Created new dataset '{dataset_name}' with shape: {data.shape}")
        else:
            # Append to existing dataset
            existing_dataset = hdf5_group[dataset_name]
            old_size = existing_dataset.shape[0]
            new_size = old_size + data.shape[0]
            existing_dataset.resize((new_size, *existing_dataset.shape[1:]))
            existing_dataset[old_size:new_size] = data
            logger.info(f"Appended to dataset '{dataset_name}'. Total: {new_size}")

    def convert_e2e_file(self, e2e_path: str) -> None:
        """
        Convert a single .e2e file and append it to the HDF5 dataset.
        
        Args:
            e2e_path (str): Path to the .e2e file
        """
        # Extract data from .e2e file
        images, layers, filename = self.extract_e2e_data(e2e_path)
        
        logger.info(f"Adding data from {filename} to HDF5 file: {self.hdf5_path}")
        
        with h5py.File(self.hdf5_path, 'a') as f:
            
            # Handle images dataset
            self._append_or_create_dataset(f, 'images', images)
            
            # Handle layers group
            if 'layers' not in f:
                layers_group = f.create_group('layers')
            else:
                layers_group = f['layers']
            
            # Add each layer
            for layer_name, layer_data in layers.items():
                self._append_or_create_dataset(layers_group, layer_name, layer_data)
            
            # Handle filenames dataset
            dt = h5py.string_dtype(encoding='utf-8')
            filenames_array = np.array([filename] * images.shape[0], dtype=dt)
            self._append_or_create_dataset(f, 'names', filenames_array)
        
        logger.info(f"Successfully processed: {e2e_path}")
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the HDF5 dataset.
        
        Returns:
            Dict: Dataset information
        """
        if not os.path.exists(self.hdf5_path):
            return {"error": "HDF5 file does not exist"}
        
        info = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'images' in f:
                info['images_shape'] = f['images'].shape
                info['images_dtype'] = f['images'].dtype
            
            if 'layers' in f:
                info['layers'] = {}
                for layer_name in f['layers'].keys():
                    info['layers'][layer_name] = {
                        'shape': f['layers'][layer_name].shape,
                        'dtype': f['layers'][layer_name].dtype
                    }
            
            if 'names' in f:
                info['names_count'] = f['names'].shape[0]
                info['unique_files'] = len(set(f['names'][:]))
        
        return info

def main():
    # Configuration
    hdf5_output_path = "/home/suraj/Git/SCR-Progression/e2e/Nemours_Jing_RL_Annotated.h5"
    e2e_folder = "/home/suraj/Git/SCR-Progression/e2e/data"
    
    # Initialize converter
    converter = E2EToHDF5Converter(hdf5_output_path)
    
    # Find all .e2e files in the folder
    e2e_files = [str(f) for f in Path(e2e_folder).glob("*.e2e")]
    if not e2e_files:
        logger.error(f"No .e2e files found in folder: {e2e_folder}")
        return

    logger.info(f"Found {len(e2e_files)} .e2e files in {e2e_folder}")

    # Load already processed filenames if HDF5 exists
    existing_names = set()
    if os.path.exists(hdf5_output_path):
        with h5py.File(hdf5_output_path, "r") as f:
            if "names" in f:
                existing_names = set(str(name) for name in f["names"][:])

    # Process each .e2e file
    for e2e_file in e2e_files:
        filename = Path(e2e_file).stem
        if filename in existing_names:
            logger.info(f"Skipping {e2e_file} (already exists in HDF5)")
            continue
        logger.info(f"Converting file: {e2e_file}")
        try:
            converter.convert_e2e_file(e2e_file)
        except Exception as e:
            logger.error(f"Failed to process {e2e_file}: {e}")

    # Print dataset information
    info = converter.get_dataset_info()
    logger.info("Dataset Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()


'''
def main():
    # Configuration
    hdf5_output_path = "/home/suraj/Git/SCR-Progression/e2e/Nemours_Jing_RL_Annotated.h5"
    test_e2e_file = "/home/suraj/Git/SCR-Progression/e2e-data/245_R_1_1.e2e"
    
    # Initialize converter
    converter = E2EToHDF5Converter(hdf5_output_path)
    
    # Check if test file exists
    if not os.path.exists(test_e2e_file):
        logger.error(f"Test file not found: {test_e2e_file}")
        return
    
    logger.info(f"Converting file: {test_e2e_file}")
    
    # Convert the file
    converter.convert_e2e_file(test_e2e_file)
    
    # Print dataset information
    info = converter.get_dataset_info()
    logger.info("Dataset Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

        '''