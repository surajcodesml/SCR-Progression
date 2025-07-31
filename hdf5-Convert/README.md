# OCT Dataset Conversion Documentation
> Converting Duke OCT WLOA Dataset from MATLAB to HDF5 format

## Overview

This documentation describes the conversion process and structure of the Duke OCT WLOA Dataset from MATLAB (.mat) files to a single HDF5 file optimized for machine learning tasks, particularly retinal layer segmentation.

## Dataset Description

### Original Dataset
- **Name**: Duke OCT WLOA Dataset
- **Format**: MATLAB (.mat) files
- **Size**: 269 OCT volumes
- **Content**: OCT B-scans with manual layer annotations
- **Layer Annotations**: 3 retinal boundaries
  - Inner Limiting Membrane (ILM)
  - Retinal Pigment Epithelium/Drusen Complex (RPEDC)
  - Bruch's Membrane

### Data Dimensions

#### Input (.mat) Structure
- **OCT Volume**: `images` array (512 × 1000 × 100)
  - 512: Axial depth (A-scan)
  - 1000: Lateral width
  - 100: Number of B-scans
- **Layer Annotations**: `layerMaps` array (100 × 1000 × 3)
  - 100: Number of B-scans
  - 1000: Lateral positions
  - 3: Layer boundaries

#### Output (HDF5) Structure
```
oct_dataset.h5/
├── images/                 # (N, 512, 1000) - Normalized B-scans
├── layer_maps/            # (N, 1000, 3) - Layer annotations
├── patient_indices/       # (N,) - Patient ID mapping
├── layer_names/          # Layer boundary names
└── splits/               # Dataset partitions
    ├── train/           # 70% of data
    ├── val/            # 15% of data
    └── test/           # 15% of data
```

## Conversion Process

### Data Selection
- Uses B-scans 20-80 from each volume
- Excludes peripheral scans (typically containing artifacts/missing data)
- Requires ≥60% valid annotations per B-scan

### Processing Steps

1. **Volume Normalization**
   - Per-volume normalization to [0,1] range
   - Preserves relative intensity relationships

2. **Data Validation**
   - Checks annotation validity
   - Filters out scans with insufficient annotations
   - Tracks patient attribution

3. **Data Organization**
   - Flattened structure for efficient access
   - Random train/val/test splits
   - Patient index tracking for analysis

### Storage Optimization

- **Chunking**: 1 B-scan per chunk
- **Compression**: GZIP
- **Extensible Datasets**: Dynamic size allocation
- **Memory Efficient**: Two-pass processing
  1. Count valid scans
  2. Store data with proper chunking

## Usage

### Dependencies
```bash
pip install h5py numpy scipy tqdm
```

### Running the Conversion
```bash
python mat2hdf5.py
```

### Output Verification
```python
import h5py

# Open dataset
with h5py.File('oct_dataset.h5', 'r') as f:
    # Print structure
    print(list(f.keys()))
    # Check shapes
    print(f['images'].shape)
    print(f['layer_maps'].shape)
```

## Implementation Details

### Key Functions

1. **read_mat_file**: Loads MATLAB files
2. **normalize_volume**: Performs volume-wise normalization
3. **check_annotation_validity**: Validates layer annotations
4. **process_volume**: Main processing pipeline
5. **convert_to_h5**: Orchestrates the conversion process

### Data Split Ratios
- Training: 70%
- Validation: 15%
- Test: 15%

## Notes

- NaN values indicate missing/invalid layer boundaries
- Patient indices enable cross-validation strategies
- Chunked storage optimizes random access during training
- Dataset structure supports efficient data loading and shuffling

## References

1. [Dataset Publication](https://iovs.arvojournals.org/article.aspx?articleid=2127341)
2.