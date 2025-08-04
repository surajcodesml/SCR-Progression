# SCR-Progression: Retinal Layer Annotation Prediction in OCT B-Scan Images

## Project Overview

This repository contains models and tools for retinal layer prediction in Optical Coherence Tomography (OCT) B-scan images. The primary focus is on predicting the Inner Limiting Membrane (ILM) and Bruch's Membrane (BM), for now, and try to improve model performance but hyper-parameter tuning and introducing new model architectures.

## Repository Structure


```
SCR-Progression/
├── CNN-Model/               # CNN regression models (PyTorch, TensorFlow)
├── SegFormer-Model/         # SegFormer transformer model
├── Swin-Model/              # Swin Transformer model
├── Swin-Regression-Model/   # Swin regression experiments
├── Image-Segmentation/      # Core segmentation models & evaluation
├── Annotations-Test-Case/   # Test cases & annotation examples
├── e2e/                     # E2E dataset processing & conversion
├── hdf5-Convert/            # Data format conversion tools (.mat/.e2e to .h5)
├── Img-Preprocessing/       # Image preprocessing utilities
├── data_processing.ipynb    # Main data processing pipeline
├── bscan_ILM.6.png          # Sample B-scan image
└── README.md                # Project documentation
```

## Models Implemented

implementing **3 different models** for retinal layer annotation prediction:

### 1. CNN Regression Model
- **Architecture**: Convolutional Neural Network with regression head
- **Purpose**: Baseline model for layer coordinate prediction
- **Output**: Coordinate points for ILM, BM, and PR1 layers
- **Files**: `CNN-Model/CNN_pytorch.py`, `CNN-Model/CNN_tensorflow.py`

### 2. SegFormer Model
- **Architecture**: Vision Transformer optimized for segmentation
- **Purpose**: Transformer-based approach for layer detection
- **File**: `SegFormer-Model/train_segformer.py`

### 3. Swin Transformer Model
- **Architecture**: Shifted Window Vision Transformer
- **Purpose**: Advanced feature extraction using hierarchical vision transformer
- **Files**: `Swin-Model/swin_model_train.ipynb`, `Swin-Regression-Model/`

## Datasets

The project works with **2 different datasets** of processed OCT B-scan images:

### 1. Duke Control Dataset
- **Source**: Duke University AMD dataset
- **Format**: MATLAB (.mat) files
- **Content**: Annotated B-scan dataset of AMD and Control patients
- **Reference**: [Duke AMD Dataset](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm)
- **Conversion**: Use `hdf5-Convert/mat2hdf5.py` to convert .mat files to HDF5 format

### 2. Internal Nemours Dataset  
- **Source**: Nemours Children's Hospital
- **Format**: E2E files (Heidelberg OCT format)
- **Content**: Annotated B-scan dataset of SCR and Control patients
- **Conversion**: Use `e2e/e2e_to_hdf5_converter.py` to convert .e2e files to HDF5 format

## Getting Started  
```bash
# Core dependencies
conda install numpy pandas opencv-python matplotlib h5py
conda install -c conda-forge tensorflow torch torchvision
conda install -c conda-forge transformers scikit-learn

# For E2E file processing
pip install eyepy

# For experiment tracking
pip install wandb
```
---

*This repository is part of ongoing research in medical image analysis and computer vision for ophthalmology applications at Nemours Children's Hospital.*
