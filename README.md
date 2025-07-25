# SCR-Progression: Retinal Layer Annotation Prediction in OCT B-Scan Images

## Project Overview

This repository contains machine learning models and tools for automated prediction of retinal layer annotations in Optical Coherence Tomography (OCT) B-scan images. The primary focus is on predicting critical retinal layers, specifically the Inner Limiting Membrane (ILM) and Bruch's Membrane (BM), to assist in medical diagnosis of SCR disease in patients.

This code is part of research work conducted at Nemours Children's Hospital.

## Repository Structure

```
SCR-Progression/
├── CNN-Model/                   # CNN-based regression model implementations
│   ├── CNN_tensorflow.py        # TensorFlow CNN implementation
│   ├── CNN_pytorch.py          # PyTorch CNN implementation
│   └── cnn_data_ops.ipynb      # CNN data operations notebook
│
├── SegFormer-Model/            # SegFormer transformer model
│   └── train_segformer.py      # SegFormer model training script
│
├── Swin-Model/                 # Swin Transformer model implementations
│   ├── swin_model_train.ipynb  # Swin model training notebook
│   └── swin_model1.ipynb       # Swin model experiments
│
├── Swin-Regression-Model/      # Swin regression model experiments
│   ├── data_ops.ipynb          # Data operations for Swin model
│   ├── run_cnn_train_.slurm    # SLURM job script for training
│   └── logs/                   # Training logs and outputs
│
├── Image-Segmentation/         # Core segmentation models and training
│   ├── hdf5_read.ipynb         # HDF5 data reading utilities
│   ├── evaluation/             # Model evaluation scripts and results
│   ├── logs/                   # Training logs and metrics
│   ├── models/                 # Saved model artifacts and checkpoints
│   └── wandb/                  # Weights & Biases experiment tracking
│
├── Annotations-Test-Case/      # Test cases and annotation examples
│   ├── img_lyr_overlay.ipynb   # Layer overlay visualization
│   └── 3_R_4_1_Segm.json      # Sample annotation file
│
├── e2e/                       # E2E dataset processing
│   ├── e2e_to_hdf5_converter.py # E2E to HDF5 conversion using eyepy
│   ├── read_e2e.ipynb         # E2E data reading utilities
│   ├── data_ops.ipynb         # E2E data operations
│   └── README.md              # E2E processing documentation
│
├── hdf5-Convert/              # Data format conversion tools
│   ├── mat2hdf5.py             # MATLAB (.mat) to HDF5 conversion script
│   ├── crop_image.ipynb        # Image cropping utilities
│   └── README.md               # Conversion tools documentation
│
├── Img-Preprocessing/          # Image preprocessing utilities
│   └── noise_reduction.py      # Noise reduction algorithms
│
├── data_processing.ipynb       # Main data processing pipeline
├── bscan_ILM.6.png            # Sample B-scan image with ILM annotation
└── README.md                  # This file
```

## Models Implemented

This project implements **3 different models** for retinal layer annotation prediction:

### 1. CNN Regression Model
- **Frameworks**: PyTorch and TensorFlow
- **Architecture**: Convolutional Neural Network with regression head
- **Purpose**: Baseline model for layer coordinate prediction
- **Output**: Coordinate points for ILM, BM, and PR1 layers
- **Files**: `CNN-Model/CNN_pytorch.py`, `CNN-Model/CNN_tensorflow.py`

### 2. SegFormer Model
- **Framework**: Hugging Face Transformers
- **Architecture**: Vision Transformer optimized for segmentation
- **Purpose**: Transformer-based approach for layer detection
- **File**: `SegFormer-Model/train_segformer.py`

### 3. Swin Transformer Model
- **Framework**: PyTorch with Transformers library
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
- **Conversion**: Use `e2e/e2e_to_hdf5_converter.py` with eyepy library for E2E to HDF5 conversion

## Data Preprocessing Using OpenCV
- **Folder**: Img-Preprocessing

### Noise Reduction  
- In Progress...

### Edge Detection  
- In Progress...

## Data Conversion Tools

### MATLAB to HDF5 Conversion
- **Script**: `hdf5-Convert/mat2hdf5.py`
- **Purpose**: Converts Duke dataset from .mat format to HDF5
- **Usage**: Processes MATLAB annotation files and OCT images

### E2E to HDF5 Conversion  
- **Script**: `e2e/e2e_to_hdf5_converter.py`
- **Library**: Uses eyepy for E2E file processing
- **Purpose**: Converts Heidelberg E2E files to standardized HDF5 format
- **Features**: Extracts B-scan images and layer annotations 
## Getting Started

### Prerequisites
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

### Data Processing Workflow
1. **Convert raw data to HDF5**:
   - For .mat files: `python hdf5-Convert/mat2hdf5.py`
   - For .e2e files: `python e2e/e2e_to_hdf5_converter.py`

2. **Data exploration**: Use `data_processing.ipynb` for initial data analysis

3. **Model training**: Choose from CNN, SegFormer, or Swin models based on requirements

## Model Performance

- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Dice Score
  - Precision-Recall
  - Sensitivity
- **Tracking**: Experiments tracked using Weights & Biases (wandb)

## Project Features

- **Multi-model approach**: Comparison of CNN, SegFormer, and Swin Transformer architectures
- **Dual dataset support**: Handles both Duke (.mat) and Nemours (.e2e) datasets  
- **Automated conversion**: Scripts for converting different data formats to standardized HDF5
- **Comprehensive evaluation**: Multiple metrics and visualization tools
- **Experiment tracking**: Integration with wandb for monitoring training progress

## Future Enhancements

- [ ] Model fine-tuning and hyperparameter optimization
- [ ] Advanced image preprocessing using OpenCV for enhanced results
- [ ] Model ensemble techniques for improved accuracy
- [ ] Advanced data augmentation strategies
- [ ] Integration with clinical workflows
- [ ] Performance optimization for real-time deployment
- [ ] Cross-dataset validation and generalization studies

## Contributing

This is an active research project at Nemours Children's Hospital. Contributions and suggestions are welcome for:
- Model architecture improvements
- Data preprocessing enhancements  
- Evaluation metrics and visualization tools
- Documentation and code organization

---

*This repository is part of ongoing research in medical image analysis and computer vision for ophthalmology applications at Nemours Children's Hospital.*
