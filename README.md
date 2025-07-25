# SCR-Progression: Retinal Layer Annotation Prediction in OCT B-Scan Images

## Project Overview

This repository contains machine learning models and tools for automated prediction of retinal layer annotations in Optical Coherence Tomography (OCT) B-scan images. The primary focus is on predicting critical retinal layers, specifically the Inner Limiting Membrane (ILM) and Bruch's Membrane (BM), to assist in medical diagnosis of SCR disease in patients.

## Repository Structure

```
SCR-Progression/
├── Image-Segmentation/          # Core segmentation models and training scripts
│   ├── cnn-regression-model.py  # CNN-based regression model
│   ├── train_segformer.py       # SegFormer model training
│   ├── swin_train.ipynb         # Swin Transformer model training
│   ├── evaluation/              # Model evaluation scripts
│   └── models/                  # Saved model artifacts
│
├── Swin-Regression-Model/       # Swin Transformer regression implementations
│   ├── CNN_train.py             # TensorFlow CNN implementation
│   ├── CNN_pytorch.py           # PyTorch CNN implementation
│   ├── model.py                 # Model architectures
│   ├── main.py                  # Training pipeline
│   └── logs/                    # Training logs and outputs
│
├── Lab-Data-Annotations/        # Annotation processing and data operations
│   ├── data_ops.ipynb          # Data manipulation notebooks
│   ├── img_overlay_micron.ipynb # Image overlay utilities
│   └── layer_maps.h5           # Processed layer annotation data
│
├── Annotations-Test-Case/       # Test cases and annotation examples
│   ├── img_lyr_overlay.ipynb   # Layer overlay visualization
│   └── pairs/                  # Image-annotation pairs
│
├── Img-Preprocessing/          # Image preprocessing utilities
│   ├── noise_reduction.py      # Noise reduction algorithms
│   └── output/                 # Processed images
│
├── hdf5-Convert/              # Data format conversion tools
│   ├── mat2hdf5.py           # MATLAB to HDF5 conversion
│   └── crop_image.ipynb      # Image cropping utilities
│
├── e2e/                      # End-to-end pipeline implementations
│   └── read_e2e.ipynb       # E2E processing notebook
│
└── data_processing.ipynb    # Main data processing pipeline
```

## Models Implemented

### 1. CNN Regression Model
- **Frameworks**: PyTorch and transformer library
- **Architecture**: Convolutional layers with regression head (benchmark)
- **Output**: 224 coordinate points for ILM and BM layers
- **Files**: `CNN_train.py`, `CNN_pytorch.py`

### 2. Swin Transformer Model
- **Framework**: Pytorch with Transformers library
- **Architecture**: Vision Transformer with Swin architecture
- **Purpose**: Advanced feature extraction for layer prediction(benchmark)
- **Files**: `swin_train.ipynb`, `swin_model_train.ipynb`

### 3. SegFormer Model
- **Framework**: Hugging Face Transformers
- **Architecture**: Vision transformer
- **File**: `train_segformer.py`

## Datasets

The project works with processed OCT B-scan datasets:
- **Duke Control Dataset**: Annotated b-scan dataset of AMD and Control patients. [Reference](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm)
- **Internal Nemours Dataset**:  Annotated B-scan dataset of SCR and Control patients.
- 
## Getting Started

### Prerequisites
```bash
# Core dependencies
conda install numpy pandas opencv-python
conda install -c conda-forge tensorflow torch torchvision
conda install -c conda-forg transformers scikit-learn matplotlib h5py
```

## Model Performance

- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**:
  -- Mean Absolute Error (MAE)
  -- Dice Score
  -- Precison-Recall
  -- Senstivity

## Future Enhancements

- [ ] Model fine-tuning and hyperparameter tuning to improve performance
- [ ] More advanced Imaage Preprocessing using OpenCV for better results
- [ ] Model ensemble techniques
- [ ] Advanced data augmentation
- [ ] Integration with clinical workflows
- [ ] Performance optimization for deployment

## Contributing

This is an active research project. Contributions and suggestions are welcome for:
- Model architecture improvements
- Data preprocessing enhancements
- Evaluation metrics and visualization tools
- Documentation and code organization
---

*This repository is part of ongoing research in medical image analysis and computer vision for ophthalmology applications at Nemours Childrens Hospital.*
