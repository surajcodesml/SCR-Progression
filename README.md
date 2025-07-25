# SCR-Progression: Retinal Layer Annotation Prediction in OCT B-Scan Images

## Project Overview

This repository contains machine learning models and tools for automated prediction of retinal layer annotations in Optical Coherence Tomography (OCT) B-scan images. The primary focus is on predicting critical retinal layers, specifically the Inner Limiting Membrane (ILM) and Bruch's Membrane (BM), to assist in medical diagnosis and research.

## Key Features

- **Deep Learning Models**: CNN and Swin Transformer-based architectures for layer segmentation
- **Multi-Framework Support**: Both TensorFlow/Keras and PyTorch implementations
- **Data Processing Pipeline**: Comprehensive tools for OCT image preprocessing and annotation handling
- **Evaluation Tools**: Model performance analysis and visualization utilities

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
- **Frameworks**: TensorFlow/Keras and PyTorch
- **Architecture**: Convolutional layers with regression head
- **Output**: 224 coordinate points for ILM and BM layers
- **Files**: `CNN_train.py`, `CNN_pytorch.py`

### 2. Swin Transformer Model
- **Framework**: TensorFlow with Transformers library
- **Architecture**: Vision Transformer with Swin architecture
- **Purpose**: Advanced feature extraction for layer prediction
- **Files**: `swin_train.ipynb`, `swin_model_train.ipynb`

### 3. SegFormer Model
- **Framework**: Hugging Face Transformers
- **Architecture**: Efficient transformer for semantic segmentation
- **File**: `train_segformer.py`

## Datasets

The project works with processed OCT B-scan datasets:
- **Duke Control Dataset**: `Duke_Control_processed.h5`
- **Layer Annotations**: Preprocessed retinal layer maps
- **Image Format**: 224x224 grayscale images
- **Annotations**: ILM and BM layer coordinates

## Getting Started

### Prerequisites
```bash
# Core dependencies
pip install tensorflow torch torchvision
pip install transformers scikit-learn matplotlib h5py
pip install numpy pandas opencv-python
```

### Quick Start
1. **Data Preparation**:
   ```bash
   # Process your OCT data using the conversion tools
   python hdf5-Convert/mat2hdf5.py
   ```

2. **Train CNN Model (PyTorch)**:
   ```bash
   cd Swin-Regression-Model
   python CNN_pytorch.py
   ```

3. **Train CNN Model (TensorFlow)**:
   ```bash
   cd Swin-Regression-Model
   python CNN_train.py
   ```

4. **Evaluate Results**:
   - Check generated plots in the logs directory
   - Review model performance metrics

## Model Performance

- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Validation**: 80/20 train-test split
- **Output Format**: (224, 2) coordinate predictions for ILM and BM layers

## Future Enhancements

- [ ] Multi-layer prediction (beyond ILM and BM)
- [ ] Real-time inference pipeline
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

## License

[Add your license information here]

## Contact

[Add your contact information here]

---

*This repository is part of ongoing research in medical image analysis and computer vision for ophthalmology applications.*
