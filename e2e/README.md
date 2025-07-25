# E2E to HDF5 Converter Script: e2e_to_hdf5_converter.py

## What does it do?

This script (`e2e_to_hdf5_converter.py`) converts Heidelberg Spectralis OCT `.e2e` files into a single HDF5 file. It extracts raw B-scan images and retinal layer annotations from each `.e2e` file and stores them in a structured, compressed HDF5 dataset. The script can process multiple `.e2e` files in a folder, appending new data while skipping files that have already been processed.

---

## How does it work?

1. **Configuration**:  
   Set the input folder containing `.e2e` files and the output HDF5 file path in the `main()` function.

2. **Batch Processing**:  
   The script scans the input folder for all `.e2e` files.

3. **Duplicate Checking**:  
   Before processing, it checks the HDF5 file for filenames already stored and skips any `.e2e` files that have been processed previously.

4. **Data Extraction**:  
   For each new `.e2e` file, it extracts:
   - B-scan images (raw data)
   - Supported retinal layer annotations (ILM, BM, ELM, PR1, RPE, ONL, OPL, INL, IPL, GCL, NFL)
   - The source filename

5. **Appending to HDF5**:  
   The extracted data is appended to the HDF5 file under the following structure:



# HDF5 Dataset Structure: Nemours_Jing_RL_Annotated.h5

This HDF5 file stores retinal image data and corresponding layer annotations, converted from .e2e files. The structure is organized for easy access and efficient storage.

## Structure Overview

- **images**  
  - Dataset containing all B-scan images.  
  - Shape: `(N, H, W)` where `N` is the batch size, `H` is height, and `W` is width.  
  - Each entry is a single OCT B-scan.

- **layers**  
  - Group containing datasets for each annotated retinal layer.  
  - Each key under `layers` is a layer name (e.g., `ILM`, `BM`, `PR1`, etc.).
  - Each layer dataset contains the annotation coordinates for that layer, for each image.
    - Shape: `(N, W)` where `N` is the batch size and `W` is x coordinate storing the cooresponding Y coordinate value
    - Example: `layers['ILM'][i]` gives the ILM annotation for image `i`.

- **names**  
  - Dataset containing the source filename for each image.
  - Shape: `(N,)` (one name per image).
  - Example: `names[i]` gives the filename (without extension) from which image `i` and its annotations were extracted.

## Example Access Pattern

```python
import h5py

with h5py.File('Nemours_Jing_RL_Annotated.h5', 'r') as f:
    images = f['images']              # All images
    layers = f['layers']              # Layer group
    ilm_coords = layers['ILM'][0]     # ILM annotation for image 0
    bm_coords = layers['BM'][0]       # BM annotation for image 0
    filename = f['names'][0]          # Source file for image 0
```

## Notes

- Layer datasets may have different shapes depending on the annotation format.
- The `names` dataset allows you to trace each image and its annotations back to the original .e2e file.
- Supported layers include: `ILM`, `BM`, `ELM`, `PR1`, `RPE`, `ONL`, `OPL`, `INL`, `IPL`, `GCL`, `NFL`
