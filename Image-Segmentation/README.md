# OCT Layer Regression with SegFormer

## Project Overview  
This repository implements a deep-learning pipeline to predict retinal layer boundaries—specifically the Inner Limiting Membrane (ILM) and Bruch’s Membrane (BM)—from optical coherence tomography (OCT) B-scan images. Unlike traditional semantic segmentation, we perform **coordinate regression**: for each horizontal position (x-pixel) in the scan, the model outputs the vertical coordinate (y-pixel) of each target layer.

### Why Coordinate Regression?  
- Precise boundary localization is more directly expressed as a sequence of y-coordinates than as a pixel-wise mask.  
- Downstream clinical metrics (e.g., layer thickness maps) consume coordinate arrays.

---

## Repository Structure  
```text
Image-Segmentation/
├── train_segformer.py    # Main training & model definition
├── logs/                 # Training run logs and configs
└── README.md             # This documentation
```

---

## Data Format  
Data are stored in an HDF5 file with two primary datasets:  
1. **images**: OCT B-scan images, shape `(N, H, W)`, grayscale.  
2. **layer_maps**: annotated layer coordinates, shape `(N, W_orig, L)`, where:
   - `L` ≥ 3 (number of annotated layers per column)
   - `W_orig` ≈ 1000

We select only the first and last layers (`layer_indices = [0, 2]`) and downsample/interpolate them to `output_width = 512`.

---

## Model Architecture  

```python
class OCTLayerRegressionModel(nn.Module):
    def __init__(self, backbone_name="nvidia/mit-b0", output_width=512, num_layers=2):
        super().__init__()
        # 1) SegFormer backbone for feature extraction
        self.backbone = SegformerModel.from_pretrained(backbone_name)
        # 2) Regression head: projects features → (B, num_layers, output_width)
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, output_width)),
            nn.Flatten(start_dim=2),
            nn.Conv1d(backbone_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_layers, kernel_size=1),
        )

    def forward(self, x):
        # Extract token embeddings
        outputs = self.backbone(x)
        feats = outputs.last_hidden_state  # (B, seq_len, hidden_size)
        # Reshape to spatial map: (B, C, H_feat, W_feat)
        # Apply regression head → (B, num_layers, output_width)
        coords = self.reg_head(feats_reshaped)
        return coords
```

- **Backbone**: Pretrained SegFormer (`mit-b0`) provides rich multi-scale features.  
- **Regression head**: Pools features to a 1×`output_width` map, then uses 1D convolutions to predict y-coordinates for each layer.

---

## Dataset & Preprocessing  

```python
class OCTDataset(Dataset):
    def __getitem__(self, idx):
        # Load and normalize image
        image = images[idx]                  # (H, W)
        coords = annotations[idx][:, layer_indices]  # (W_orig, 2)
        
        # 1) NaN handling: linear interpolation or fill midpoint
        # 2) Resize image → (512, 512), repeat channels → (3, 512, 512)
        # 3) Interpolate coords → output_width (512)
        # 4) Scale y-values by (512/H) and clip to [0, 512]
        
        return image_tensor, coords_tensor    # (3,512,512), (2,512)
```

- **NaN handling**: If ≥2 valid points, linearly interpolate; otherwise fill with mid-height.  
- **Coordinate resizing**: Uses `torch.nn.functional.interpolate` for smooth sampling.  
- **Clipping & scaling**: Ensures y-values remain in valid pixel range.

---

## Training Pipeline  

1. **Configuration**  
   ```json
   {
     "model_name": "nvidia/mit-b0",
     "image_size": 512,
     "batch_size": 8,
     "learning_rate": 1e-4,
     "epochs": 10,
     "layer_indices": [0, 2],
     "output_width": 512
   }
   ```
2. **Data Loading**  
   - Split indices into train/test (e.g., 1000/50 samples).  
   - Wrap in PyTorch `DataLoader` with shuffling.
3. **Loss & Optimization**  
   - **Loss**: `nn.MSELoss()` between predicted and true coordinates.  
   - **Optimizer**: AdamW (`lr=1e-4`).
4. **Logging**  
   - Uses Python’s `logging` module.  
   - Outputs per-epoch loss to `logs/runs/<timestamp>/training.log`.  
   - Saves config to `config.json`.

---

## How to Run  

```bash
pip install torch torchvision transformers sklearn h5py tqdm

python train_segformer.py \
  --data-path /path/to/data.h5 \
  --num-train 1000 \
  --num-test 50
```

---

## Future Directions  
- Extend to predict additional retinal layers (e.g., RPE, ONL).  
- Incorporate data augmentation (elastic, intensity).  
- Add validation loop with early stopping & LR scheduling.  
- Introduce uncertainty estimation (e.g., MC dropout).  
- Build inference script and visualization tools.

---

*This project adapts a semantic-segmentation backbone for precise coordinate regression of retinal layer boundaries in