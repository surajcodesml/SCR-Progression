import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
import json
from pathlib import Path
import os

# Configure logging
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs/runs") / run_id
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

config = {
    "model_name": "nvidia/mit-b0",
    "image_size": 512,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "epochs": 10,
    "num_train_samples": 1000,
    "num_test_samples": 50,
    "layer_indices": [0, 2],  # ILM and Bruch's membrane
    "output_width": 512,  # Width of coordinate output
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Save config
with open(log_dir / "config.json", "w") as f:
    json.dump(config, f, indent=4)

logger.info(f"Starting training run: {run_id}")
logger.info(f"Configuration: {config}")

class OCTLayerRegressionModel(nn.Module):
    """
    Custom model for OCT layer coordinate regression using SegFormer backbone
    """
    def __init__(self, backbone_name="nvidia/mit-b0", output_width=512, num_layers=2):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(backbone_name)
        self.output_width = output_width
        self.num_layers = num_layers
        
        # Get feature dimensions from backbone
        # SegFormer backbone outputs features at different scales
        # We'll use the last layer features
        backbone_channels = self.backbone.config.hidden_sizes[-1]  # e.g., 256 for mit-b0
        
        # Regression head to predict y-coordinates
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, output_width)),  # Pool to (1, output_width)
            nn.Flatten(start_dim=2),  # Shape: (batch, channels, output_width)
            nn.Conv1d(backbone_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_layers, kernel_size=1),  # Output: (batch, num_layers, output_width)
        )
        
    def forward(self, x):
        # Get features from backbone
        outputs = self.backbone(x)
        # Use the last hidden state
        features = outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        
        # Reshape to spatial format for processing
        # The SegFormer output is flattened, we need to reshape it back
        if len(features.shape) == 3:
            batch_size, seq_len, hidden_size = features.shape
            # Calculate spatial dimensions (assuming square feature maps)
            spatial_size = int(seq_len ** 0.5)
            features = features.transpose(1, 2).reshape(batch_size, hidden_size, spatial_size, spatial_size)
        elif len(features.shape) == 4:
            # Already in spatial format
            batch_size, hidden_size, height, width = features.shape
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        # Apply regression head
        coords = self.regression_head(features)  # Shape: (batch, num_layers, output_width)
        
        return coords

class OCTDataset(Dataset):
    def __init__(self, images, annotations, layer_indices=[0, 2], img_size=512, output_width=512):
        self.images = images
        self.annotations = annotations
        self.layer_indices = layer_indices
        self.img_size = img_size
        self.output_width = output_width
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and normalize
        image = self.images[idx]  # Original image shape from HDF5
        original_height, original_width = image.shape
        
        # Get layer annotations for selected layers (shape: [1000, num_layers])
        coords = self.annotations[idx][:, self.layer_indices]  # Shape: [1000, 2]
        
        # Handle NaN values by replacing them with interpolated values
        for layer_idx in range(coords.shape[1]):
            layer_coords = coords[:, layer_idx]
            if np.isnan(layer_coords).any():
                # Find valid coordinates
                valid_mask = ~np.isnan(layer_coords)
                if valid_mask.sum() > 1:  # Need at least 2 points for interpolation
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = layer_coords[valid_mask]
                    # Interpolate missing values
                    coords[:, layer_idx] = np.interp(
                        np.arange(len(layer_coords)), 
                        valid_indices, 
                        valid_values
                    )
                else:
                    # If too few valid points, fill with mean of image height
                    coords[:, layer_idx] = original_height // 2
        
        # Resize image to match SegFormer input size
        image = torch.from_numpy(image).float()
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0).unsqueeze(0), 
            size=(self.img_size, self.img_size)
        ).squeeze()
        
        # Convert to 3 channels by repeating the grayscale channel
        image = image.unsqueeze(0).repeat(3, 1, 1)  # Shape: [3, 512, 512]
        
        # Resize coordinates to match output width
        if coords.shape[0] != self.output_width:
            # Interpolate coordinates to match output width
            coords_tensor = torch.from_numpy(coords).float()
            coords_tensor = coords_tensor.permute(1, 0).unsqueeze(0)  # Shape: [1, 2, 1000]
            coords_tensor = torch.nn.functional.interpolate(
                coords_tensor, 
                size=self.output_width, 
                mode='linear', 
                align_corners=True
            )
            coords_tensor = coords_tensor.squeeze(0).permute(1, 0)  # Shape: [512, 2]
            coords = coords_tensor.numpy()
        
        # Scale y-coordinates based on image resizing
        # Scale from original image height to resized image height
        y_scale_factor = self.img_size / original_height
        coords = coords * y_scale_factor
        
        # Normalize coordinates to prevent overflow
        # Coordinates should be in range [0, img_size]
        coords = np.clip(coords, 0, self.img_size)
        
        # Transpose to match expected output format: [num_layers, output_width]
        coords = coords.T  # Shape: [2, 512]
        
        return image, torch.from_numpy(coords).float()

def load_data(path, num_train, num_test) -> tuple:
    """
    Load and split data from H5 file with sorted indices
    Args:
        path (str): Path to the H5 file.
        num_train (int): Number of training samples.
        num_test (int): Number of testing samples.
    Returns:
        tuple: Training images, training layer maps, testing images, testing layer maps.
    """
    with h5py.File(path, 'r') as f:
        # Load subset of data
        total_indices = np.random.permutation(len(f['images']))
        selected_indices = total_indices[:num_train + num_test]
        
        # Sort indices for H5py compatibility
        train_indices = np.sort(selected_indices[:num_train])
        test_indices = np.sort(selected_indices[num_train:num_train + num_test])
        
        # Load data using sorted indices
        train_images = f['images'][train_indices]
        train_layers = f['layer_maps'][train_indices]
        test_images = f['images'][test_indices]
        test_layers = f['layer_maps'][test_indices]
        
    return train_images, train_layers, test_images, test_layers

def train_epoch(model, dataloader, criterion, optimizer, device) -> float:
    """
    Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on.
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, coords) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        coords = coords.to(device)
        
        # Debug first batch
        if batch_idx == 0:
            print(f"Input coords stats: min={coords.min():.3f}, max={coords.max():.3f}, mean={coords.mean():.3f}")
            print(f"Input coords shape: {coords.shape}")
        
        optimizer.zero_grad()
        outputs = model(images)  # Shape: (batch, num_layers, output_width)
        
        # Debug first batch
        if batch_idx == 0:
            print(f"Output coords stats: min={outputs.min():.3f}, max={outputs.max():.3f}, mean={outputs.mean():.3f}")
            print(f"Output coords shape: {outputs.shape}")
        
        loss = criterion(outputs, coords)
        
        if batch_idx == 0:
            print(f"Loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device) -> tuple:
    """ Evaluate the model on the validation/test set.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation/test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on.
    Returns:
        tuple: Average loss and Mean Absolute Error (MAE) for the evaluation.
    """
    model.eval()
    running_loss = 0.0
    mae = 0.0
    
    with torch.no_grad():
        for images, coords in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            coords = coords.to(device)
            
            outputs = model(images)  # Shape: (batch, num_layers, output_width)
            loss = criterion(outputs, coords)
            running_loss += loss.item()
            
            mae += torch.abs(outputs - coords).mean().item()
    
    return running_loss / len(dataloader), mae / len(dataloader)

def main():
    device = torch.device(config["device"])
    
    # Load data
    train_images, train_layers, test_images, test_layers = load_data(
        "/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control.h5",
        config["num_train_samples"],
        config["num_test_samples"]
    )
    
    # Create datasets
    train_dataset = OCTDataset(
        train_images, 
        train_layers, 
        layer_indices=config["layer_indices"],
        img_size=config["image_size"],
        output_width=config["output_width"]
    )
    test_dataset = OCTDataset(
        test_images, 
        test_layers, 
        layer_indices=config["layer_indices"],
        img_size=config["image_size"],
        output_width=config["output_width"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    
    # Initialize model
    model = OCTLayerRegressionModel(
        backbone_name=config["model_name"],
        output_width=config["output_width"],
        num_layers=len(config["layer_indices"])
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    best_test_loss = float('inf')
    training_history = []
    
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_mae = evaluate(model, test_loader, criterion, device)
        
        # Log metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_mae": test_mae
        }
        training_history.append(epoch_metrics)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f"models/checkpoints/best_model_{run_id}.pt")
            logger.info(f"New best model saved with test loss: {best_test_loss:.4f}")
            
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, MAE = {test_mae:.4f}")
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, MAE = {test_mae:.4f}")
    
    # Save final metrics
    final_metrics = {
        "best_test_loss": float(best_test_loss),
        "final_train_loss": float(train_loss),
        "final_test_mae": float(test_mae),
        "training_history": training_history
    }
    
    with open(log_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
    
    # Save training history as CSV for easy plotting
    import pandas as pd
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(log_dir / "training_history.csv", index=False)
    
    logger.info(f"Training completed. Best test loss: {best_test_loss:.4f}")
    logger.info(f"Results saved to: {log_dir}")

if __name__ == "__main__":
    main()