import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from skimage.transform import resize

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class LayerAnnotationDataset(Dataset):
    def __init__(self, images, layer_maps):
        self.images = images.astype(np.float32)
        self.layer_maps = layer_maps.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        layers = self.layer_maps[idx]
        # Transpose image to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(layers)

# Model definition
class LayerAnnotationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 224 * 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        x = x.view(-1, 224, 2)
        return x

def denormalize_layers(layers, layer_min=0, layer_max=224):
    return layers * (layer_max - layer_min) + layer_min

def plot_layer_annotations(model, images, layer_maps, num_samples=5, save_dir=None, model_name="model"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    for idx in range(num_samples):
        img = images[idx]
        true_layers = denormalize_layers(layer_maps[idx], 0, 224)
        with torch.no_grad():
            inp = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
            pred_layers = model(inp).cpu().numpy()[0]
            pred_layers = denormalize_layers(pred_layers, 0, 224)
        plt.figure(figsize=(8, 5))
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.plot(range(224), true_layers[:, 0], 'g-', label='True ILM')
        plt.plot(range(224), true_layers[:, 1], 'b-', label='True BM')
        plt.plot(range(224), pred_layers[:, 0], 'r--', label='Pred ILM')
        plt.plot(range(224), pred_layers[:, 1], 'm--', label='Pred BM')
        plt.title(f"Sample {idx}: Layer Annotations (Denormalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_dir:
            filename = f"{model_name}_sample{idx}.png"
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        plt.close()

def mae_metric(pred, target)-> float:
    '''
    Mean Absolute Error
    args: 
        pred (Tensor)
        target (Tensor)
    returns:
        Mean Absolute Error as a float
    '''
    return torch.mean(torch.abs(pred - target)).item()

def lines_to_mask(lines, height=224, width=224):
    # lines: (batch, width, 2) - for each x, y1 and y2
    # Returns: (batch, 2, height, width) binary masks for ILM and BM
    batch = lines.shape[0]
    mask = torch.zeros((batch, 2, height, width), device=lines.device)
    for b in range(batch):
        for i in range(2):  # ILM and BM
            y_coords = torch.clamp(lines[b, :, i].round().long(), 0, height-1)
            mask[b, i, y_coords, torch.arange(width)] = 1
    return mask

def dice_coefficient(pred_mask, target_mask, eps=1e-6):
    # pred_mask, target_mask: (batch, 2, H, W)
    intersection = (pred_mask * target_mask).sum(dim=(2,3))
    union = pred_mask.sum(dim=(2,3)) + target_mask.sum(dim=(2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def precision_recall_f1(pred_mask, target_mask, eps=1e-6):
    # pred_mask, target_mask: (batch, 2, H, W)
    tp = (pred_mask * target_mask).sum(dim=(2,3))
    fp = (pred_mask * (1 - target_mask)).sum(dim=(2,3))
    fn = ((1 - pred_mask) * target_mask).sum(dim=(2,3))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return precision.mean().item(), recall.mean().item(), f1.mean().item()

def plot_training_metrics(train_losses, val_losses, val_f1s, val_precisions, val_recalls, n_epochs, save_dir=None):
    
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot Loss vs Epoch
    plt.figure()
    plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    if save_dir:
        filename = f"loss_vs_epoch.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()

    # Plot F1 vs Epoch
    plt.figure()
    plt.plot(range(1, n_epochs+1), val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')
    plt.legend()
    if save_dir:
        filename = f"f1_vs_epoch.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()

    # Plot Precision-Recall Curve (per epoch)
    plt.figure()
    plt.plot(val_recalls, val_precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (per epoch)')
    if save_dir:
        filename = f"precision_recall_curve.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()

def load_nemours_data(hdf5_path, target_size=(224, 224)):
    """
    Load data from Nemours_Jing_RL_Annotated.h5 file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        target_size (tuple): Target size for images (height, width)
    
    Returns:
        tuple: (images, layer_maps) where images have shape (N, H, W, 1) 
               and layer_maps have shape (N, W, 2) for ILM and BM layers
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load images: shape (310, 496, 768)
        images = f['images'][:]
        
        # Load layer annotations - only ILM and BM
        layers = f['layers']
        ilm_annotations = layers['ILM'][:]  # shape (310, 768)
        bm_annotations = layers['BM'][:]    # shape (310, 768)
        
        print(f"Original images shape: {images.shape}")
        print(f"Original ILM annotations shape: {ilm_annotations.shape}")
        print(f"Original BM annotations shape: {bm_annotations.shape}")
        
        # Resize images and scale layer annotations
        resized_images = []
        resized_layer_maps = []
        
        original_height, original_width = images.shape[1], images.shape[2]
        target_height, target_width = target_size
        
        # Calculate scaling factors
        height_scale = target_height / original_height
        width_scale = target_width / original_width
        
        for i in range(images.shape[0]):
            # Resize image
            img = images[i]
            resized_img = resize(img, target_size, preserve_range=True, anti_aliasing=True)
            resized_images.append(resized_img)
            
            # Scale layer annotations
            # Original annotations are y-coordinates for each x-position
            # We need to resample and scale them to match the new dimensions
            
            # Create new x-coordinates for target width
            new_x = np.linspace(0, original_width - 1, target_width)
            original_x = np.arange(original_width)
            
            # Interpolate ILM and BM y-coordinates for new x-coordinates
            ilm_y_resized = np.interp(new_x, original_x, ilm_annotations[i])
            bm_y_resized = np.interp(new_x, original_x, bm_annotations[i])
            
            # Scale y-coordinates to new height
            ilm_y_resized = ilm_y_resized * height_scale
            bm_y_resized = bm_y_resized * height_scale
            
            # Combine ILM and BM into layer_map format (W, 2)
            layer_map = np.column_stack([ilm_y_resized, bm_y_resized])
            resized_layer_maps.append(layer_map)
        
        # Convert to numpy arrays
        images = np.array(resized_images)
        layer_maps = np.array(resized_layer_maps)
        
        # Add channel dimension to images if needed
        if images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
        
        print(f"Resized images shape: {images.shape}")
        print(f"Resized layer_maps shape: {layer_maps.shape}")
        
        return images, layer_maps

def load_duke_data(hdf5_path):
    """
    Load data from Duke_Control_processed.h5 file (existing format).
    
    Args:
        hdf5_path (str): Path to the HDF5 file
    
    Returns:
        tuple: (images, layer_maps) in the original format
    """
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]  # (N, 224, 224)
        layer_maps = f['layer_maps'][:]  # (N, 224, 3) or (N, 224, 2)

    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    layer_maps = layer_maps[:, :, [0, 2]]  # Only ILM and BM
    
    return images, layer_maps

def test_lines_to_mask_visualization(model, images, layer_maps, num_samples=3, save_dir="mask_test"):
    """
    Test the lines_to_mask function by visualizing masks overlaid on images.
    
    Args:
        model: Trained model
        images: Test images
        layer_maps: True layer annotations
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(images))):
            img = images[idx]
            true_layers = layer_maps[idx]
            
            # Get model prediction
            inp = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
            pred_layers = model(inp).cpu().numpy()[0]
            
            # Convert to torch tensors for mask generation
            true_layers_torch = torch.from_numpy(true_layers).unsqueeze(0)
            pred_layers_torch = torch.from_numpy(pred_layers).unsqueeze(0)
            
            # Generate masks
            true_mask = lines_to_mask(true_layers_torch)
            pred_mask = lines_to_mask(pred_layers_torch)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(img[:, :, 0], cmap='gray')
            axes[0, 0].set_title(f'Original Image {idx}')
            axes[0, 0].axis('off')
            
            # True mask overlay
            axes[0, 1].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[0, 1].imshow(true_mask[0, 0].numpy(), cmap='Reds', alpha=0.5)
            axes[0, 1].imshow(true_mask[0, 1].numpy(), cmap='Blues', alpha=0.5)
            axes[0, 1].set_title('True Masks (Red: ILM, Blue: BM)')
            axes[0, 1].axis('off')
            
            # Predicted mask overlay
            axes[0, 2].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            axes[0, 2].imshow(pred_mask[0, 0].numpy(), cmap='Reds', alpha=0.5)
            axes[0, 2].imshow(pred_mask[0, 1].numpy(), cmap='Blues', alpha=0.5)
            axes[0, 2].set_title('Pred Masks (Red: ILM, Blue: BM)')
            axes[0, 2].axis('off')
            
            # Line annotations on image
            axes[1, 0].imshow(img[:, :, 0], cmap='gray')
            axes[1, 0].plot(range(224), true_layers[:, 0], 'g-', label='True ILM', linewidth=2)
            axes[1, 0].plot(range(224), true_layers[:, 1], 'b-', label='True BM', linewidth=2)
            axes[1, 0].plot(range(224), pred_layers[:, 0], 'r--', label='Pred ILM', linewidth=2)
            axes[1, 0].plot(range(224), pred_layers[:, 1], 'm--', label='Pred BM', linewidth=2)
            axes[1, 0].set_title('Line Annotations Comparison')
            axes[1, 0].legend()
            axes[1, 0].axis('off')
            
            # True masks only
            axes[1, 1].imshow(true_mask[0, 0].numpy(), cmap='Reds', alpha=0.8)
            axes[1, 1].imshow(true_mask[0, 1].numpy(), cmap='Blues', alpha=0.8)
            axes[1, 1].set_title('True Masks Only')
            axes[1, 1].axis('off')
            
            # Predicted masks only
            axes[1, 2].imshow(pred_mask[0, 0].numpy(), cmap='Reds', alpha=0.8)
            axes[1, 2].imshow(pred_mask[0, 1].numpy(), cmap='Blues', alpha=0.8)
            axes[1, 2].set_title('Pred Masks Only')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'mask_test_sample_{idx}.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Sample {idx}: True mask ILM pixels: {true_mask[0, 0].sum().item()}, "
                  f"BM pixels: {true_mask[0, 1].sum().item()}")
            print(f"Sample {idx}: Pred mask ILM pixels: {pred_mask[0, 0].sum().item()}, "
                  f"BM pixels: {pred_mask[0, 1].sum().item()}")

if __name__ == "__main__":
    # Configuration
    use_nemours_data = False  # Set to False to use Duke data
    
    # Load dataset based on configuration
    if use_nemours_data:
        # Load Nemours data
        nemours_path = '/home/suraj/Git/SCR-Progression/Nemours_Jing_RL_Annotated.h5'
        if os.path.exists(nemours_path):
            print("Loading Nemours dataset...")
            images, layer_maps = load_nemours_data(nemours_path)
        else:
            print(f"Nemours dataset not found at {nemours_path}")
            print("Falling back to Duke dataset (if available)...")
            use_nemours_data = False
    
    if not use_nemours_data:
        # Load Duke data (original format)
        duke_path = '/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control_normalized_corrected.h5'
        if os.path.exists(duke_path):
            print("Loading Duke dataset...")
            images, layer_maps = load_duke_data(duke_path)
        else:
            print(f"Duke dataset not found at {duke_path}")
            print("Please ensure at least one dataset is available.")
            exit(1)
    
    print(f"Dataset loaded: {len(images)} samples")
    print(f"Image shape: {images.shape}")
    print(f"Layer maps shape: {layer_maps.shape}")

    
    # Shuffle the dataset
    indices = np.random.permutation(len(images))
    images = images[indices]
    layer_maps = layer_maps[indices]
    
    # Limit to 1000 images for training
    #images = images[:1000]
    #layer_maps = layer_maps[:1000]

    # Split data
    dataset = LayerAnnotationDataset(images, layer_maps)
    n_total = len(dataset)
    n_test = int(0.2 * n_total)
    n_train = n_total - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # Model, loss, optimizer
    model = LayerAnnotationCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # Tracking lists
    train_losses = []
    val_losses = []
    val_f1s = []
    val_precisions = []
    val_recalls = []

    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train
        train_losses.append(train_loss) # Track training loss

        # Validation loop
        model.eval()
        val_loss = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        val_loss /= n_test
        val_losses.append(val_loss) # Track validation loss

        # Concatenate all outputs/targets for metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # As these metrics are used to measure performance for masks, we are converting lines to masks
        # and then calculating the metrics
        pred_mask = lines_to_mask(all_outputs)
        target_mask = lines_to_mask(all_targets)
        dice = dice_coefficient(pred_mask, target_mask)
        precision, recall, f1 = precision_recall_f1(pred_mask, target_mask)
        val_f1s.append(f1)
        val_precisions.append(precision)
        val_recalls.append(recall)

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Dice: {dice:.4f}")


    
    # Save model
    torch.save(model.state_dict(), "CNN_regression_model.pth")

    # Visualize predictions
    X_test = np.array([test_set[i][0].numpy().transpose(1, 2, 0) for i in range(min(5, n_test))])
    y_test = np.array([test_set[i][1].numpy() for i in range(min(5, n_test))])
    plot_layer_annotations(model, X_test, y_test, num_samples=5, save_dir="pytorch_logs", model_name="CNN_regression_model")
    plot_training_metrics(train_losses, val_losses, val_f1s, val_precisions, val_recalls, n_epochs, save_dir="pytorch_logs")
    
    # Test lines_to_mask function visualization
    print("\nTesting lines_to_mask function...")
    test_lines_to_mask_visualization(model, X_test, y_test, num_samples=3, save_dir="pytorch_logs/mask_tests")
    
    print("Training complete. Model saved as 'CNN_regression_model.pth'.")
    print("Check 'pytorch_logs/mask_tests/' for mask visualization results.")