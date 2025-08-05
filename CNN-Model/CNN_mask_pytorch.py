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
    def __init__(self, images, masks):
        self.images = images.astype(np.float32)
        self.masks = masks.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        # Transpose image to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(mask)

# Model definition for segmentation
class LayerSegmentationCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 28x28
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # 224x224
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def generate_annotation_masks(images, layers, height=None, width=None):
    """
    Generate segmentation masks from ILM, PR1, and BM layer annotations.
    Class 0: Background
    Class 1: ILM to PR1 region
    Class 2: PR1 to BM region
    """
    if height is None:
        height = images.shape[1]
    if width is None:
        width = images.shape[2]

    batch_size = images.shape[0]
    masks = np.zeros((batch_size, height, width), dtype=np.uint8)

    for b in range(batch_size):
        ilm_line = layers['ILM'][b]
        pr1_line = layers['PR1'][b]
        bm_line = layers['BM'][b]

        for x in range(width):
            ilm_y = ilm_line[x]
            pr1_y = pr1_line[x]
            bm_y = bm_line[x]

            if np.isnan(ilm_y) or np.isnan(pr1_y) or np.isnan(bm_y):
                continue

            # Convert to integer pixel coordinates and clip to image bounds
            ilm_y = int(np.clip(round(ilm_y), 0, height-1))
            pr1_y = int(np.clip(round(pr1_y), 0, height-1))
            bm_y = int(np.clip(round(bm_y), 0, height-1))

            # Sort to ensure correct order (top < middle < bottom)
            top, mid, bottom = sorted([ilm_y, pr1_y, bm_y])

            # ILM to PR1 region (Class 1)
            if top < mid:
                masks[b, top:mid, x] = 1
            # PR1 to BM region (Class 2)
            if mid < bottom:
                masks[b, mid:bottom, x] = 2
            # Everything else remains background (Class 0)

    return masks



def load_nemours_data_with_masks(hdf5_path, target_size=(224, 224)):
    """
    Load data from Nemours_Jing_RL_Annotated.h5 file and generate segmentation masks.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        target_size (tuple): Target size for images and masks (height, width)
    
    Returns:
        tuple: (images, masks) where images have shape (N, H, W, 1) 
               and masks have shape (N, H, W) with classes 0,1,2
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load images and layers at original resolution
        images_orig = f['images'][:]  # Shape: (310, 496, 768)
        layers = f['layers']
        
        # Extract ILM, PR1, BM layers
        layer_data = {
            'ILM': layers['ILM'][:],  # Shape: (310, 768)
            'PR1': layers['PR1'][:],  # Shape: (310, 768)
            'BM': layers['BM'][:]     # Shape: (310, 768)
        }
    
    print(f"Original image shape: {images_orig.shape}")
    print(f"Generating masks at original resolution...")
    
    # Generate masks at original resolution (496x768)
    masks_orig = generate_annotation_masks(images_orig, layer_data)
    print(f"Generated masks shape: {masks_orig.shape}")
    print(f"Mask classes: {np.unique(masks_orig)}")
    
    # Resize images and masks to target size
    resized_images = []
    resized_masks = []
    
    target_height, target_width = target_size
    
    for i in range(images_orig.shape[0]):
        # Resize image
        img = images_orig[i]
        resized_img = resize(img, target_size, preserve_range=True, anti_aliasing=True)
        
        # Normalize image to [0, 1] range
        if resized_img.max() > 1.0:
            resized_img = resized_img / 255.0
        
        resized_images.append(resized_img)
        
        # Resize mask using nearest neighbor to preserve class labels
        mask = masks_orig[i]
        resized_mask = resize(mask, target_size, preserve_range=True, 
                            anti_aliasing=False, order=0)  # order=0 for nearest neighbor
        
        # Ensure mask values are integers
        resized_mask = resized_mask.astype(np.uint8)
        resized_masks.append(resized_mask)
    
    # Convert to numpy arrays
    images = np.array(resized_images)
    masks = np.array(resized_masks)
    
    # Add channel dimension to images if needed
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    
    print(f"Final resized image shape: {images.shape}")
    print(f"Final resized mask shape: {masks.shape}")
    print(f"Final mask classes: {np.unique(masks)}")
    
    return images, masks


def load_duke_data_with_masks(hdf5_path, target_size=(224, 224)):
    """
    Load data from Duke dataset HDF5 file and generate segmentation masks.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        target_size (tuple): Target size for images and masks (height, width)
    
    Returns:
        tuple: (images, masks) where images have shape (N, H, W, 1) 
               and masks have shape (N, H, W) with classes 0,1,2
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load images and layer maps
        images_orig = f['images'][:]  # (N, 224, 224) - already at target size
        layer_maps = f['layer_maps'][:]  # (N, 224, 3)
        layer_names = [name.decode() for name in f['layer_names'][:]]
    
    print(f"Duke original image shape: {images_orig.shape}")
    print(f"Duke layer names: {layer_names}")
    
    # Find indices for ILM, PR1, and BM
    ilm_idx = layer_names.index('ILM') if 'ILM' in layer_names else 0
    pr1_idx = next((i for i, name in enumerate(layer_names) if 'PR1' in name), 1)
    bm_idx = next((i for i, name in enumerate(layer_names) if 'Bruch' in name or 'BM' in name), 2)
    
    print(f"Duke layer indices - ILM: {ilm_idx}, PR1: {pr1_idx}, BM: {bm_idx}")
    
    # Extract the three layers we need
    layer_data = {
        'ILM': layer_maps[:, :, ilm_idx],
        'PR1': layer_maps[:, :, pr1_idx], 
        'BM': layer_maps[:, :, bm_idx]
    }
    
    # Normalize images to [0, 1] if needed
    if images_orig.max() > 1.0:
        images_orig = images_orig.astype(np.float32) / 255.0
    
    # Generate masks at original resolution
    print("Generating masks for Duke dataset...")
    masks_orig = generate_annotation_masks(images_orig, layer_data)
    print(f"Generated Duke masks shape: {masks_orig.shape}")
    print(f"Duke mask classes: {np.unique(masks_orig)}")
    
    # Resize if target size is different from original
    if images_orig.shape[1:] != target_size:
        resized_images = []
        resized_masks = []
        
        for i in range(images_orig.shape[0]):
            # Resize image
            img = images_orig[i]
            resized_img = resize(img, target_size, preserve_range=True, anti_aliasing=True)
            resized_images.append(resized_img)
            
            # Resize mask using nearest neighbor to preserve class labels
            mask = masks_orig[i]
            resized_mask = resize(mask, target_size, preserve_range=True, 
                                anti_aliasing=False, order=0)  # order=0 for nearest neighbor
            resized_mask = resized_mask.astype(np.uint8)
            resized_masks.append(resized_mask)
        
        images = np.array(resized_images)
        masks = np.array(resized_masks)
    else:
        images = images_orig
        masks = masks_orig
    
    # Add channel dimension to images if needed
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    
    print(f"Final Duke image shape: {images.shape}")
    print(f"Final Duke mask shape: {masks.shape}")
    print(f"Final Duke mask classes: {np.unique(masks)}")
    
    return images, masks


def combine_datasets(datasets_info):
    """
    Combine multiple datasets into a single training dataset.
    
    Args:
        datasets_info (list): List of tuples (images, masks, dataset_name)
    
    Returns:
        tuple: (combined_images, combined_masks)
    """
    all_images = []
    all_masks = []
    
    for images, masks, name in datasets_info:
        print(f"Adding {name} dataset: {len(images)} samples")
        all_images.append(images)
        all_masks.append(masks)
    
    combined_images = np.concatenate(all_images, axis=0)
    combined_masks = np.concatenate(all_masks, axis=0)
    
    print(f"Combined dataset: {len(combined_images)} total samples")
    return combined_images, combined_masks


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation"""
    pred_softmax = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=3).permute(0, 3, 1, 2).float()
    
    intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
    union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def segmentation_metrics(pred, target):
    """Calculate segmentation metrics"""
    pred_classes = torch.argmax(pred, dim=1)
    
    # Calculate per-class metrics
    dice_scores = []
    iou_scores = []
    
    for class_id in range(3):
        pred_mask = (pred_classes == class_id).float()
        target_mask = (target == class_id).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union > 0:
            dice = (2 * intersection) / union
            iou = intersection / (union - intersection)
        else:
            dice = 1.0  # Perfect score if both are empty
            iou = 1.0
            
        dice_scores.append(dice.item())
        iou_scores.append(iou.item())
    
    return np.mean(dice_scores), np.mean(iou_scores)

def plot_segmentation_results(model, images, masks, num_samples=3, save_dir="segmentation_results"):
    """Plot segmentation results"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(images))):
            img = images[idx]
            true_mask = masks[idx]
            
            # Get model prediction
            inp = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
            pred_logits = model(inp)
            pred_mask = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img[:, :, 0], cmap='gray')
            axes[0].set_title(f'Sample {idx}: Original Image')
            axes[0].axis('off')
            
            # True mask
            axes[1].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            masked_true = np.ma.masked_where(true_mask == 0, true_mask)
            axes[1].imshow(masked_true, cmap='viridis', alpha=0.8, vmin=0, vmax=2)
            axes[1].set_title('True Segmentation')
            axes[1].axis('off')
            
            # Predicted mask
            axes[2].imshow(img[:, :, 0], cmap='gray', alpha=0.7)
            masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
            axes[2].imshow(masked_pred, cmap='viridis', alpha=0.8, vmin=0, vmax=2)
            axes[2].set_title('Predicted Segmentation')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'segmentation_sample_{idx}.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()

if __name__ == "__main__":
    # Configuration
    use_nemours_data = True
    use_duke_data = True
    max_samples_per_dataset = 700  # Set to number to limit samples, None for all

    # Load datasets
    datasets = []
    
    if use_nemours_data:
        nemours_path = '/home/suraj/Git/SCR-Progression/Nemours_Jing_RL_Annotated.h5'
        if os.path.exists(nemours_path):
            print("Loading Nemours dataset...")
            nemours_images, nemours_masks = load_nemours_data_with_masks(nemours_path, target_size=(224, 224))
            if max_samples_per_dataset:
                nemours_images = nemours_images[:max_samples_per_dataset]
                nemours_masks = nemours_masks[:max_samples_per_dataset]
            datasets.append((nemours_images, nemours_masks, "Nemours"))
        else:
            print(f"Nemours dataset not found at {nemours_path}")
    
    if use_duke_data:
        duke_path = '/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control_normalized_corrected.h5'
        if os.path.exists(duke_path):
            print("Loading Duke dataset...")
            duke_images, duke_masks = load_duke_data_with_masks(duke_path, target_size=(224, 224))
            if max_samples_per_dataset:
                duke_images = duke_images[:max_samples_per_dataset]
                duke_masks = duke_masks[:max_samples_per_dataset]
            datasets.append((duke_images, duke_masks, "Duke"))
        else:
            print(f"Duke dataset not found at {duke_path}")
    
    if not datasets:
        print("No datasets found. Please ensure at least one dataset is available.")
        exit(1)
    
    # Combine datasets
    if len(datasets) > 1:
        images, masks = combine_datasets(datasets)
    else:
        images, masks = datasets[0][0], datasets[0][1]
        print(f"Using single dataset: {datasets[0][2]}")
    
    print(f"Final dataset: {len(images)} samples")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")
    print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Mask classes: {np.unique(masks)}")
    
    # Check for NaN values
    if np.isnan(images).any():
        print("⚠️  NaN values found in images - fixing...")
        images = np.nan_to_num(images, nan=0.0)
    
    if np.isnan(masks).any():
        print("⚠️  NaN values found in masks - fixing...")
        masks = np.nan_to_num(masks, nan=0)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(images))
    images = images[indices]
    masks = masks[indices]
    
    # Split data
    dataset = LayerAnnotationDataset(images, masks)
    n_total = len(dataset)
    n_test = int(0.2 * n_total)
    n_train = n_total - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)  # Smaller batch for segmentation
    test_loader = DataLoader(test_set, batch_size=8)

    # Model, loss, optimizer for segmentation
    model = LayerSegmentationCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    dice_criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []

    n_epochs = 20
    print(f"\nStarting segmentation training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Combined loss: CrossEntropy + Dice
            ce_loss = criterion(outputs, targets)
            dice_loss_val = dice_criterion(outputs, targets)
            loss = ce_loss + dice_loss_val
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_list = []
        val_iou_list = []
        
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(device), targets.to(device).long()
                outputs = model(imgs)
                
                ce_loss = criterion(outputs, targets)
                dice_loss_val = dice_criterion(outputs, targets)
                loss = ce_loss + dice_loss_val
                
                val_loss += loss.item()
                
                # Calculate metrics
                dice, iou = segmentation_metrics(outputs, targets)
                val_dice_list.append(dice)
                val_iou_list.append(iou)
        
        val_loss /= len(test_loader)
        val_dice = np.mean(val_dice_list)
        val_iou = np.mean(val_iou_list)
        
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        val_iou_scores.append(val_iou)

        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

    # Save model and generate visualizations
    torch.save(model.state_dict(), "CNN_segmentation_model.pth")
    print("\nGenerating visualizations...")
    
    # Get test samples for visualization
    X_test = np.array([test_set[i][0].numpy().transpose(1, 2, 0) for i in range(min(3, n_test))])
    y_test = np.array([test_set[i][1].numpy() for i in range(min(3, n_test))])
    
    plot_segmentation_results(model, X_test, y_test, num_samples=3, save_dir="segmentation_logs")
    
    # Plot training metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(range(1, n_epochs+1), train_losses, label='Train Loss')
    axes[0].plot(range(1, n_epochs+1), val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(range(1, n_epochs+1), val_dice_scores, label='Dice Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(range(1, n_epochs+1), val_iou_scores, label='IoU Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].set_title('Validation IoU Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    os.makedirs("segmentation_logs", exist_ok=True)
    plt.savefig("segmentation_logs/training_metrics.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    print("Training complete. Model saved as 'CNN_segmentation_model.pth'.")
    print("Visualizations saved in 'segmentation_logs/' directory.")
    print(f"\nFinal Performance Summary:")
    print(f"  Final Dice Score: {val_dice_scores[-1]:.4f}")
    print(f"  Final IoU Score: {val_iou_scores[-1]:.4f}")