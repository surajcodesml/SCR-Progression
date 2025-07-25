import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

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
        plt.show()
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


if __name__ == "__main__":
    # Load data
    with h5py.File('/home/skumar/Git/SCR-Progression/Duke_Control_processed.h5', 'r') as f:
        images = f['images'][:]  # (N, 224, 224)
        layer_maps = f['layer_maps'][:]  # (N, 224, 3) or (N, 224, 2)

    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    layer_maps = layer_maps[:, :, [0, 2]]  # Only ILM and BM

    
    # Shuffle the dataset
    indices = np.random.permutation(len(images))
    images = images[indices]
    layer_maps = layer_maps[indices]
    
    # Limit to 1000 images for training
    images = images[:1000]
    layer_maps = layer_maps[:1000]

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

    n_epochs = 50
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

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Plot Loss vs Epoch
    plt.figure()
    plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.savefig('loss_vs_epoch.png')
    plt.show()

    # Plot F1 vs Epoch
    plt.figure()
    plt.plot(range(1, n_epochs+1), val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')
    plt.legend()
    plt.savefig('f1_vs_epoch.png')
    plt.show()

    # Plot Precision-Recall Curve (per epoch)
    plt.figure()
    plt.plot(val_recalls, val_precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (per epoch)')
    plt.savefig('precision_recall_curve.png')
    plt.show()

    '''

    # Tracking lists
    train_losses = []
    val_losses = []
    val_f1s = []
    val_precisions = []
    val_recalls = []

    # Training loop
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            mae = mae_metric(outputs, targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            mae = mae_metric(outputs, targets)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * imgs.size(0)
    test_loss /= n_test
    print(f"Test MSE: {test_loss:.4f}")




    pred_mask = lines_to_mask(outputs)
    target_mask = lines_to_mask(targets)
    dice = dice_coefficient(pred_mask, target_mask)
    precision, recall, f1 = precision_recall_f1(pred_mask, target_mask)
    print(f"Test Dice: {dice:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}")
    
    '''
    
    # Save model
    torch.save(model.state_dict(), "CNN_regression_model.pth")

    # Visualize predictions
    X_test = np.array([test_set[i][0].numpy().transpose(1, 2, 0) for i in range(min(5, n_test))])
    y_test = np.array([test_set[i][1].numpy() for i in range(min(5, n_test))])
    plot_layer_annotations(model, X_test, y_test, num_samples=5, save_dir="pytorch_logs", model_name="CNN_regression_model")