import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from pycm import ConfusionMatrix
import uuid
from sklearn.model_selection import KFold

# Edge Attention Block
class EdgeAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # Simulate Canny
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_out = F.relu(self.conv(x))
        edge_out = F.relu(self.edge_conv(conv_out))  # Edge detection
        attention = self.sigmoid(edge_out)
        return conv_out * attention

# Spatial Attention Block
class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        avg_out = avg_out.expand(-1, -1, x.size(2), x.size(3))
        concat = torch.cat([max_out, avg_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        attention = self.upsample(attention)
        return x * attention

# Hybrid U-Net Model
class HybridUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(HybridUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ea1 = EdgeAttentionBlock(64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ea2 = EdgeAttentionBlock(128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.sa3 = SpatialAttentionBlock(256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.sa4 = SpatialAttentionBlock(512)
        self.pool4 = nn.MaxPool2d(2)

        self.base = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.sa5 = SpatialAttentionBlock(1024)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec6 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec7 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec8 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec9 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        c1_ea = self.ea1(c1)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        c2_ea = self.ea2(c2)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        c3_sa = self.sa3(c3)
        p3 = self.pool3(c3)

        c4 = self.enc4(p3)
        c4_sa = self.sa4(c4)
        p4 = self.pool4(c4)

        c5 = self.base(p4)
        c5_sa = self.sa5(c5)

        # Decoder
        u6 = self.up6(c5_sa)
        u6 = torch.cat([u6, c4_sa], dim=1)
        c6 = self.dec6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3_sa], dim=1)
        c7 = self.dec7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2_ea], dim=1)
        c8 = self.dec8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1_ea], dim=1)
        c9 = self.dec9(u9)

        out = self.out(c9)
        return out

# Custom Dataset
class OCTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)  # Assuming masks are grayscale

        # Preprocess image
        image = cv2.resize(image, (512, 256))
        image = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# AdaBound Optimizer (simplified implementation)
class AdaBound(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps)
        super(AdaBound, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                step_size = group['lr']
                final_lr = group['final_lr']
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                lr = min(max(step_size, lower_bound), upper_bound)

                p.data.add_(-lr * exp_avg / (torch.sqrt(exp_avg_sq) + group['eps']))

# Dice Coefficient
def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = torch.argmax(y_pred, dim=1).flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true_f.sum() + y_pred_f.sum() + 1.)

# Training and Evaluation
def train_model(model, train_loader, val_loader, device, num_epochs=300):
    optimizer = AdaBound(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    best_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_dice += dice_coefficient(masks, outputs).item()
        
        val_dice /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}')

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')

        # Learning rate scheduling
        if epoch > 5 and val_dice <= best_dice:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Early stopping
        if epoch > 10 and val_dice <= best_dice:
            print("Early stopping")
            break

# Main
if __name__ == "__main__":
    # Dataset paths (replace with your Duke WLOA dataset paths)
    image_paths = ["path/to/images/1.png", "path/to/images/2.png"]  # Update with actual paths
    mask_paths = ["path/to/masks/1.png", "path/to/masks/2.png"]    # Update with actual paths

    # Augmentation pipeline
    transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomSnow(p=0.1),
        A.CLAHE(p=0.2),
        A.Blur(p=0.2),
        A.InvertImg(p=0.2),
        A.CoarseDropout(p=0.2),
        A.Downscale(p=0.2),
        A.Equalize(p=0.2),
        A.ToTensorV2()
    ])

    # Dataset and DataLoader
    dataset = OCTDataset(image_paths, mask_paths, transform=transform)
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridUNet(num_classes=5).to(device)  # Adjust num_classes based on your annotations

    # 6-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/6")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)
        
        train_model(model, train_loader, val_loader, device)

    # Evaluation (example with PyCM)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy().flatten())
            ground_truths.extend(masks.cpu().numpy().flatten())
    
    cm = ConfusionMatrix(actual_vector=ground_truths, predict_vector=predictions)
    print(f"ARI: {cm.ARI:.4f}, Dice: {dice_coefficient(torch.tensor(ground_truths), torch.tensor(predictions)):.4f}")