import numpy as np
import torch
import matplotlib.pyplot as plt

def test_thick_vs_thin_masks():
    """Test to demonstrate the difference between thick and thin masks for metric calculation"""
    
    def lines_to_mask(lines, height=224, width=224, thickness=1):
        batch = lines.shape[0]
        mask = torch.zeros((batch, 2, height, width), device=lines.device)
        
        for b in range(batch):
            for i in range(2):  # ILM and BM
                y_coords = torch.clamp(lines[b, :, i].round().long(), 0, height-1)
                
                # Create thick lines by adding neighboring pixels
                for offset in range(-thickness//2, thickness//2 + 1):
                    y_thick = torch.clamp(y_coords + offset, 0, height-1)
                    mask[b, i, y_thick, torch.arange(width)] = 1
        
        return mask

    def dice_coefficient(pred_mask, target_mask, eps=1e-6):
        intersection = (pred_mask * target_mask).sum(dim=(2,3))
        union = pred_mask.sum(dim=(2,3)) + target_mask.sum(dim=(2,3))
        dice = (2 * intersection + eps) / (union + eps)
        return dice.mean().item()

    def iou_metric(pred_mask, target_mask, eps=1e-6):
        intersection = (pred_mask * target_mask).sum(dim=(2,3))
        union = pred_mask.sum(dim=(2,3)) + target_mask.sum(dim=(2,3)) - intersection
        iou = (intersection + eps) / (union + eps)
        return iou.mean().item()

    # Create synthetic example: perfect prediction
    height, width = 224, 224
    batch_size = 1
    
    true_lines = torch.zeros(batch_size, width, 2)
    true_lines[0, :, 0] = 50  # ILM at y=50
    true_lines[0, :, 1] = 150  # BM at y=150
    
    # Test different prediction offsets
    offsets = [0, 1, 2, 3, 5, 10]
    results_thin = []
    results_thick = []
    
    for offset in offsets:
        pred_lines = true_lines.clone()
        pred_lines[0, :, 0] = 50 + offset  # Offset ILM
        pred_lines[0, :, 1] = 150 + offset  # Offset BM
        
        # Thin masks (1 pixel)
        true_mask_thin = lines_to_mask(true_lines, thickness=1)
        pred_mask_thin = lines_to_mask(pred_lines, thickness=1)
        dice_thin = dice_coefficient(pred_mask_thin, true_mask_thin)
        iou_thin = iou_metric(pred_mask_thin, true_mask_thin)
        
        # Thick masks (3 pixels)
        true_mask_thick = lines_to_mask(true_lines, thickness=3)
        pred_mask_thick = lines_to_mask(pred_lines, thickness=3)
        dice_thick = dice_coefficient(pred_mask_thick, true_mask_thick)
        iou_thick = iou_metric(pred_mask_thick, true_mask_thick)
        
        results_thin.append((dice_thin, iou_thin))
        results_thick.append((dice_thick, iou_thick))
        
        print(f"Offset {offset}px:")
        print(f"  Thin masks - Dice: {dice_thin:.4f}, IoU: {iou_thin:.4f}")
        print(f"  Thick masks - Dice: {dice_thick:.4f}, IoU: {iou_thick:.4f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dice scores
    dice_thin_vals = [r[0] for r in results_thin]
    dice_thick_vals = [r[0] for r in results_thick]
    
    ax1.plot(offsets, dice_thin_vals, 'r-o', label='Thin masks (1px)', linewidth=2)
    ax1.plot(offsets, dice_thick_vals, 'b-o', label='Thick masks (3px)', linewidth=2)
    ax1.set_xlabel('Prediction Offset (pixels)')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Score vs Prediction Offset')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IoU scores
    iou_thin_vals = [r[1] for r in results_thin]
    iou_thick_vals = [r[1] for r in results_thick]
    
    ax2.plot(offsets, iou_thin_vals, 'r-o', label='Thin masks (1px)', linewidth=2)
    ax2.plot(offsets, iou_thick_vals, 'b-o', label='Thick masks (3px)', linewidth=2)
    ax2.set_xlabel('Prediction Offset (pixels)')
    ax2.set_ylabel('IoU Score')
    ax2.set_title('IoU Score vs Prediction Offset')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mask_thickness_comparison.png', bbox_inches='tight', dpi=150)
    plt.show()
    
    print("\nConclusion:")
    print("- Thin masks (1px) are extremely sensitive to pixel-level errors")
    print("- Thick masks (3px) provide more reasonable evaluation for line-based predictions")
    print("- This explains why the original metrics were so low!")

if __name__ == "__main__":
    test_thick_vs_thin_masks()
