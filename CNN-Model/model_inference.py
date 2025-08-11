'''
This script is used to perform inference using a pre-trained .pth model.
Using to test the model performance on a large set of test data.
'''

# eval_duke_multilayer.py
# Evaluate a multi-class (0=bg, 1=ILM, 2=BM) OCT layer model on a DUKE .h5 dataset.

import argparse
from pathlib import Path
import importlib
import numpy as np
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Import your utilities from your training file.
# Make sure CNN_mask_pytorch.py is importable (same folder or on PYTHONPATH).
from CNN_mask_pytorch import generate_annotations_masks  # <-- uses your rasterizer

def _to_tensor_img(img_np: np.ndarray, normalize="minmax"):
    img_np = img_np.astype(np.float32)
    if normalize == "minmax":
        vmin, vmax = float(img_np.min()), float(img_np.max())
        img_np = (img_np - vmin) / (vmax - vmin) if vmax > vmin else img_np * 0.0
    return torch.from_numpy(img_np).unsqueeze(0)  # [1,H,W]

def _resize_img(img: np.ndarray, size_hw):
    H, W = size_hw
    return np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))

def _resize_mask(mask: np.ndarray, size_hw):
    H, W = size_hw
    return np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))

def _make_multiclass_mask(hw_native, ilm_poly, bm_poly):
    """Multi-class mask at native resolution: 0=bg, 1=ILM, 2=BM."""
    Hn, Wn = hw_native
    m = np.zeros((Hn, Wn), dtype=np.uint8)
    if ilm_poly is not None:
        m_ilm = generate_annotations_masks((Hn, Wn), {"ILM": ilm_poly}).astype(bool)
        m[m_ilm] = 1
    if bm_poly is not None:
        m_bm = generate_annotations_masks((Hn, Wn), {"BM": bm_poly}).astype(bool)
        # BM wins ties if lines overlap (adjust if you prefer ILM priority)
        m[m_bm] = 2
    return m

class DukeH5MultiLayerDS(Dataset):
    """
    Expects:
      - h5['image']            -> [N, H, W]
      - h5['layers']['ILM']    -> [N, W]  (optional but expected)
      - h5['layers']['BM']     -> [N, W]  (optional but expected)
    If your combined H5 includes Nemours too, pass --idx-csv with DUKE indices to subset.
    """
    def __init__(self, h5_path, input_hw=(225,225), normalize="minmax", idx_list=None):
        self.h5 = h5py.File(h5_path, "r")
        self.imgs = self.h5["image"]          # [N,H,W]
        self.layers = self.h5["layers"]
        self.has_ilm = "ILM" in self.layers
        self.has_bm  = "BM"  in self.layers
        self.input_hw = input_hw
        self.normalize = normalize

        N = self.imgs.shape[0]
        if idx_list is None:
            self.indices = np.arange(N)
        else:
            self.indices = np.array(idx_list, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        img_native = self.imgs[idx]  # (Hn, Wn)
        Hn, Wn = img_native.shape

        ilm_poly = self.layers["ILM"][idx] if self.has_ilm else None  # (Wn,)
        bm_poly  = self.layers["BM"][idx]  if self.has_bm  else None

        # 1) native multi-class mask
        mask_native = _make_multiclass_mask((Hn, Wn), ilm_poly, bm_poly)

        # 2) resize to model input
        img_resized  = _resize_img(img_native, self.input_hw)
        mask_resized = _resize_mask(mask_native, self.input_hw).astype(np.int64)

        # 3) tensors
        img_t  = _to_tensor_img(img_resized, normalize=self.normalize)  # [1,H,W]
        mask_t = torch.from_numpy(mask_resized)                         # [H,W] long

        return {"image": img_t, "mask": mask_t, "index": idx}

def _update_confmat(cm, pred, tgt, C):
    # per-class binary CM accumulation
    with torch.no_grad():
        for c in range(C):
            pc = (pred == c); tc = (tgt == c)
            TP = (pc & tc).sum().item()
            FP = (pc & ~tc).sum().item()
            FN = (~pc & tc).sum().item()
            TN = pred.numel() - (TP + FP + FN)
            cm[c,0,0] += TN; cm[c,0,1] += FP; cm[c,1,0] += FN; cm[c,1,1] += TP

def _dice_iou(cm, ignore_index=0):
    eps = 1e-7
    C = cm.shape[0]
    dice = np.zeros(C, dtype=np.float64)
    iou  = np.zeros(C, dtype=np.float64)
    for c in range(C):
        TN, FP = cm[c,0,0], cm[c,0,1]
        FN, TP = cm[c,1,0], cm[c,1,1]
        dice[c] = (2*TP + eps) / (2*TP + FP + FN + eps)
        iou[c]  = (TP   + eps) / (TP + FP + FN + eps)
    if ignore_index is not None and 0 <= ignore_index < C:
        dice[ignore_index] = np.nan
        iou[ignore_index]  = np.nan
    return dice, iou

def run_eval(h5_path, model_module, model_class, checkpoint,
             input_size=(225,225), device=None, batch_size=8,
             num_workers=2, ignore_index=0, save_preds=None, idx_csv=None):

    # If your H5 has both datasets mixed, provide DUKE indices via --idx-csv (one index per line).
    idx_list = None
    if idx_csv:
        idx_list = [int(x.strip()) for x in Path(idx_csv).read_text().splitlines() if x.strip()]

    ds = DukeH5MultiLayerDS(h5_path, input_hw=input_size, idx_list=idx_list)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Build model (multi-class: 3 channels expected: bg, ILM, BM)
    mod = importlib.import_module(model_module.replace(".py",""))
    ModelClass = getattr(mod, model_class)
    try:
        model = ModelClass(num_classes=3)
    except TypeError:
        model = ModelClass()

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    if save_preds:
        out_dir = Path(save_preds); out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    cm = np.zeros((3, 2, 2), dtype=np.int64)  # 3 classes

    with torch.no_grad():
        for batch in dl:
            imgs = batch["image"].to(device)
            tgts = batch["mask"].to(device)

            logits = model(imgs)  # [B,3,H,W]
            if isinstance(logits, (list,tuple)): logits = logits[0]

            if logits.shape[1] != 3:
                raise ValueError(f"Model output ch={logits.shape[1]} but expected 3 (bg, ILM, BM)")

            preds = torch.argmax(logits, dim=1)  # [B,H,W] in {0,1,2}

            for b in range(preds.shape[0]):
                _update_confmat(cm, preds[b], tgts[b], C=3)
                if save_preds:
                    idx = int(batch["index"][b])
                    Image.fromarray(preds[b].cpu().numpy().astype(np.uint8), mode="L").save(
                        out_dir / f"pred_{idx:05d}.png"
                    )

    dice, iou = _dice_iou(cm, ignore_index=ignore_index)

    def _nanmean(x):
        xx = x[~np.isnan(x)]
        return float(xx.mean()) if len(xx) else float("nan")

    mean_dice, mean_iou = _nanmean(dice), _nanmean(iou)

    print("\nPer-class metrics (0=bg, 1=ILM, 2=BM):")
    for c, name in enumerate(["bg","ILM","BM"]):
        print(f"{name:>3} | Dice: {dice[c]:.4f} | IoU: {iou[c]:.4f}")
    print(f"\nMean (excluding ignored): Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")
    return dice, iou, mean_dice, mean_iou

def main():
    # Set arguments directly in code instead of parsing from command line
    args = argparse.Namespace(
        h5="/path/to/your/duke.h5",
        model_module="CNN_mask_pytorch",
        model_class="YourModelClassName",
        checkpoint="CNN_mask_pytorch.pth",
        input_size="225,225",
        device=None,
        batch_size=8,
        num_workers=2,
        ignore_index=0,
        save_preds=None,
        idx_csv=None
    )

    H, W = map(int, args.input_size.split(","))
    run_eval(
        h5_path=args.h5,
        model_module=args.model_module,
        model_class=args.model_class,
        checkpoint=args.checkpoint,
        input_size=(H, W),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ignore_index=args.ignore_index if args.ignore_index >= 0 else None,
        save_preds=args.save_preds,
        idx_csv=args.idx_csv,
    )

if __name__ == "__main__":
    main()
