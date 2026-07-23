"""
Building Footprint Extraction - U-Net (ResNet backbone)
========================================================
End-to-end pipeline for 4-band aerial imagery:

  1. prepare : rasterize footprint shapefile -> per-tile masks, chip tiles into patches
  2. train   : train U-Net (ResNet-34 encoder, 4-channel input) with Dice+BCE
  3. infer   : sliding-window inference on new tiles -> prob/binary GeoTIFFs + GeoPackage

Usage (from VS Code terminal):
    python building_footprint_unet.py prepare
    python building_footprint_unet.py train
    python building_footprint_unet.py infer
    python building_footprint_unet.py all        # run everything in sequence

Dependencies:
    pip install torch torchvision segmentation-models-pytorch albumentations \
                rasterio geopandas shapely tqdm matplotlib

Assumptions (edit CONFIG below):
  - Training tiles are GeoTIFFs (5000x5000, 4 bands) in one folder
  - Inference tiles in another folder
  - Shapefile and tiles share (or can be reprojected to) the same CRS
  - Band order is consistent between training and inference tiles
"""

import os
import glob
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import rasterio
from rasterio import features as rio_features
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import shape

import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

ROOT_DIR = Path("/opt/dlami/nvme/code/Project")

CONFIG = {
    # ---- paths ----
    "train_tiles_dir":  ROOT_DIR / r"TrainingTiles/3in",        # 36 GeoTIFFs, 5000x5000, 4 bands
    "footprints_shp":   ROOT_DIR / r"IndyMapShapefile/IndyBuildingTrain.shp",
    "infer_tiles_dir":  ROOT_DIR / r"RawTiles10_3inch",        # 10 GeoTIFFs for inference
    "work_dir":         ROOT_DIR / r"CustomModel/Results_3in",                     # masks, chips, checkpoints, outputs

    # ---- data prep ----
    "patch_size":       512,
    "train_stride":     384,     # 512 - 384 = 128 px overlap between training chips
    "val_tile_frac":    0.17,    # ~6 of 36 tiles held out for validation (split by TILE)
    "keep_empty_frac":  0.15,    # fraction of building-free patches kept as negatives

    # ---- normalization ----
    # Set to lists of 4 floats to override; None = auto-compute from training chips
    "band_means":       None,
    "band_stds":        None,

    # ---- model ----
    "encoder":          "resnet34",   # try "resnet50" if you have GPU headroom
    "encoder_weights":  "imagenet",   # smp adapts the first conv for 4 channels
    "in_channels":      4,

    # ---- training ----
    "epochs":           1,
    "batch_size":       8,
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "num_workers":      4,
    "amp":              True,

    # ---- inference ----
    "infer_patch":      512,
    "infer_stride":     256,     # 50% overlap, blended
    "infer_batch":      8,
    "prob_threshold":   0.5,
    "min_building_px":  30,      # drop polygons smaller than this many pixels
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORK = Path(CONFIG["work_dir"])
MASK_DIR = WORK / "masks"
CHIP_IMG_DIR = WORK / "chips/images"
CHIP_MSK_DIR = WORK / "chips/masks"
CKPT_DIR = WORK / "checkpoints"
PRED_DIR = WORK / "predictions"
INDEX_PATH = WORK / "chips/index.json"
STATS_PATH = WORK / "chips/band_stats.json"
CKPT_BEST = CKPT_DIR / "best.pt"

for d in (MASK_DIR, CHIP_IMG_DIR, CHIP_MSK_DIR, CKPT_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def list_tiles(folder):
    return sorted(glob.glob(os.path.join(folder, "*.tif")))


def resolve_footprints_path():
    cfg_path = Path(CONFIG["footprints_shp"])
    if cfg_path.is_file():
        return str(cfg_path)

    if cfg_path.is_dir():
        candidates = sorted(cfg_path.glob("*.shp"))
        if not candidates:
            raise FileNotFoundError(f"No .shp shapefile found in {cfg_path}")
        return str(candidates[0])

    parent = cfg_path.parent
    if parent.exists():
        candidates = sorted(parent.glob("*.shp"))
        if candidates:
            print(f"Configured footprints path missing; using discovered shapefile: {candidates[0]}")
            return str(candidates[0])

    raise FileNotFoundError(f"Footprint shapefile not found: {cfg_path}")


# ============================================================================
# Stage 1 - PREPARE: rasterize masks + chip tiles
# ============================================================================

def rasterize_tile_mask(tile_path, gdf, out_path):
    """Burn footprints intersecting this tile into a binary mask aligned to it."""
    with rasterio.open(tile_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        b = src.bounds
        cand = gdf.cx[b.left:b.right, b.bottom:b.top]
        if len(cand) > 0:
            mask = rio_features.rasterize(
                ((geom, 1) for geom in cand.geometry),
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0, dtype="uint8", all_touched=False,
            )
        else:
            mask = np.zeros((src.height, src.width), dtype="uint8")
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    return float(mask.mean())


def window_positions(H, W, patch, stride):
    """Top-left corners covering the full extent, incl. edge-aligned last row/col."""
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)
    return [(y, x) for y in ys for x in xs]


def chip_tile(tile_path, mask_path, split, patch, stride, keep_empty_frac, rng):
    """Write image/mask chips as .npy; return index records."""
    records = []
    stem = Path(tile_path).stem
    with rasterio.open(tile_path) as img_src, rasterio.open(mask_path) as msk_src:
        for y, x in window_positions(img_src.height, img_src.width, patch, stride):
            win = Window(x, y, patch, patch)
            m = msk_src.read(1, window=win)
            if m.max() == 0 and rng.random() > keep_empty_frac:
                continue  # subsample empty patches
            im = img_src.read(window=win)  # (4, patch, patch)
            img_name = f"{stem}_{y}_{x}_img.npy"
            msk_name = f"{stem}_{y}_{x}_msk.npy"
            np.save(CHIP_IMG_DIR / img_name, im)
            np.save(CHIP_MSK_DIR / msk_name, m)
            records.append({"img": img_name, "msk": msk_name, "split": split})
    return records


def split_tiles(train_tile_paths):
    """Deterministic train/val split BY TILE to avoid spatial leakage."""
    n_val = max(1, round(len(train_tile_paths) * CONFIG["val_tile_frac"]))
    shuffled = train_tile_paths.copy()
    random.Random(SEED).shuffle(shuffled)
    return set(shuffled[n_val:]), set(shuffled[:n_val])  # train, val


def stage_prepare():
    train_tile_paths = list_tiles(CONFIG["train_tiles_dir"])
    assert train_tile_paths, f"No .tif tiles found in {CONFIG['train_tiles_dir']}"
    print(f"{len(train_tile_paths)} training tiles found")

    # --- load & clean footprints ---
    footprints_path = resolve_footprints_path()
    gdf = gpd.read_file(footprints_path)
    print(f"{len(gdf)} footprint polygons | CRS: {gdf.crs}")
    gdf["geometry"] = gdf.geometry.buffer(0)  # fix invalid geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    # --- rasterize masks ---
    coverage = {}
    for tp in tqdm(train_tile_paths, desc="Rasterizing masks"):
        out = MASK_DIR / (Path(tp).stem + "_mask.tif")
        if not out.exists():
            coverage[Path(tp).stem] = rasterize_tile_mask(tp, gdf, out)
    if coverage:
        vals = np.array(list(coverage.values()))
        print(f"Building pixel fraction per tile - "
              f"mean {vals.mean():.3f}, min {vals.min():.3f}, max {vals.max():.3f}")

    # --- split & chip ---
    train_tiles, val_tiles = split_tiles(train_tile_paths)
    print(f"Train tiles: {len(train_tiles)} | Val tiles: {len(val_tiles)}")
    print("Val:", [Path(p).name for p in sorted(val_tiles)])

    rng = random.Random(SEED)
    records = []
    for tp in tqdm(train_tile_paths, desc="Chipping"):
        mp = MASK_DIR / (Path(tp).stem + "_mask.tif")
        split = "val" if tp in val_tiles else "train"
        records += chip_tile(tp, mp, split, CONFIG["patch_size"],
                             CONFIG["train_stride"], CONFIG["keep_empty_frac"], rng)
    INDEX_PATH.write_text(json.dumps(records))
    n_tr = sum(r["split"] == "train" for r in records)
    n_va = sum(r["split"] == "val" for r in records)
    print(f"Chipped {len(records)} patches -> train {n_tr} | val {n_va}")

    # --- normalization stats from a sample of training chips ---
    compute_band_stats(records)


def compute_band_stats(records):
    if CONFIG["band_means"] is not None:
        stats = {"means": CONFIG["band_means"], "stds": CONFIG["band_stds"]}
    else:
        sample = [r for r in records if r["split"] == "train"]
        sample = random.Random(SEED).sample(sample, min(200, len(sample)))
        acc = []
        for r in sample:
            im = np.load(CHIP_IMG_DIR / r["img"]).astype(np.float64)
            acc.append(im.reshape(im.shape[0], -1))
        stacked = np.concatenate(acc, axis=1)
        stats = {"means": stacked.mean(axis=1).tolist(),
                 "stds": (stacked.std(axis=1) + 1e-6).tolist()}
    STATS_PATH.write_text(json.dumps(stats))
    print("Per-band means:", np.round(stats["means"], 2))
    print("Per-band stds: ", np.round(stats["stds"], 2))
    return stats


def load_band_stats():
    stats = json.loads(STATS_PATH.read_text())
    return (np.array(stats["means"], dtype=np.float32),
            np.array(stats["stds"], dtype=np.float32))


# ============================================================================
# Stage 2 - TRAIN
# ============================================================================

TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                       border_mode=0, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
])


class BuildingDataset(Dataset):
    def __init__(self, records, split, band_means, band_stds, augment=False):
        self.recs = [r for r in records if r["split"] == split]
        self.augment = augment
        self.means = band_means
        self.stds = band_stds

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = np.load(CHIP_IMG_DIR / r["img"]).astype(np.float32)  # (4,H,W)
        msk = np.load(CHIP_MSK_DIR / r["msk"]).astype(np.float32)  # (H,W)

        if self.augment:
            out = TRAIN_AUG(image=np.moveaxis(img, 0, -1), mask=msk)  # HWC for albumentations
            img = np.moveaxis(out["image"], -1, 0)
            msk = out["mask"]

        img = (img - self.means[:, None, None]) / self.stds[:, None, None]
        return (torch.from_numpy(img.astype(np.float32)),
                torch.from_numpy(msk)[None, ...])


def build_model():
    return smp.Unet(
        encoder_name=CONFIG["encoder"],
        encoder_weights=CONFIG["encoder_weights"],
        in_channels=CONFIG["in_channels"],
        classes=1,
    ).to(DEVICE)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tp = fp = fn = tn = 0
    loss_sum, n = 0.0, 0
    for img, msk in loader:
        img, msk = img.to(DEVICE), msk.to(DEVICE)
        with torch.autocast(device_type="cuda",
                            enabled=CONFIG["amp"] and DEVICE == "cuda"):
            logits = model(img)
            loss = criterion(logits, msk)
        loss_sum += loss.item() * img.size(0)
        n += img.size(0)
        pred = (torch.sigmoid(logits) > 0.5).long()
        _tp, _fp, _fn, _tn = smp.metrics.get_stats(pred, msk.long(), mode="binary")
        tp += _tp.sum(); fp += _fp.sum(); fn += _fn.sum(); tn += _tn.sum()
    iou = (tp / (tp + fp + fn + 1e-9)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-9)).item()
    return loss_sum / max(n, 1), iou, f1


def stage_train():
    print("Device:", DEVICE)
    records = json.loads(INDEX_PATH.read_text())
    band_means, band_stds = load_band_stats()

    train_ds = BuildingDataset(records, "train", band_means, band_stds, augment=True)
    val_ds = BuildingDataset(records, "val", band_means, band_stds, augment=False)
    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                        num_workers=CONFIG["num_workers"], pin_memory=True)
    print(f"{len(train_ds)} train chips | {len(val_ds)} val chips")

    model = build_model()
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()

    def criterion(logits, target):
        return 0.5 * bce(logits, target) + 0.5 * dice(logits, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=CONFIG["amp"] and DEVICE == "cuda")

    best_iou = 0.0
    history = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for img, msk in pbar:
            img = img.to(DEVICE, non_blocking=True)
            msk = msk.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",
                                enabled=CONFIG["amp"] and DEVICE == "cuda"):
                logits = model(img)
                loss = criterion(logits, msk)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * img.size(0)
            n += img.size(0)
            pbar.set_postfix(loss=f"{running / n:.4f}")
        scheduler.step()

        val_loss, val_iou, val_f1 = evaluate(model, val_dl, criterion)
        history.append({"epoch": epoch, "train_loss": running / n,
                        "val_loss": val_loss, "val_iou": val_iou, "val_f1": val_f1})
        print(f"  val_loss {val_loss:.4f} | IoU {val_iou:.4f} | F1 {val_f1:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(),
                        "band_means": band_means.tolist(),
                        "band_stds": band_stds.tolist(),
                        "config": CONFIG}, CKPT_BEST)
            print(f"  saved new best (IoU {best_iou:.4f})")

    (WORK / "history.json").write_text(json.dumps(history, indent=2))
    print("Best val IoU:", best_iou)
    plot_history(history)
    save_random_validation_examples(model, records, band_means, band_stds)


def plot_history(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = [e["epoch"] for e in history]
    ax[0].plot(ep, [e["train_loss"] for e in history], label="train")
    ax[0].plot(ep, [e["val_loss"] for e in history], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ep, [e["val_iou"] for e in history], label="IoU")
    ax[1].plot(ep, [e["val_f1"] for e in history], label="F1")
    ax[1].set_title("Validation metrics"); ax[1].legend()
    plt.tight_layout()
    out = WORK / "training_curves.png"
    plt.savefig(out, dpi=300)
    print("Training curves saved to", out)

def save_random_validation_examples(model, records, band_means, band_stds, num_examples=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val_records = [r for r in records if r["split"] == "val"]
    selected = random.Random(SEED).sample(val_records, min(num_examples, len(val_records)))

    examples = []
    model.eval()
    with torch.no_grad():
        for r in selected:
            img = np.load(CHIP_IMG_DIR / r["img"]).astype(np.float32)
            msk = np.load(CHIP_MSK_DIR / r["msk"]).astype(np.uint8)

            norm_img = (img - band_means[:, None, None]) / band_stds[:, None, None]
            batch = torch.from_numpy(norm_img[None]).to(DEVICE)
            logits = model(batch)
            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

            examples.append((img, msk, pred))

    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 3 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, 0)

    for i, (img, msk, pred) in enumerate(examples):
        rgb = np.moveaxis(img[:3], 0, -1)
        if rgb.max() > 1.0:
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(msk, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(WORK / "validation_examples.png", dpi=300)



# ============================================================================
# Stage 3 - INFER: sliding window + blending, then vectorize
# ============================================================================

def make_blend_weight(patch):
    """Hanning window weight - high confidence at center, low at edges."""
    ramp = np.hanning(patch)
    return np.clip(np.outer(ramp, ramp).astype(np.float32), 1e-3, None)


@torch.no_grad()
def predict_tile(tile_path, model, band_means, band_stds):
    patch = CONFIG["infer_patch"]
    stride = CONFIG["infer_stride"]
    with rasterio.open(tile_path) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()
        img = src.read().astype(np.float32)  # (4,H,W) ~400 MB/tile, fits in RAM

    band_means = band_means.astype(np.float32, copy=False)
    band_stds = band_stds.astype(np.float32, copy=False)
    img = (img - band_means[:, None, None]) / band_stds[:, None, None]

    prob_acc = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    wpatch = make_blend_weight(patch)
    positions = window_positions(H, W, patch, stride)
    bs = CONFIG["infer_batch"]

    for b in tqdm(range(0, len(positions), bs),
                  desc=Path(tile_path).stem, leave=False):
        batch_pos = positions[b:b + bs]
        batch = np.stack([img[:, y:y + patch, x:x + patch] for y, x in batch_pos])
        batch = batch.astype(np.float32, copy=False)
        t = torch.from_numpy(batch).to(DEVICE)
        with torch.autocast(device_type="cuda",
                            enabled=CONFIG["amp"] and DEVICE == "cuda"):
            p = torch.sigmoid(model(t))[:, 0].float().cpu().numpy()
        for (y, x), pi in zip(batch_pos, p):
            prob_acc[y:y + patch, x:x + patch] += pi * wpatch
            weight_acc[y:y + patch, x:x + patch] += wpatch

    prob = prob_acc / weight_acc
    pred = (prob >= CONFIG["prob_threshold"]).astype("uint8")

    stem = Path(tile_path).stem
    p_prof = profile.copy()
    p_prof.update(count=1, dtype="float32", nodata=None, compress="lzw")
    with rasterio.open(PRED_DIR / f"{stem}_prob.tif", "w", **p_prof) as dst:
        dst.write(prob, 1)
    m_prof = profile.copy()
    m_prof.update(count=1, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(PRED_DIR / f"{stem}_pred.tif", "w", **m_prof) as dst:
        dst.write(pred, 1)


def vectorize_predictions(infer_tile_paths):
    """Binary rasters -> polygons -> one combined GeoPackage."""
    all_polys = []
    crs = None
    transform = None
    for tp in infer_tile_paths:
        stem = Path(tp).stem
        with rasterio.open(PRED_DIR / f"{stem}_pred.tif") as src:
            pred = src.read(1)
            transform, crs = src.transform, src.crs
            px_area = abs(transform.a * transform.e)
        for geom, _val in rio_features.shapes(pred, mask=pred == 1,
                                              transform=transform):
            poly = shape(geom)
            if poly.area / px_area < CONFIG["min_building_px"]:
                continue
            all_polys.append({"geometry": poly, "tile": stem,
                              "area_m2": round(poly.area, 1)})

    if not all_polys:
        print("No polygons produced - check threshold / model quality.")
        return
    gdf_pred = gpd.GeoDataFrame(all_polys, crs=crs)
    gdf_pred["geometry"] = gdf_pred.geometry.simplify(abs(transform.a) * 0.75)
    out_gpkg = PRED_DIR / "predicted_footprints.gpkg"
    gdf_pred.to_file(out_gpkg, driver="GPKG")
    print(f"{len(gdf_pred)} predicted footprints -> {out_gpkg}")


def stage_infer():
    print("Device:", DEVICE)
    infer_tile_paths = list_tiles(CONFIG["infer_tiles_dir"])
    assert infer_tile_paths, f"No .tif tiles found in {CONFIG['infer_tiles_dir']}"
    print(f"{len(infer_tile_paths)} inference tiles")

    ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only= True)
    band_means = np.array(ckpt["band_means"], dtype=np.float32)
    band_stds = np.array(ckpt["band_stds"], dtype=np.float32)
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model = model.float()
    model.eval()

    for tp in infer_tile_paths:
        predict_tile(tp, model, band_means, band_stds)
    print("Rasters written to", PRED_DIR)

    vectorize_predictions(infer_tile_paths)


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="U-Net building footprint extraction pipeline")
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["prepare", "train", "infer", "all"],
                        help="Pipeline stage to run (default: all)")
    args = parser.parse_args()

    if args.stage in ("prepare", "all"):
        stage_prepare()
    if args.stage in ("train", "all"):
        stage_train()
    if args.stage in ("infer", "all"):
        stage_infer()


if __name__ == "__main__":
    main()
