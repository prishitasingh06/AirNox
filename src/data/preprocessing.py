"""
Reproducible dataset splitting and preprocessing pipeline.
Prevents data leakage by enforcing stratified, seed-locked train/val/test splits
and saving exact split manifests to data/splits/.
"""

import json
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42


def prepare_splits(val_size: float = 0.15, test_size: float = 0.15):
    records = []
    if not RAW_DIR.exists():
        raise FileNotFoundError("data/raw not found. Run `python src/data/download_dataset.py` first.")

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                records.append({"filepath": str(img_path), "label": label})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No images found in data/raw/.")

    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), stratify=df["label"], random_state=RANDOM_SEED
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label"], random_state=RANDOM_SEED
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    # Create standardized processed folders with autocontrast normalization
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for _, row in split_df.iterrows():
            src = Path(row["filepath"])
            dst_dir = PROCESSED_DIR / split_name / row["label"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB")
                im = ImageOps.autocontrast(im, cutoff=1)
                im = im.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                im.save(dst_dir / src.name, quality=95)

    meta = {
        "random_seed": RANDOM_SEED,
        "image_size": IMAGE_SIZE,
        "num_classes": int(df["label"].nunique()),
        "classes": sorted(df["label"].unique().tolist()),
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    (SPLITS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Splits created successfully: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


if __name__ == "__main__":
    prepare_splits()
