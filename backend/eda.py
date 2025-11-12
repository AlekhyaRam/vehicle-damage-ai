from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json

DATA = Path(__file__).parent / "data"

def load_yolo_annotations(split="train"):
    imgs_dir = DATA / "images" / split
    labels_dir = DATA / "annotations" / split
    rows = []
    for txt in labels_dir.glob("*.txt"):
        img_file = imgs_dir / (txt.stem + ".jpg")
        if not img_file.exists():
            img_file = imgs_dir / (txt.stem + ".png")
        if not img_file.exists():
            continue
        W, H = Image.open(img_file).size
        with open(txt, "r") as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                rows.append({"img": img_file.name, "cls": int(cls), "x": x, "y": y, "w": w, "h": h, "W": W, "H": H})
    return pd.DataFrame(rows)

def stats_and_heatmap(split="train", out_dir=DATA / "eda"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_yolo_annotations(split)
    df.to_csv(out_dir / f"{split}_annots.csv", index=False)

    # class counts & box size stats
    stats = df.groupby("cls")[["w","h"]].agg(["mean","median","count"]).reset_index()
    stats.to_csv(out_dir / f"{split}_class_stats.csv", index=False)

    # pixel heatmap (accumulate boxes)
    # Convert normalized YOLO -> pixel boxes and paint
    imgs_dir = DATA / "images" / split
    # pick a canonical size
    CANW, CANH = 640, 640
    heat = np.zeros((CANH, CANW), dtype=np.float32)
    for _, r in df.iterrows():
        x,y,w,h = r["x"],r["y"],r["w"],r["h"]
        px = int(x * CANW); py = int(y * CANH)
        pw = int(w * CANW); ph = int(h * CANH)
        x1 = max(0, px - pw//2); y1 = max(0, py - ph//2)
        x2 = min(CANW-1, px + pw//2); y2 = min(CANH-1, py + ph//2)
        heat[y1:y2+1, x1:x2+1] += 1.0

    plt.figure()
    plt.imshow(heat, cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{split}_pixel_heatmap.png", dpi=200)
    plt.close()
    return str(out_dir)
