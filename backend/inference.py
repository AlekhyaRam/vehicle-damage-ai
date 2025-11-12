from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Lazy-load model
_model = None
WEIGHTS = Path(__file__).parent / "models" / "best.pt"  # put your fine-tuned weights here; fallback to yolov8n

def get_model():
    global _model
    if _model is None:
        if WEIGHTS.exists():
            _model = YOLO(str(WEIGHTS))
        else:
            _model = YOLO("yolov8n.pt")  # warm start if fine-tuned weights not present
    return _model

def run_yolo(image_path: Path) -> Dict[str, Any]:
    model = get_model()
    results = model.predict(source=str(image_path), imgsz=640, conf=0.25)
    detections = []
    for r in results:
        for b in r.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy.cpu().numpy().astype(float).tolist()[0]
            detections.append({
                "class_id": cls,
                "class_name": model.names.get(cls, str(cls)),
                "confidence": conf,
                "bbox_xyxy": xyxy
            })
    return {"detections": detections}

def features_for_cost(img_path: Path, dets: List[Dict[str,Any]]) -> Dict[str, float]:
    # Simple, explainable features you can expand later
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    area_img = w * h
    num_boxes = len(dets)
    total_box_area = 0.0
    max_box_area = 0.0
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        a = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        total_box_area += a
        max_box_area = max(max_box_area, a)
    return {
        "num_boxes": float(num_boxes),
        "total_box_area_frac": float(total_box_area / area_img if area_img else 0.0),
        "max_box_area_frac": float(max_box_area / area_img if area_img else 0.0),
    }
