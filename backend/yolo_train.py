from ultralytics import YOLO
from pathlib import Path

DATA_YAML = Path(__file__).parent / "data" / "dataset.yaml"
OUT_DIR = Path(__file__).parent / "models"

def train():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    model = YOLO("yolov8n.pt")
    model.train(data=str(DATA_YAML), epochs=50, imgsz=640, device=0 if YOLO.check_gpu() else 'cpu', project=str(OUT_DIR), name="exp", exist_ok=True)
    # Symlink/copy best.pt to models/best.pt for inference
    best = OUT_DIR / "exp" / "weights" / "best.pt"
    if best.exists():
        (OUT_DIR / "best.pt").write_bytes(best.read_bytes())

if __name__ == "__main__":
    train()
