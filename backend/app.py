from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
from inference import run_yolo, features_for_cost
from cost_model import load_cost_model, predict_cost
from mesh import mesh_from_depth

app = FastAPI(title="Vehicle Damage AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

UPLOADS = Path(__file__).parent / "uploads"
UPLOADS.mkdir(exist_ok=True, parents=True)
_cost_model = load_cost_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...), build_mesh: bool = False):
    # Save
    fp = UPLOADS / file.filename
    with open(fp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # YOLO
    det = run_yolo(fp)
    feats = features_for_cost(fp, det["detections"])
    cost = predict_cost(_cost_model, feats)

    mesh_info = None
    if build_mesh:
        mesh_path = UPLOADS / (fp.stem + "_mesh.ply")
        mesh_file, curvature = mesh_from_depth(fp, mesh_path)
        mesh_info = {"mesh_file": Path(mesh_file).name, "curvature_proxy": curvature}

    return JSONResponse({
        "file": file.filename,
        "detections": det["detections"],
        "features": feats,
        "cost_estimate": cost,
        "mesh": mesh_info
    })
@app.get("/")
async def root():
    return {"message": "Vehicle Damage AI is running."}