# Quick-and-dirty mesh from monocular depth for a *relative* severity cue
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import open3d as o3d
from skimage import measure

def depth_map(image_path: Path):
    # Use MiDaS via OpenCV DNN (weights downloaded on first run)
    # Small model for portability
    import urllib.request, os, cv2
    model_url = "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_s32_256.onnx"
    model_path = Path(__file__).parent / "models" / "dpt_s32_256.onnx"
    model_path.parent.mkdir(exist_ok=True, parents=True)
    if not model_path.exists():
        urllib.request.urlretrieve(model_url, model_path)

    net = cv2.dnn.readNet(str(model_path))
    im = cv2.imread(str(image_path))
    blob = cv2.dnn.blobFromImage(im, 1/255.0, (256,256), mean=(0.5,0.5,0.5), swapRB=True, crop=False)
    net.setInput(blob)
    depth = net.forward()[0,0]
    depth = cv2.resize(depth, (im.shape[1], im.shape[0]))
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

def mesh_from_depth(image_path: Path, out_file: Path):
    d = depth_map(image_path)
    # Marching cubes on a pseudo-volume (stack depth as thin volume)
    vol = np.stack([d]*8, axis=-1)
    verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_file), mesh)
    # simple severity = curvature proxy
    curvature = np.asarray(mesh.get_curvature().mean())
    return str(out_file), float(curvature)
