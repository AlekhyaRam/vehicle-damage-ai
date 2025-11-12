import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = Path(__file__).parent / "models" / "cost_regressor.pkl"

def load_cost_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    # cold start: a dumb regressor to avoid crashes (train your own quickly with yolo_train.py)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    # trivial fit on mock data to allow predict()
    X = np.array([[0,0,0],[1,0.1,0.1],[2,0.2,0.2]])
    y = np.array([100, 800, 1500])
    rf.fit(X, y)
    return rf

def predict_cost(model, feats_dict):
    X = np.array([[feats_dict["num_boxes"], feats_dict["total_box_area_frac"], feats_dict["max_box_area_frac"]]])
    cost = float(model.predict(X)[0])
    # Simple severity 0-1
    severity = min(1.0, feats_dict["total_box_area_frac"] * 4 + 0.1 * feats_dict["num_boxes"])
    return {"estimated_cost": round(cost, 2), "severity_score": round(severity, 3)}
