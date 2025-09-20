# app.py
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# --- optional CORS (for browser/Vue direct calls) ---
try:
    from flask_cors import CORS
    CORS(app, resources={r"/ai/*": {"origins": "*"}})
except Exception:
    pass

# ----------------
# Utility helpers
# ----------------
def _stddev(arr):
    a = np.array(arr, dtype=float)
    return float(a.std(ddof=1)) if a.size > 1 else 0.0

def _slope(arr):
    y = np.array(arr, dtype=float)
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype=float)
    den = n * (x**2).sum() - (x.sum())**2
    if den == 0:
        return 0.0
    num = n * (x*y).sum() - x.sum() * y.sum()
    return float(num / den)

def _max_neg_run(arr):
    m = c = 0
    for v in arr:
        if v < 0:
            c += 1
            if c > m: m = c
        else:
            c = 0
    return m

# ----------------
# Core compute (pure)
# ----------------
def compute_forecast(history, horizon=3, alpha=0.3, seasonal_window=3):
    """Lightweight SES + recent seasonal tilt; returns dict."""
    hist = np.array(history, dtype=float).tolist()  # sanitize
    if not hist:
        return {
            "forecast": [0.0] * horizon,
            "stats": {"mean": 0.0, "std": 0.0, "last": 0.0, "smoothed": 0.0, "seasonal": 1.0}
        }

    mean = float(np.mean(hist))
    sd = _stddev(hist)
    last = float(hist[-1])

    # simple exponential smoothing
    s = hist[0]
    for v in hist[1:]:
        s = alpha * v + (1 - alpha) * s

    # seasonal tilt from the most recent window vs overall mean
    seasonal = 1.0
    if len(hist) >= seasonal_window * 2:  # require at least a bit of history
        recent = hist[-seasonal_window:]
        recent_mean = float(np.mean(recent))
        if mean != 0:
            seasonal = max(0.7, min(1.3, recent_mean / mean))  # clamp tilt

    base = s * seasonal
    forecast = []
    for _ in range(horizon):
        f = 0.8 * base + 0.2 * mean
        forecast.append(float(f))
        base = alpha * f + (1 - alpha) * base

    return {
        "forecast": forecast,
        "stats": {"mean": mean, "std": sd, "last": last, "smoothed": float(s), "seasonal": float(seasonal)}
    }

def compute_risk(history, cash_on_hand=0.0, avg_expense=0.0):
    """Heuristic risk score; returns dict."""
    hist = np.array(history, dtype=float).tolist()
    mean = float(np.mean(hist)) if hist else 0.0
    sd = _stddev(hist) if hist else 0.0
    m = _slope(hist) if hist else 0.0
    vr = abs(sd/mean) if mean != 0 else (1.0 if sd > 0 else 0.0)
    neg_run = _max_neg_run(hist)
    runway = (cash_on_hand / avg_expense) if avg_expense > 0 else float("inf")

    score = 0
    # downward trend
    if m < 0:
        # normalize trend intensity by |mean| to ~[0,1] then map to 0..35
        trend_intensity = min(1.0, abs(m) / (abs(mean) + 1e-6))
        score += min(35, int(trend_intensity * 35))
    # volatility penalty (0..25)
    score += min(25, int(min(1.0, vr) * 25))
    # consecutive negative runs (each step 5 pts, up to 20)
    score += min(20, neg_run * 5)
    # cash runway pressure (0/5/10/20)
    if runway != float("inf"):
        if runway < 1: score += 20
        elif runway < 2: score += 10
        elif runway < 3: score += 5

    score = max(0, min(100, score))
    level = "LOW" if score < 40 else ("MEDIUM" if score < 70 else "HIGH")

    return {
        "risk_score": int(score),
        "level": level,
        "signals": {
            "slope": float(m),
            "vol_ratio": float(vr),
            "max_neg_run": int(neg_run),
            "runway": (None if runway == float("inf") else float(runway)),
            "mean": float(mean),
            "std": float(sd)
        }
    }

# ----------------
# API routes
# ----------------
@app.get("/ai/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/ai/predict-cashflow")
def predict_cashflow():
    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history", [])
    horizon = int(data.get("horizon", 3))
    alpha = float(data.get("alpha", 0.3))

    # validation
    if not isinstance(history, (list, tuple)):
        return jsonify({"error": "history must be an array"}), 400
    try:
        history = [float(x) for x in history][-360:]  # cap history length
    except Exception:
        return jsonify({"error": "history values must be numbers"}), 400
    horizon = max(1, min(24, int(horizon)))
    alpha = max(0.0, min(1.0, float(alpha)))

    result = compute_forecast(history, horizon=horizon, alpha=alpha)
    return jsonify(result), 200

@app.post("/ai/detect-risk")
def detect_risk():
    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history", [])
    cash_on_hand = float(data.get("cash_on_hand", 0))
    avg_expense = float(data.get("avg_expense", 0))

    if not isinstance(history, (list, tuple)):
        return jsonify({"error": "history must be an array"}), 400
    try:
        history = [float(x) for x in history][-360:]
    except Exception:
        return jsonify({"error": "history values must be numbers"}), 400

    result = compute_risk(history, cash_on_hand=cash_on_hand, avg_expense=avg_expense)
    return jsonify(result), 200

@app.post("/ai/predict-and-detect")
def predict_and_detect():
    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history", [])
    horizon = int(data.get("horizon", 3))
    alpha = float(data.get("alpha", 0.3))
    cash_on_hand = float(data.get("cash_on_hand", 0))
    avg_expense = float(data.get("avg_expense", 0))

    # sanitize once
    if not isinstance(history, (list, tuple)):
        return jsonify({"error": "history must be an array"}), 400
    try:
        history = [float(x) for x in history][-360:]
    except Exception:
        return jsonify({"error": "history values must be numbers"}), 400
    horizon = max(1, min(24, int(horizon)))
    alpha = max(0.0, min(1.0, float(alpha)))

    prediction = compute_forecast(history, horizon=horizon, alpha=alpha)
    risk = compute_risk(history, cash_on_hand=cash_on_hand, avg_expense=avg_expense)
    return jsonify({"prediction": prediction, "risk": risk}), 200

if __name__ == "__main__":
    # For local dev; cPanel will use passenger_wsgi.py
    app.run(host="0.0.0.0", port=5000, debug=False)
