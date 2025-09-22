# app.py
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# --- optional CORS ---
try:
    from flask_cors import CORS
    CORS(app, resources={r"/ai/*": {"origins": "*"}})
except Exception:
    pass

# ===== Dependencies (optional TF) =====
_TF_OK = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    _TF_OK = False

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ---------- utils ----------
def _to_1d_float(arr):
    return [float(x) for x in arr]

def _make_sequences(values, win):
    X, y = [], []
    for i in range(len(values) - win):
        X.append(values[i:i + win])
        y.append(values[i + win])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    return X, y

def _trim_leading_zeros(arr):
    """Potong leading zeros, kembalikan (trimmed, offset)."""
    i0 = 0
    for i, v in enumerate(arr):
        if float(v) != 0.0:
            i0 = i
            break
    else:
        # semua nol
        return list(arr), 0
    return list(arr[i0:]), i0

# ---------- LSTM forecast with fitted ----------
def forecast_lstm(history, horizon=3, window=3, epochs=50, batch_size=16, verbose=0):
    """
    Univariate LSTM:
    - Scaling MinMax
    - Sliding window
    - Kembalikan: forecast, fitted (dipad ke panjang history), backtest (y_hat untuk setiap jendela)
    """
    hist_in = list(_to_1d_float(history))
    # trimming nol di depan (agar training tidak didominasi 0)
    hist_trim, offset = _trim_leading_zeros(hist_in)

    hist = np.array(hist_trim, dtype=float).reshape(-1, 1)
    if len(hist) < max(5, window + 1) or not _TF_OK:
        # Fallback SES
        out = forecast_ses(hist_in, horizon=horizon, alpha=0.3, window=window)
        out["meta"].update({"engine": "lstm_fallback", "tf": _TF_OK, "window": window, "offset": offset})
        return out

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(hist).ravel()

    X, y = _make_sequences(scaled, window)
    if len(X) == 0:
        return {
            "forecast": [0.0] * horizon,
            "fitted": [None] * len(hist_in),
            "backtest": [],
            "meta": {"window": window, "scaled": True, "engine": "lstm", "offset": offset},
        }

    # reshape for LSTM: (samples, timesteps, features)
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))

    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(X_lstm, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # --- fitted/backtest (one-step-ahead untuk setiap jendela training) ---
    yhat_scaled = model.predict(X_lstm, verbose=0).ravel()
    yhat = scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).ravel().tolist()   # panjang = len(history_trim)-window

    # pad ke panjang history_trim: [None]*window + yhat
    fitted_trim = [None] * window + [float(v) for v in yhat]
    # lalu pad ke panjang history asli: tambahkan None di depan sesuai offset
    fitted_full = [None] * offset + fitted_trim
    # jika masih kurang (jarang terjadi), pad kanan None
    if len(fitted_full) < len(hist_in):
        fitted_full += [None] * (len(hist_in) - len(fitted_full))
    elif len(fitted_full) > len(hist_in):
        fitted_full = fitted_full[:len(hist_in)]

    # --- autoregressive multi-step forecast ---
    last_seq = scaled[-window:].tolist()
    preds_scaled = []
    base_seq = last_seq[:]
    for _ in range(horizon):
        x = np.array(base_seq[-window:], dtype=float).reshape(1, window, 1)
        p = float(model.predict(x, verbose=0).ravel()[0])
        preds_scaled.append(p)
        base_seq.append(p)

    preds = scaler.inverse_transform(np.array(preds_scaled, dtype=float).reshape(-1, 1)).ravel().tolist()

    return {
        "forecast": [float(v) for v in preds],
        "fitted": [None if v is None else float(v) for v in fitted_full],
        "backtest": [float(v) for v in yhat],  # tanpa pad, ukuran = len(history_trim)-window
        "meta": {"window": window, "epochs": epochs, "engine": "lstm", "scaled": True, "offset": offset},
    }

# ---------- SES fallback (ringan) ----------
def forecast_ses(history, horizon=3, alpha=0.3, window=3):
    arr = np.array(_to_1d_float(history), dtype=float)
    if arr.size == 0:
        return {"forecast": [0.0] * horizon, "fitted": [], "backtest": [], "meta": {"engine": "ses"}}
    # smoothing dan fitted
    fitted = [None] * window
    s = arr[0]
    for i in range(1, arr.size):
        s = alpha * arr[i-1] + (1 - alpha) * s  # one-step-ahead untuk titik i
        fitted.append(float(s))
    # forecast konstan-ish
    s_final = alpha * arr[-1] + (1 - alpha) * s
    preds = [float(0.8 * s_final + 0.2 * arr.mean()) for _ in range(horizon)]
    return {
        "forecast": preds,
        "fitted": [None if v is None else float(v) for v in fitted[:len(arr)]],
        "backtest": [],  # bisa diisi sama dengan fitted[window:] bila mau
        "meta": {"alpha": alpha, "engine": "ses", "window": window},
    }

# ---------- Isolation Forest ----------
def detect_iforest(values, contamination="auto", random_state=42):
    x = np.array(_to_1d_float(values), dtype=float).reshape(-1, 1)
    if x.size == 0:
        return {"anomalies": [], "scores": []}
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(x)
    pred = clf.predict(x)          # -1 anomaly, 1 normal
    score = clf.decision_function(x).tolist()
    anomalies = []
    for i, (p, s) in enumerate(zip(pred, score)):
        if p == -1:
            anomalies.append({"index": i, "value": float(x[i][0]), "score": float(s)})
    return {"anomalies": anomalies, "scores": [float(s) for s in score]}

# ---------- API ----------
@app.get("/ai/health")
def health():
    return jsonify({"status": "ok", "tf": _TF_OK}), 200

@app.post("/ai/predict-lstm")
def api_predict_lstm():
    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history", [])
    horizon = int(data.get("horizon", 3))
    window = int(data.get("window", 3))
    epochs = int(data.get("epochs", 60))
    try:
        history = _to_1d_float(history)[-360:]
    except Exception:
        return jsonify({"error": "history must be numeric array"}), 400
    horizon = max(1, min(24, horizon))
    window = max(2, min(12, window))
    epochs = max(10, min(200, epochs))

    if _TF_OK:
        out = forecast_lstm(history, horizon=horizon, window=window, epochs=epochs)
    else:
        out = forecast_ses(history, horizon=horizon, alpha=0.3, window=window)
    return jsonify(out), 200

@app.post("/ai/detect-anomaly-iforest")
def api_detect_iforest():
    data = request.get_json(force=True, silent=True) or {}
    series = data.get("series", [])
    contamination = data.get("contamination", "auto")
    try:
        series = _to_1d_float(series)[-360:]
    except Exception:
        return jsonify({"error": "series must be numeric array"}), 400
    out = detect_iforest(series, contamination=contamination)
    return jsonify(out), 200

@app.post("/ai/predict-and-detect")
def api_predict_and_detect():
    """
    - history   : deret target (per bulan)
    - outflows  : deret yang dipakai untuk anomali
    - horizon   : langkah prediksi ke depan
    """
    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history", [])
    outflows = data.get("outflows", [])
    horizon = int(data.get("horizon", 3))

    try:
        history = _to_1d_float(history)[-360:]
        outflows = _to_1d_float(outflows)[-360:]
    except Exception:
        return jsonify({"error": "history/outflows must be numeric arrays"}), 400

    # Forecast (+ fitted/backtest)
    if _TF_OK:
        pred = forecast_lstm(history, horizon=horizon, window=3, epochs=60)
    else:
        pred = forecast_ses(history, horizon=horizon, alpha=0.3, window=3)

    # Deteksi anomali
    det = detect_iforest(outflows, contamination="auto")

    return jsonify({"prediction": pred, "anomaly": det}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
