import base64
import os
import tempfile
import joblib
import numpy as np
import librosa
import scipy.stats as stats
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ==============================
# CONFIG
# ==============================

API_KEY = os.environ.get("API_KEY")
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("voice_auth_model.pkl")
scaler = joblib.load("voice_auth_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ==============================
# FASTAPI INIT
# ==============================

app = FastAPI(title="AI Voice Authenticity API")

# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def health_check():
    return {
        "status": "success",
        "message": "AI Voice Authenticity API is running"
    }


# ==============================
# REQUEST SCHEMA
# ==============================

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(audio_bytes):

    # Save to temp file (CRITICAL FIX)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)
    except Exception:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Invalid or corrupted MP3 file")

    os.remove(tmp_path)

    duration = len(y) / sr

    # Duration Guard
    if duration < 5:
        raise HTTPException(
            status_code=400,
            detail="Audio duration must be between 5 and 15 seconds"
        )

    if duration > 15:
        y = y[: int(15 * sr)]

    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(y_harmonic**2)
    percussive_energy = np.sum(y_percussive**2)
    hpr_ratio = harmonic_energy / (percussive_energy + 1e-6)

    hist, _ = np.histogram(y, bins=100, density=True)
    entropy = stats.entropy(hist + 1e-10)

    dynamic_range = np.max(np.abs(y)) - np.min(np.abs(y))
    energy_modulation = np.var(np.diff(energy))
    temporal_smoothness = np.var(np.diff(y))

    features = {
        "duration": duration,
        "mean_energy": float(np.mean(energy)),
        "energy_variance": float(np.var(energy)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_variance": float(np.var(zcr)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_variance": float(np.var(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "harmonic_percussive_ratio": float(hpr_ratio),
        "entropy": float(entropy),
        "dynamic_range": float(dynamic_range),
        "energy_modulation_variance": float(energy_modulation),
        "temporal_smoothness_index": float(temporal_smoothness)
    }

    return features

# ==============================
# ENDPOINT
# ==============================

@app.post("/api/voice-detection")
def detect_voice(request: VoiceRequest, x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 string")

    features = extract_features(audio_bytes)

    sample_vector = np.array(
        [features[col] for col in feature_columns]
    ).reshape(1, -1)

    sample_scaled = scaler.transform(sample_vector)

    probabilities = model.predict_proba(sample_scaled)[0]
    prediction = model.predict(sample_scaled)[0]

    confidence = float(max(probabilities))
    classification = "AI_GENERATED" if prediction == 1 else "HUMAN"

    explanation = (
        f"{classification} classification supported by statistical "
        f"variance in duration, entropy, and spectral features."
    )

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 4),
        "explanation": explanation
    }



