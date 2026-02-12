import base64
import io
import os
import joblib
import numpy as np
import librosa
import scipy.stats as stats
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import tempfile
import os

# ==============================
# CONFIGURATION
# ==============================

API_KEY = os.environ.get("API_KEY")  # MUST set on Render

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ==============================
# LOAD TRAINED MODEL ARTIFACTS
# ==============================

model = joblib.load("voice_auth_model.pkl")
scaler = joblib.load("voice_auth_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
feature_stats = joblib.load("feature_stats.pkl")

# ==============================
# FASTAPI INIT
# ==============================

app = FastAPI(title="AI Voice Authenticity API")

# ==============================
# HEALTH CHECK ROUTE
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
# DSP FEATURE EXTRACTION
# ==============================

def extract_features_from_bytes(audio_bytes):
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        # Load using librosa from file path
        y, sr = librosa.load(temp_path, sr=None)

        # Remove temp file
        os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

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

    return {
        "duration": len(y)/sr,
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

# ==============================
# STATISTICALLY ANCHORED EXPLANATION
# ==============================

def generate_statistical_explanation(features, classification):

    deviations = []

    for feature in feature_columns:
        mean = feature_stats[feature]["mean"]
        std = feature_stats[feature]["std"]

        if std == 0:
            continue

        z_score = (features[feature] - mean) / std

        if abs(z_score) > 1.5:
            direction = "high" if z_score > 0 else "low"
            deviations.append(f"{direction} {feature.replace('_', ' ')}")

    if not deviations:
        return "Signal characteristics fall within learned distribution ranges."

    top_features = ", ".join(deviations[:3])

    if classification == "AI_GENERATED":
        return f"Statistical deviation detected with {top_features}, consistent with synthetic signal behavior."
    else:
        return f"Natural variability observed with {top_features}, consistent with human acoustic patterns."

# ==============================
# MAIN ENDPOINT
# ==============================

@app.post("/api/voice-detection")
def detect_voice(request: VoiceRequest, x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    features = extract_features_from_bytes(audio_bytes)

    sample_vector = np.array([features[col] for col in feature_columns]).reshape(1, -1)
    sample_scaled = scaler.transform(sample_vector)

    probabilities = model.predict_proba(sample_scaled)[0]
    prediction = model.predict(sample_scaled)[0]

    ai_probability = float(probabilities[1])
    human_probability = float(probabilities[0])

    classification = "AI_GENERATED" if prediction == 1 else "HUMAN"
    confidence = round(float(max(ai_probability, human_probability)), 2)

    explanation = generate_statistical_explanation(features, classification)

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
