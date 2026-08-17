import os, io
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite

app = FastAPI()

# CHO PHÉP APP MOBILE GỌI ĐẾN SERVER
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CẤU HÌNH ---
FS = 44100; PROCESS_DURATION = 0.7; N_MFCC = 30; MAX_PAD_LEN = 100; USE_DELTAS = True

# --- TẢI MODEL TFLITE ---
print("Đang tải model TFLite...")
interpreter = tflite.Interpreter(model_path="sound_model_hybrid.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_audio_idx = next(i for i, d in enumerate(input_details) if 'audio' in d['name'].lower())
input_phys_idx = next(i for i, d in enumerate(input_details) if 'physics' in d['name'].lower())

# --- TẢI STATS VÀ LABELS ---
labels = np.load("label_classes.npy", allow_pickle=True)
labels_mapping = {idx: str(label).lower().strip() for idx, label in enumerate(labels)}
stats = np.load("mfcc_stats.npz")
mean = stats["mean"].squeeze(); std = stats["std"].squeeze()
phys_mean = stats["phys_mean"].squeeze(); phys_scale = stats["phys_scale"].squeeze()

# --- TOÀN BỘ HÀM XỬ LÝ ÂM THANH ---
def fix_length_infer(y, sr, target_duration=PROCESS_DURATION):
    L = int(sr * target_duration)
    return np.pad(y, (0, L - len(y)), mode='constant') if len(y) < L else y[:L]

def normalize_rms(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    return (y / rms * target_rms).astype(np.float32)

def preprocess_for_model(y, sr):
    return normalize_rms(fix_length_infer(y, sr), target_rms=0.1)

def _frames_to_seconds(n_frames, hop_length, sr):
    return float(n_frames) * hop_length / sr

def compute_physics_features(y, sr, hop_length=512):
    fft = np.abs(np.fft.rfft(y)); freqs = np.fft.rfftfreq(len(y), 1/sr)
    window_size = 5
    fft_smooth = np.convolve(fft, np.ones(window_size)/window_size, mode='same') if len(fft) > window_size else fft
    res_freq = freqs[np.argmax(fft_smooth[1:]) + 1] if len(fft_smooth) > 1 else 0.0
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    A0 = np.max(env)
    if A0 < 1e-6: return np.array([0.0]*6, dtype=np.float32)
    peak_idx_env = int(np.argmax(env)); env_after_peak = env[peak_idx_env:]
    threshold = 0.1 * A0
    decay_idx_rel = np.where(env_after_peak < threshold)[0]
    n_decay_frames = decay_idx_rel[0] if len(decay_idx_rel) > 0 else len(env_after_peak)
    decay_time = _frames_to_seconds(n_decay_frames, hop_length, sr)
    At = env_after_peak[-1] + 1e-6
    n_decay_span_frames = len(env) - peak_idx_env
    decay_span_sec = _frames_to_seconds(n_decay_span_frames, hop_length, sr)
    beta = np.log(A0 / At) / decay_span_sec if decay_span_sec > 0 else 0.0
    spec_bw_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0])
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]; rms_max = np.max(rms)
    if rms_max < 1e-6: attack_time = 0.0
    else:
        try: start_idx = np.where(rms > 0.01 * rms_max)[0][0]
        except IndexError: start_idx = 0
        peak_idx_rms = max(np.argmax(rms), start_idx)
        attack_time = _frames_to_seconds(peak_idx_rms - start_idx, hop_length, sr)
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0])
    return np.array([res_freq, decay_time, beta, spec_bw_mean, attack_time, rolloff_mean], dtype=np.float32)

def extract_mfcc_from_array(y, sr, mean, std):
    try:
        L = int(sr * PROCESS_DURATION)
        y = np.pad(y, (0, L - len(y)), mode='constant') if len(y) < L else y[:L]
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        if USE_DELTAS: mf = np.vstack([mf, librosa.feature.delta(mf), librosa.feature.delta(mf, order=2)])
        mf = np.vstack([mf, librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MFCC))])
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if onset_env.ndim == 1: onset_env = onset_env.reshape(1, -1)
        transient = np.vstack([librosa.feature.zero_crossing_rate(y), librosa.feature.rms(y=y), librosa.feature.spectral_centroid(y=y, sr=sr), onset_env])
        T = min(transient.shape[1], mf.shape[1])
        if T == 0: T = 1; transient = np.zeros((4, 1)); mf = mf[:, :1]
        else: transient = transient[:, :T]; mf = mf[:, :T]
        mf = np.vstack([mf, transient])
        if mf.shape[1] < MAX_PAD_LEN: mf = np.pad(mf, ((0, 0), (0, MAX_PAD_LEN - mf.shape[1])), mode='constant')
        else: mf = mf[:, :MAX_PAD_LEN]
        if mean.ndim == 1: 
            if mean.shape[0] < mf.shape[0]: pad_len = mf.shape[0] - mean.shape[0]; mean_vec = np.concatenate([mean, np.zeros(pad_len)]); std_vec = np.concatenate([std, np.ones(pad_len)])
            elif mean.shape[0] > mf.shape[0]: mean_vec, std_vec = mean[:mf.shape[0]], std[:mf.shape[0]]
            else: mean_vec, std_vec = mean, std
            feat_norm = (mf - mean_vec[:, np.newaxis]) / (std_vec[:, np.newaxis] + 1e-10)
        else: feat_norm = mf
        return feat_norm.astype(np.float32)
    except Exception as e: return None

# --- API DỰ ĐOÁN ---
@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
    rms_thresh: float = Form(0.0025),
    dt_thresh: float = Form(50.0),
    xn_thresh: float = Form(35.0)
):
    try:
        audio_bytes = await file.read()
        # THÊM BACKEND="FFMPEG" ĐỂ ĐỌC ĐỊNH DẠNG WEBM TỪ ĐIỆN THOẠI
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=FS, backend="ffmpeg")
        raw_rms = float(np.sqrt(np.mean(y**2)))
        audio_model = preprocess_for_model(y, FS)
        
        mfcc_features = extract_mfcc_from_array(audio_model, FS, mean, std)
        if mfcc_features is None: return JSONResponse({"error": "Lỗi MFCC"})
        
        physics_raw = compute_physics_features(audio_model, FS)
        phys_norm = (physics_raw - phys_mean) / phys_scale
        
        input_audio = np.expand_dims(mfcc_features, axis=0).astype(np.float32)
        input_physics = np.expand_dims(phys_norm, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_audio_idx, input_audio)
        interpreter.set_tensor(input_phys_idx, input_physics)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_idx = int(np.argmax(prediction))
        raw_label = labels_mapping.get(predicted_idx, "Unknown")
        confidence = float(np.max(prediction)) * 100.0

        if raw_label in ("dutuoi", "du_tuoi"):
            if raw_rms <= rms_thresh or confidence < dt_thresh: decision = "MỜI GÕ LẠI"
            else: decision = "ĐỦ TUỔI"
        elif raw_label == "xanh":
            if raw_rms <= rms_thresh or confidence < xn_thresh: decision = "MỜI GÕ LẠI"
            else: decision = "XANH"
        else: decision = "NOISE"

        return {"decision": decision, "confidence": round(confidence,1), "raw_rms": raw_rms, "raw_label": raw_label}
    except Exception as e:
        return JSONResponse({"error": str(e)})
