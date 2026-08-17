import os, io
import numpy as np
import librosa
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CẤU HÍNH (Đúng như durian.py) ---
FS = 44100; DURATION = 1.0; PROCESS_DURATION = 0.7 
N_MFCC = 30; MAX_PAD_LEN = 100; USE_DELTAS = True 

# --- 1. TẢI MODEL TFLITE ---
print("Đang tải model TFLite...")
interpreter = tflite.Interpreter(model_path="sound_model_hybrid.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tìm index chính xác của 2 đầu vào
input_audio_idx = -1
input_phys_idx = -1
for i, d in enumerate(input_details):
    if 'audio' in d['name'].lower(): input_audio_idx = i
    elif 'physics' in d['name'].lower(): input_phys_idx = i

# --- 2. LẤY NHÃN & STATS (Bám sát logic class DurianH5Model trong durian.py) ---
labels = np.load("label_classes.npy", allow_pickle=True)
labels_mapping = {idx: str(label).lower().strip() for idx, label in enumerate(labels)}
NOISE_LABELS = {v for v in labels_mapping.values() if v not in ('dutuoi', 'xanh', 'du_tuoi')}

stats = np.load("mfcc_stats.npz")
mean = stats["mean"].squeeze()
std = stats["std"].squeeze()
phys_mean = stats["phys_mean"].squeeze()
phys_scale = stats["phys_scale"].squeeze() # Tương đương phys_std

print(f"✅ Model loaded | Classes: {list(labels_mapping.values())}")

# --- 3. FEATURE EXTRACTION ---
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

# Dùng hàm Physics chuẩn của TrainCNN_v12 (Đã fix lỗi thời gian suy hao)
def compute_physics_features(y, sr, hop_length=512):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_max = np.max(rms)
    if rms_max < 1e-6: return np.array([0.0] * 6, dtype=np.float32)
        
    peak_idx = int(np.argmax(rms))
    threshold = 0.1 * rms_max
    
    try: start_idx = np.where(rms > threshold)[0][0]
    except IndexError: start_idx = 0
    if peak_idx < start_idx: peak_idx = start_idx
    attack_time = _frames_to_seconds(peak_idx - start_idx, hop_length, sr)
    
    rms_after_peak = rms[peak_idx:]
    decay_idx_rel = np.where(rms_after_peak < threshold)[0]
    n_decay_frames = decay_idx_rel[0] if len(decay_idx_rel) > 0 else len(rms_after_peak)
    decay_time = _frames_to_seconds(n_decay_frames, hop_length, sr)
    
    At = rms_after_peak[-1] + 1e-6
    decay_span_sec = _frames_to_seconds(len(rms_after_peak), hop_length, sr)
    beta = np.log(rms_max / At) / decay_span_sec if decay_span_sec > 0 else 0.0
    
    pad_samples = int(0.05 * sr)
    start_sample = max(0, (peak_idx * hop_length) - pad_samples)
    end_sample = min(len(y), (peak_idx * hop_length) + pad_samples + hop_length)
    y_segment = y[start_sample:end_sample]
    
    fft = np.abs(np.fft.rfft(y_segment))
    window_size = 7
    fft_smooth = np.convolve(fft, np.ones(window_size) / window_size, mode='same') if len(fft) > window_size else fft
    freqs = np.fft.rfftfreq(len(y_segment), 1 / sr)
    res_freq = freqs[np.argmax(fft_smooth[1:]) + 1] if len(fft_smooth) > 1 else 0.0

    valid_frames_mask = rms > threshold
    spec_bw_mean = 0.0
    rolloff_mean = 0.0
    if np.any(valid_frames_mask):
        spec_bw_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0][valid_frames_mask])
        rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0][valid_frames_mask])

    return np.array([res_freq, decay_time, beta, spec_bw_mean, attack_time, rolloff_mean], dtype=np.float32)

# Giữ nguyên logic xử lý ma trận MFCC chuẩn của durian.py
def extract_mfcc_from_array(y, sr, mean, std):
    try:
        L = int(sr * PROCESS_DURATION)
        y = np.pad(y, (0, L - len(y)), mode='constant') if len(y) < L else y[:L]
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        if USE_DELTAS:
            mf = np.vstack([mf, librosa.feature.delta(mf), librosa.feature.delta(mf, order=2)])
        mf = np.vstack([mf, librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MFCC))])
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if onset_env.ndim == 1: onset_env = onset_env.reshape(1, -1)
        
        transient = np.vstack([
            librosa.feature.zero_crossing_rate(y), librosa.feature.rms(y=y),
            librosa.feature.spectral_centroid(y=y, sr=sr), onset_env
        ])

        T = min(transient.shape[1], mf.shape[1])
        if T == 0: T = 1; transient = np.zeros((4, 1)); mf = mf[:, :1]
        else: transient = transient[:, :T]; mf = mf[:, :T]
        mf = np.vstack([mf, transient])

        if mf.shape[1] < MAX_PAD_LEN: mf = np.pad(mf, ((0, 0), (0, MAX_PAD_LEN - mf.shape[1])), mode='constant')
        else: mf = mf[:, :MAX_PAD_LEN]

        if mean.ndim == 1: 
            if mean.shape[0] < mf.shape[0]:
                pad_len = mf.shape[0] - mean.shape[0]
                mean_vec = np.concatenate([mean, np.zeros(pad_len)], axis=0)
                std_vec = np.concatenate([std, np.ones(pad_len)], axis=0)
            elif mean.shape[0] > mf.shape[0]:
                mean_vec, std_vec = mean[:mf.shape[0]], std[:mf.shape[0]]
            else: mean_vec, std_vec = mean, std
            feat_norm = (mf - mean_vec[:, np.newaxis]) / (std_vec[:, np.newaxis] + 1e-10)
        else: feat_norm = mf
        return feat_norm.astype(np.float32)
    except Exception as e:
        print("MFCC error:", e)
        return None

# --- 4. LOGIC RA QUYẾT ĐỊNH (Chuyển đổi chuẩn xác từ _make_decision của durian.py) ---
def make_decision(label, conf, rms, rms_thresh, dt_thresh, xn_thresh):
    # conf và dt_thresh/xn_thresh đều đang ở dạng % (VD: 65.0 và 50.0)
    if label in ("dutuoi", "du_tuoi"):
        if rms <= rms_thresh: return "MỜI GÕ LẠI", "gray"
        if conf >= dt_thresh: return "ĐỦ TUỔI", "#e67e22"
        return "MỜI GÕ LẠI", "gray"
    elif label == "xanh":
        if rms <= rms_thresh: return "MỜI GÕ LẠI", "gray"
        if conf >= xn_thresh: return "XANH", "#27ae60"
        return "MỜI GÕ LẠI", "gray"
    else:
        return "NOISE", "#3498db"

# --- 5. API DỰ ĐOÁN (Giống flow predict() của durian.py) ---
@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
    rms_thresh: float = Form(0.0025),
    dt_thresh: float = Form(50.0),
    xn_thresh: float = Form(35.0)
):
    try:
        audio_bytes = await file.read()
        
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            y, sr = librosa.load(wav_buffer, sr=FS) 
        except Exception as e:
            return JSONResponse({"error": f"Lỗi đọc âm thanh: {str(e)}"})

        raw_rms = float(np.sqrt(np.mean(y**2)))
        audio_model = preprocess_for_model(y, FS)
        
        mfcc_features = extract_mfcc_from_array(audio_model, FS, mean, std)
        if mfcc_features is None: return JSONResponse({"error": "Lỗi trích xuất MFCC"})

        physics_raw = compute_physics_features(audio_model, FS)
        if physics_raw.shape[0] != phys_mean.shape[0]: return JSONResponse({"error": "Lỗi shape Physics"})

        phys_norm = (physics_raw - phys_mean) / phys_scale
        if np.any(np.isnan(mfcc_features)) or np.any(np.isnan(phys_norm)): 
            return JSONResponse({"error": "NaN/Inf detected"})

        input_audio = np.expand_dims(mfcc_features, axis=0).astype(np.float32)
        input_physics = np.expand_dims(phys_norm, axis=0).astype(np.float32)
        
        # Dùng TFLite thay vì Keras
        interpreter.set_tensor(input_audio_idx, input_audio)
        interpreter.set_tensor(input_phys_idx, input_physics)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_idx = int(np.argmax(prediction))
        raw_label = labels_mapping.get(predicted_idx, "Unknown")
        confidence = float(np.max(prediction)) * 100.0

        # Gọi hàm quyết định chuẩn của durian.py
        decision_label, decision_color = make_decision(raw_label, confidence, raw_rms, rms_thresh, dt_thresh, xn_thresh)
        
        # Trả về kết quả giống hệt format trả về của durian.py
        return {
            'raw_label': raw_label, 'confidence': confidence,
            'raw_rms': raw_rms, 'decision_label': decision_label, 
            'color': decision_color, 
            'decision': decision_label # Thêm trường này cho App Mobile dễ đọc
        }
    except Exception as e:
        return JSONResponse({"error": str(e)})
