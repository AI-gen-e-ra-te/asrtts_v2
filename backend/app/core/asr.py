import os
from faster_whisper import WhisperModel

# 模型大小：base, small, medium, large-v3
# 建议先用 base 测试，速度快
MODEL_SIZE = "base" 
# 强制优先使用 CUDA
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading Whisper model ({MODEL_SIZE}) on {device}...")
try:
    # 根据设备选择合适的compute_type
    if device == "cuda":
        compute_type = "int8"  # CUDA上使用int8加速
    else:
        compute_type = "int8_float32"  # CPU上使用int8_float32或float32

    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
    print(f"[OK] Whisper model loaded with compute_type={compute_type}.")
except Exception as e:
    print(f"[ERROR] Failed to load Whisper: {e}")
    # 尝试使用默认compute_type
    try:
        model = WhisperModel(MODEL_SIZE, device=device)
        print("[OK] Whisper model loaded with default compute_type.")
    except Exception as e2:
        print(f"[ERROR] Fallback also failed: {e2}")
        model = None

def transcribe_audio(file_path: str) -> str:
    if not model:
        return "Error: ASR model not loaded."

    # 优化识别参数以提高灵敏度
    # 根据日志调整：log_prob_threshold从-1.0降到-2.0，no_speech_threshold从0.6升到0.7
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        language="zh",
        no_speech_threshold=0.7,  # 提高阈值，减少误判为无语音
        log_prob_threshold=-2.0,  # 降低阈值，接受更多低置信度音频
        condition_on_previous_text=False,  # 短语音不需要上下文
        vad_filter=True,  # 启用VAD过滤，改善语音检测
        vad_parameters=dict(min_silence_duration_ms=500)  # VAD参数
    )

    text = ""
    for segment in segments:
        text += segment.text

    # 记录识别结果
    if text.strip():
        print(f"[OK] ASR识别成功: '{text.strip()}'")
    else:
        print(f"[WARN] ASR未识别到语音内容")

    return text.strip()