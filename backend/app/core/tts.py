import os
import sys
import base64
import io
import asyncio
import torch
import soundfile as sf
from pathlib import Path

# Disable torchcodec before importing torchaudio
# This forces torchaudio to use soundfile backend
os.environ.setdefault('TORCHAUDIO_USE_BACKEND_DISPATCHER', '0')

import torchaudio

# Monkey-patch torchaudio.load to handle torchcodec errors
_original_torchaudio_load = torchaudio.load

def patched_torchaudio_load(filepath, **kwargs):
    """Patched torchaudio.load that falls back to soundfile if torchcodec fails"""
    # Force soundfile backend if not specified
    if 'backend' not in kwargs:
        kwargs['backend'] = 'soundfile'
    
    try:
        return _original_torchaudio_load(filepath, **kwargs)
    except (ImportError, RuntimeError, OSError) as e:
        error_str = str(e).lower()
        if 'torchcodec' in error_str or 'ffmpeg' in error_str or 'libtorchcodec' in error_str:
            # Fallback to direct soundfile loading
            speech, sample_rate = sf.read(filepath)
            speech = torch.from_numpy(speech).float()
            
            # soundfile returns (samples,) for mono or (samples, channels) for multi-channel
            # torchaudio expects (channels, samples)
            if len(speech.shape) == 1:
                # Mono: (samples,) -> (1, samples)
                speech = speech.unsqueeze(0)
            elif len(speech.shape) == 2:
                # Multi-channel: (samples, channels) -> (channels, samples)
                speech = speech.T
            
            return speech, sample_rate
        else:
            raise

torchaudio.load = patched_torchaudio_load

# Add CosyVoice to path
backend_path = Path(__file__).parent.parent.parent
cosyvoice_path = backend_path / "CosyVoice"
if cosyvoice_path.exists():
    sys.path.insert(0, str(cosyvoice_path))
    sys.path.insert(0, str(cosyvoice_path / "third_party" / "Matcha-TTS"))

# Patch load_wav to handle torchcodec errors before importing CosyVoice
def patch_cosyvoice_load_wav():
    """Patch CosyVoice's load_wav function to use soundfile directly if torchcodec fails"""
    try:
        from cosyvoice.utils import file_utils
        
        original_load_wav = file_utils.load_wav
        
        def patched_load_wav(wav, target_sr, min_sr=16000):
            """Patched load_wav that uses soundfile directly if torchaudio fails"""
            try:
                # Try original method first
                return original_load_wav(wav, target_sr, min_sr)
            except (ImportError, RuntimeError, OSError) as e:
                error_str = str(e).lower()
                if 'torchcodec' in error_str or 'ffmpeg' in error_str or 'libtorchcodec' in error_str:
                    # Fallback to direct soundfile loading
                    try:
                        speech, sample_rate = sf.read(wav)
                        speech = torch.from_numpy(speech).float()
                        
                        # Handle stereo to mono conversion
                        if len(speech.shape) > 1:
                            speech = speech.mean(dim=-1)
                        speech = speech.unsqueeze(0)  # Add channel dimension
                        
                        # Resample if needed
                        if sample_rate != target_sr:
                            assert sample_rate >= min_sr, f'wav sample rate {sample_rate} must be greater than {min_sr}'
                            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                            speech = resampler(speech)
                        
                        return speech
                    except Exception as fallback_error:
                        print(f"⚠️  Soundfile fallback also failed: {fallback_error}")
                        raise e  # Re-raise original error
                else:
                    raise
        
        # Apply the patch
        file_utils.load_wav = patched_load_wav
        return True
    except Exception as e:
        print(f"⚠️  Could not patch load_wav: {e}")
        return False

try:
    from cosyvoice.cli.cosyvoice import AutoModel
    # Patch after import
    patch_cosyvoice_load_wav()
except ImportError as e:
    print(f"⚠️  CosyVoice not available: {e}")
    print("Please ensure CosyVoice is cloned and dependencies are installed.")
    AutoModel = None

# Model configuration
MODEL_DIR_ENV = os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
# Resolve model directory relative to backend
if not os.path.isabs(MODEL_DIR_ENV):
    MODEL_DIR = str(backend_path / MODEL_DIR_ENV)
else:
    MODEL_DIR = MODEL_DIR_ENV
SPEAKER_ID = os.getenv("COSYVOICE_SPEAKER_ID", "中文女")  # Default speaker for SFT model
USE_SFT = os.getenv("COSYVOICE_USE_SFT", "false").lower() == "true"
# 实际使用的模式（SFT可能回退到zero-shot）
_SFT_AVAILABLE = False

# Check if model directory exists
if not os.path.exists(MODEL_DIR):
    print(f"⚠️ WARNING: CosyVoice model directory not found: {MODEL_DIR}")
    print("   Please download the model or set COSYVOICE_MODEL_DIR environment variable.")
    # Try alternative paths
    alternative_path = str(backend_path / "CosyVoice" / "pretrained_models")
    if os.path.exists(alternative_path):
        print(f"   Trying alternative path: {alternative_path}")
        MODEL_DIR = alternative_path
    else:
        print("   No alternative paths found. TTS will not work until model is downloaded.")

# Global model instance
_model = None
_model_lock = asyncio.Lock()
_SFT_SPEAKER_ID = SPEAKER_ID  # 默认使用环境变量配置的说话人

def _load_model():
    """Load CosyVoice model (synchronous)"""
    global _model, _SFT_AVAILABLE, _SFT_SPEAKER_ID
    if _model is None:
        if AutoModel is None:
            raise ImportError("CosyVoice AutoModel is not available. Please install CosyVoice dependencies.")
        print(f"🔄 Loading CosyVoice model from {MODEL_DIR}...")
        try:
            if USE_SFT:
                # Try SFT model first (faster if available)
                try:
                    model_path = MODEL_DIR if "SFT" in MODEL_DIR else MODEL_DIR.replace("Fun-CosyVoice3-0.5B", "CosyVoice-300M-SFT")
                    _model = AutoModel(model_dir=model_path)

                    # 检查是否有可用的说话人
                    available_speakers = _model.list_available_spks()
                    print(f"✅ CosyVoice SFT model loaded. Available speakers: {available_speakers}")

                    if available_speakers:
                        _SFT_AVAILABLE = True
                        # 选择有效的说话人ID
                        if SPEAKER_ID in available_speakers:
                            _SFT_SPEAKER_ID = SPEAKER_ID
                        else:
                            _SFT_SPEAKER_ID = available_speakers[0]
                            print(f"⚠️ Speaker ID '{SPEAKER_ID}' not in available speakers. Using '{_SFT_SPEAKER_ID}' instead.")
                        print(f"✅ Using speaker ID: {_SFT_SPEAKER_ID}")
                    else:
                        print(f"⚠️ SFT model loaded but no speakers available. This may be a zero-shot model. Falling back to zero-shot mode.")
                        # 重新加载为零样本模型
                        _model = AutoModel(model_dir=MODEL_DIR)
                        _SFT_AVAILABLE = False
                        print(f"✅ CosyVoice zero-shot model loaded (fallback).")
                except Exception as sft_error:
                    print(f"⚠️ SFT model loading failed: {sft_error}, falling back to zero-shot mode")
                    # Fallback to zero-shot
                    _model = AutoModel(model_dir=MODEL_DIR)
                    _SFT_AVAILABLE = False
                    print(f"✅ CosyVoice zero-shot model loaded (fallback).")
            else:
                # Use zero-shot inference (CosyVoice3 recommended)
                _model = AutoModel(model_dir=MODEL_DIR)
                _SFT_AVAILABLE = False
                print(f"✅ CosyVoice model loaded.")
        except Exception as e:
            print(f"❌ Failed to load CosyVoice model: {e}")
            raise
    return _model

async def _get_model():
    """Get or load model (async-safe)"""
    async with _model_lock:
        if _model is None:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _load_model)
        return _model

async def text_to_speech(text: str) -> str:
    """
    Convert text to speech using CosyVoice.
    
    Args:
        text: Text to synthesize
        
    Returns:
        Base64-encoded WAV audio data, or empty string on error
    """
    if not text or not text.strip():
        return ""
    
    try:
        model = await _get_model()
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        if _SFT_AVAILABLE:
            # Use SFT inference (faster, requires speaker ID)
            def _synthesize_sft():
                audio_chunks = []
                # 使用全局的_SFT_SPEAKER_ID
                speaker_id = _SFT_SPEAKER_ID if _SFT_SPEAKER_ID else "default"
                try:
                    for result in model.inference_sft(text, speaker_id, stream=False):
                        audio_chunks.append(result['tts_speech'])
                except KeyError as e:
                    print(f"❌ SFT synthesis failed with speaker ID '{speaker_id}': {e}")
                    # 尝试使用第一个可用的说话人（如果不同）
                    available_speakers = model.list_available_spks()
                    if available_speakers and speaker_id != available_speakers[0]:
                        print(f"⚠️ Trying alternative speaker: {available_speakers[0]}")
                        audio_chunks = []
                        for result in model.inference_sft(text, available_speakers[0], stream=False):
                            audio_chunks.append(result['tts_speech'])
                    else:
                        raise

                if audio_chunks:
                    # Concatenate all chunks
                    audio = torch.cat(audio_chunks, dim=1)
                    return audio, model.sample_rate
                return None, None

            audio, sample_rate = await loop.run_in_executor(None, _synthesize_sft)
        else:
            # Use zero-shot inference (CosyVoice3 recommended)
            # For zero-shot, we need a prompt text and prompt audio
            # Using optimized shorter prompt for better performance
            prompt_text = "你好。"  # 更短的提示文本
            prompt_wav = str(cosyvoice_path / "asset" / "zero_shot_prompt.wav")
            
            # Fallback if prompt file doesn't exist
            if not os.path.exists(prompt_wav):
                prompt_wav = None
            
            def _synthesize_zero_shot():
                audio_chunks = []
                # CosyVoice3 requires <|endofprompt|> token
                full_prompt = f"You are an assistant.<|endofprompt|>{prompt_text}"  # 更短的系统提示
                
                if prompt_wav and os.path.exists(prompt_wav):
                    for result in model.inference_zero_shot(
                        text, 
                        full_prompt, 
                        prompt_wav, 
                        stream=False
                    ):
                        audio_chunks.append(result['tts_speech'])
                else:
                    # Fallback: try without prompt audio (may not work for all models)
                    try:
                        for result in model.inference_zero_shot(
                            text,
                            full_prompt,
                            "",
                            stream=False
                        ):
                            audio_chunks.append(result['tts_speech'])
                    except:
                        # Last resort: try SFT if available
                        if hasattr(model, 'inference_sft'):
                            speakers = model.list_available_spks()
                            speaker = speakers[0] if speakers else "中文女"
                            for result in model.inference_sft(text, speaker, stream=False):
                                audio_chunks.append(result['tts_speech'])
                
                if audio_chunks:
                    audio = torch.cat(audio_chunks, dim=1)
                    return audio, model.sample_rate
                return None, None
            
            audio, sample_rate = await loop.run_in_executor(None, _synthesize_zero_shot)
        
        if audio is None:
            print("❌ TTS: No audio generated")
            return ""
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sample_rate, format="wav")
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        # Encode to base64
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        return audio_base64
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        import traceback
        traceback.print_exc()
        return ""

