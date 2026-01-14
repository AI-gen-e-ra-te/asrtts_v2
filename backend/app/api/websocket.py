import os
import json
import base64
import asyncio
import uuid
import subprocess
import tempfile
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.asr import transcribe_audio
from app.core.llm import chat_stream
from app.core.tts import text_to_speech
import soundfile as sf
import io

router = APIRouter()

async def convert_audio_to_wav(input_path: str, output_path: str = None) -> str:
    """
    将音频文件转换为 WAV 格式，支持多种方法
    返回转换后的 WAV 文件路径，如果失败则返回原始文件路径
    """
    if output_path is None:
        output_path = input_path.replace(".webm", ".wav").replace(".mp3", ".wav").replace(".ogg", ".wav")

    # 方法1: 使用本地 ffmpeg 工具
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_ffmpeg = os.path.join(project_root, "tools", "ffmpeg", "ffmpeg.exe")
    ffmpeg_cmd = local_ffmpeg if os.path.exists(local_ffmpeg) else "ffmpeg"

    # 检查输入文件是否存在且大小合理
    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
    else:
        file_size = os.path.getsize(input_path)
        if file_size < 1024:  # 小于1KB的文件可能无效
            print(f"⚠️ Input file too small ({file_size} bytes), may be invalid")

    # 重试机制
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # 尝试使用 ffmpeg 转换
            subprocess.run(
                [ffmpeg_cmd, "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30
            )
            print(f"✅ FFmpeg conversion successful (attempt {attempt+1}/{max_retries}): {input_path} -> {output_path}")
            return output_path
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            print(f"⚠️ FFmpeg conversion failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"❌ All ffmpeg attempts failed")
            else:
                # 等待片刻后重试
                import time
                time.sleep(0.5)

    # 方法2: 尝试使用 pydub (如果可用)
    try:
        # 设置 pydub 使用的 ffmpeg 路径
        import pydub
        # 确保使用我们本地的 ffmpeg
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_ffmpeg_dir = os.path.join(project_root, "tools", "ffmpeg")
        local_ffmpeg = os.path.join(local_ffmpeg_dir, "ffmpeg.exe")
        local_ffprobe = os.path.join(local_ffmpeg_dir, "ffprobe.exe")

        # 设置环境变量和 pydub 配置
        os.environ["PATH"] = local_ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        pydub.AudioSegment.converter = local_ffmpeg
        pydub.AudioSegment.ffprobe = local_ffprobe if os.path.exists(local_ffprobe) else None

        from pydub import AudioSegment
        # 尝试根据扩展名读取
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".webm":
            audio = AudioSegment.from_file(input_path, format="webm")
        elif ext == ".mp3":
            audio = AudioSegment.from_file(input_path, format="mp3")
        else:
            # 尝试自动检测
            audio = AudioSegment.from_file(input_path)

        # 转换为单声道，16000Hz采样率
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        print(f"✅ Pydub conversion successful: {input_path} -> {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠️ Pydub conversion failed: {e}")
        import traceback
        traceback.print_exc()

    # 方法3: 如果原始文件已经是.wav或无法转换，返回原始路径
    # 检查文件是否有效
    if input_path.lower().endswith('.wav'):
        try:
            # 验证WAV文件是否可以读取
            data, samplerate = sf.read(input_path)
            print(f"✅ Using original WAV file: {input_path}")
            return input_path
        except Exception as e:
            print(f"❌ WAV file validation failed: {e}")

    # 方法4: 尝试使用 torchaudio (如果可用)
    try:
        import torchaudio
        # 使用 torchaudio 加载并保存为 WAV
        waveform, sample_rate = torchaudio.load(input_path)
        # 转换为单声道（如果需要）
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 重采样到 16000 Hz（如果需要）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        # 保存为 WAV
        torchaudio.save(output_path, waveform, 16000)
        print(f"✅ Torchaudio conversion successful: {input_path} -> {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠️ Torchaudio conversion failed: {e}")
        import traceback
        traceback.print_exc()

    # 所有方法都失败
    print(f"❌ All audio conversion methods failed for: {input_path}")
    return None

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    处理全双工语音对话的 WebSocket 端点
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8] # 给每个连接生成一个短ID方便日志查看
    print(f"🔌 Client connected: {client_id}")

    # 用于暂存接收到的音频切片
    audio_buffer = bytearray()

    # 对话历史管理（维护上下文）
    message_history = [
        {"role": "system", "content": "You are a helpful voice assistant. Please keep your replies concise, short, and conversational suitable for TTS."}
    ]
    MAX_HISTORY_TURNS = 10  # 最多保留10轮对话（20条消息）

    def add_to_history(role: str, content: str):
        """添加消息到历史，并修剪超过限制的旧消息"""
        message_history.append({"role": role, "content": content})
        # 保留最近的 MAX_HISTORY_TURNS*2 条消息（每轮对话2条）
        while len(message_history) > MAX_HISTORY_TURNS * 2 + 1:  # +1 为system消息
            # 删除最旧的用户/助手消息（跳过system消息）
            if len(message_history) > 1:
                removed = message_history.pop(1)  # 移除system之后的第一条消息
                print(f"📝 [{client_id}] Removed old message from history: {removed['role']}")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                print(f"❌ [{client_id}] Invalid JSON received")
                continue

            if "type" not in message:
                print(f"❌ [{client_id}] Message missing 'type' field")
                continue
            
            if message["type"] == "audio-chunk":
                chunk = base64.b64decode(message["content"])
                audio_buffer.extend(chunk)
            
            elif message["type"] == "text-input":
                # 处理文本输入
                user_text = message.get("content", "").strip()
                if not user_text:
                    continue

                print(f"👤 [{client_id}] User text: {user_text}")

                # 发送用户消息给前端
                await websocket.send_json({
                    "type": "user-message",
                    "content": user_text
                })

                # 添加用户消息到对话历史
                add_to_history("user", user_text)

                # 通知前端处理中
                await websocket.send_json({"type": "status", "content": "processing"})

                # 处理LLM响应（使用完整的对话历史）
                sentence_buffer = ""
                full_response = ""  # 收集完整回复以便添加到历史
                punctuation = {",", "，", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", "\n"}

                try:
                    async for char in chat_stream(message_history):
                        # 实时推流文字
                        await websocket.send_json({"type": "text-update", "content": char})

                        sentence_buffer += char
                        full_response += char

                        # 断句
                        if char in punctuation:
                            if len(sentence_buffer.strip()) > 1:
                                print(f"🗣️ [{client_id}] Synthesizing: {sentence_buffer}")
                                audio_base64 = await text_to_speech(sentence_buffer)

                                if audio_base64:
                                    await websocket.send_json({
                                        "type": "audio-chunk",
                                        "content": audio_base64
                                    })
                                sentence_buffer = ""

                    # 处理剩余文本
                    if sentence_buffer.strip():
                         print(f"🗣️ [{client_id}] Synthesizing (Final): {sentence_buffer}")
                         audio_base64 = await text_to_speech(sentence_buffer)
                         if audio_base64:
                            await websocket.send_json({
                                "type": "audio-chunk",
                                "content": audio_base64
                            })

                    # 将助手回复添加到对话历史
                    if full_response.strip():
                        add_to_history("assistant", full_response.strip())
                        print(f"📝 [{client_id}] Added assistant response to history ({len(full_response)} chars)")

                except Exception as e:
                    print(f"❌ LLM/TTS Process Error: {e}")
                    await websocket.send_json({"type": "text-update", "content": f"\n[Error: {str(e)}]"})

                await websocket.send_json({"type": "status", "content": "idle"})
            
            elif message["type"] == "audio-end":
                # 生成唯一文件名并保存
                request_id = str(uuid.uuid4())
                temp_audio_path = f"temp_input_{request_id}.webm"

                # 检查音频数据是否有效（最小长度检查）
                if len(audio_buffer) < 1024:  # 至少1KB的音频数据
                    print(f"⚠️ Audio buffer too small ({len(audio_buffer)} bytes), skipping ASR")
                    # 清空缓冲区并跳过
                    audio_buffer = bytearray()
                    await websocket.send_json({"type": "status", "content": "idle"})
                    continue

                # 写入文件
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_buffer)

                # 清空缓冲区
                audio_buffer = bytearray()

                # 通知前端
                await websocket.send_json({"type": "status", "content": "processing"})

                # 转换音频格式 (WebM -> WAV) 以解决 EBML header parsing failed 问题
                # 浏览器录制的 WebM 有时没有完整的 Header，使用多重备选方案
                wav_path = temp_audio_path.replace(".webm", ".wav")

                # 使用增强的音频转换函数
                asr_input_path = await convert_audio_to_wav(temp_audio_path, wav_path)

                if asr_input_path is None:
                    print(f"❌ All audio conversion methods failed, skipping ASR")
                    # 清理临时文件
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    audio_buffer = bytearray()
                    await websocket.send_json({"type": "status", "content": "idle"})
                    continue

                # ASR
                try:
                    # 使用 asyncio.to_thread 运行同步的 Whisper 识别
                    user_text = await asyncio.to_thread(transcribe_audio, asr_input_path)
                    print(f"👂 [{client_id}] User said: {user_text}")
                except Exception as e:
                    print(f"❌ ASR Error: {e}")
                    user_text = ""
                
                # 清理临时文件
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)

                # 如果没听到说话，直接跳过
                if not user_text.strip():
                    await websocket.send_json({"type": "status", "content": "idle"})
                    continue

                # 发送用户消息给前端（使用新的消息类型）
                await websocket.send_json({
                    "type": "user-message",
                    "content": user_text
                })

                # 添加用户消息到对话历史
                add_to_history("user", user_text)

                sentence_buffer = ""
                full_response = ""  # 收集完整回复以便添加到历史
                punctuation = {",", "，", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", "\n"}

                try:
                    async for char in chat_stream(message_history):
                        # 实时推流文字
                        await websocket.send_json({"type": "text-update", "content": char})

                        sentence_buffer += char
                        full_response += char

                        # 断句
                        if char in punctuation:
                            if len(sentence_buffer.strip()) > 1:
                                print(f"🗣️ [{client_id}] Synthesizing: {sentence_buffer}")
                                audio_base64 = await text_to_speech(sentence_buffer)

                                if audio_base64:
                                    await websocket.send_json({
                                        "type": "audio-chunk",
                                        "content": audio_base64
                                    })
                                sentence_buffer = ""

                    # 处理剩余文本
                    if sentence_buffer.strip():
                         print(f"🗣️ [{client_id}] Synthesizing (Final): {sentence_buffer}")
                         audio_base64 = await text_to_speech(sentence_buffer)
                         if audio_base64:
                            await websocket.send_json({
                                "type": "audio-chunk",
                                "content": audio_base64
                            })

                    # 将助手回复添加到对话历史
                    if full_response.strip():
                        add_to_history("assistant", full_response.strip())
                        print(f"📝 [{client_id}] Added assistant response to history ({len(full_response)} chars)")

                except Exception as e:
                    print(f"❌ LLM/TTS Process Error: {e}")
                    await websocket.send_json({"type": "text-update", "content": f"\n[Error: {str(e)}]"})

                await websocket.send_json({"type": "status", "content": "idle"})

    except WebSocketDisconnect:
        print(f"👋 Client {client_id} disconnected")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        try:
            await websocket.close()
        except:
            pass