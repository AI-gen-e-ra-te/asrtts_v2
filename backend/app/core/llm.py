import os
from typing import List, Dict, Union
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# 从环境变量读取配置，提供默认值
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama默认地址
API_KEY = os.getenv("LLM_API_KEY", "ollama")  # Ollama不需要真实API密钥
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")  # 默认模型

# 验证关键配置
if not BASE_URL:
    print("⚠️ WARNING: LLM_BASE_URL not set, using default Ollama endpoint")
    BASE_URL = "http://localhost:11434/v1"

# 初始化异步客户端
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

print(f"[INFO] LLM Client initialized: {MODEL} @ {BASE_URL}")

async def chat_stream(messages: Union[str, List[Dict[str, str]]]):
    """
    异步生成器：流式返回 LLM 的文本回复
    支持两种输入格式：
    1. 字符串: 自动包装为 [system, user] 消息
    2. 消息列表: 直接使用，如果缺少system消息则自动添加
    """
    try:
        # 处理输入格式
        if isinstance(messages, str):
            # 字符串输入，包装为消息列表
            message_list = [
                {"role": "system", "content": "You are a helpful voice assistant. Please keep your replies concise, short, and conversational suitable for TTS."},
                {"role": "user", "content": messages}
            ]
        else:
            # 列表输入，确保有system消息
            message_list = messages.copy()
            # 检查是否有system消息
            has_system = any(msg.get("role") == "system" for msg in message_list)
            if not has_system:
                # 在开头添加默认system消息
                message_list.insert(0, {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Please keep your replies concise, short, and conversational suitable for TTS."
                })

        response = await client.chat.completions.create(
            model=MODEL,
            messages=message_list,
            stream=True,
            temperature=0.7,
        )

        # 逐块读取流
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        print(f"[ERROR] LLM Error: {e}")
        yield f" Error: {str(e)}"