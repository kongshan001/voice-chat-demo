"""
语音识别 + GLM 对话 Demo
依赖: pip install faster-whisper zhipuai edge-tts sounddevice numpy
"""
import os
import io
import wave
import asyncio
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from zhipuai import ZhipuAI
import edge_tts

# ============ 配置 ============
# GLM API Key (建议从环境变量读取)
API_KEY = os.getenv("ZHIPU_API_KEY", "your-api-key-here")

# Whisper 模型 (tiny/base/small/medium/large)
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"  # 或 "cuda" 如果有GPU

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# 对话历史
conversation_history = [
    {"role": "system", "content": "你是一个友好的AI助手，请用中文回复用户。"}
]


# ============ 语音识别 ============
def record_audio(duration=5, samplerate=SAMPLE_RATE):
    """录制音频"""
    print(f"🎤 录音中... ({duration}秒)")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=CHANNELS, dtype=np.float32)
    sd.wait()
    return (audio * 32767).astype(np.int16)


def speech_to_text(audio_data):
    """语音转文字"""
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8")
    segments, _ = model.transcribe(audio_data, language="zh")
    text = "".join([seg.text for seg in segments])
    return text.strip()


# ============ GLM 对话 ============
def chat_with_glm(user_input):
    """调用 GLM API 对话"""
    client = ZhipuAI(api_key=API_KEY)
    
    conversation_history.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="glm-4",
        messages=conversation_history
    )
    
    assistant_reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    
    return assistant_reply


# ============ 语音合成 ============
async def text_to_speech(text, output_file="reply.mp3"):
    """文字转语音"""
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(output_file)


def play_audio(file_path):
    """播放音频"""
    import subprocess
    # 根据系统选择播放命令
    if os.name == "nt":
        import winsound
        winsound.PlaySound(file_path, winsound.SND_FILENAME)
    else:
        os.system(f"afplay {file_path} 2>/dev/null || aplay {file_path} 2>/dev/null")


# ============ 主流程 ============
def main():
    print("=" * 50)
    print("🎙️ 语音对话 Demo - 按 Ctrl+C 退出")
    print("=" * 50)
    
    # 检查 API Key
    if API_KEY == "your-api-key-here":
        print("⚠️ 请设置环境变量 ZHIPU_API_KEY 或修改代码中的 API_KEY")
        return
    
    while True:
        try:
            # 1. 录音
            audio_data = record_audio(duration=5)
            
            # 2. 语音识别
            user_text = speech_to_text(audio_data)
            if not user_text:
                print("❌ 未检测到语音，请重试")
                continue
                
            print(f"👤 你说: {user_text}")
            
            # 3. GLM 对话
            print("🤖 AI 思考中...")
            reply = chat_with_glm(user_text)
            print(f"🤖 AI: {reply}")
            
            # 4. 语音合成并播放
            print("🔊 生成语音...")
            output_file = "reply.mp3"
            asyncio.run(text_to_speech(reply, output_file))
            play_audio(output_file)
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
