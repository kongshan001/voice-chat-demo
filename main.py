"""
语音识别 + GLM 对话 Demo (增强版)
- 连续对话
- VAD 语音活动检测
- 流式响应
依赖: pip install faster-whisper zhipuai edge-tts sounddevice numpy webrtcvad
"""
import os
import io
import asyncio
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from zhipuai import ZhipuAI
import edge_tts
import webrtcvad

# ============ 配置 ============
API_KEY = os.getenv("ZHIPU_API_KEY", "your-api-key-here")
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
SAMPLE_RATE = 16000
CHANNELS = 1

# 对话历史
conversation_history = [
    {"role": "system", "content": "你是一个友好的AI助手，请用中文简洁回复。"}
]

# 全局状态
is_listening = False
audio_buffer = []


# ============ VAD 录音 ============
def vad_collector(sample_rate, padding_duration=300, ratio=0.75):
    """
    VAD 语音活动检测收集器
    padding_duration: 静音阈值(毫秒)，超过则结束
    ratio: 语音比例阈值
    """
    vad = webrtcvad.Vad(3)  # 激进模式
    num_padding_frames = int(padding_duration / 1000 * sample_rate / 1000)
    ring_buffer = []
    triggered = False
    voiced_frames = []
    
    for frame in audio_buffer:
        if len(frame) < 640:
            continue
            
        is_speech = vad.is_speech(frame, sample_rate)
        
        if not triggered:
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            num_voiced = len([f for f in ring_buffer if f])
            if num_voiced > ratio * num_padding_frames:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer = []
        else:
            voiced_frames.append(frame)
            ring_buffer = []
            if not is_speech:
                num_voiced = len([f for f in ring_buffer if f])
                if num_voiced == 0:
                    break
    
    if triggered:
        return b''.join(voiced_frames)
    return None


def record_with_vad(duration=10):
    """带 VAD 的录音"""
    global audio_buffer, is_listening
    
    audio_buffer = []
    is_listening = True
    
    print("🎤 .listenening for speech... (say something)")
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_buffer.append(indata.tobytes())
    
    stream = sd.InputStream(
        callback=callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=320
    )
    
    with stream:
        # 等待录音或超时
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            if not is_listening:
                break
            time.sleep(0.1)
    
    # 等待一点尾音
    import time
    time.sleep(0.3)
    is_listening = False
    
    # 处理 VAD
    if len(audio_buffer) > 0:
        audio_data = b''.join(audio_buffer)
        return np.frombuffer(audio_data, dtype=np.int16)
    return None


def simple_record(duration=5):
    """简单录音（备用）"""
    print(f"🎤 录音中... ({duration}秒)")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32)
    sd.wait()
    return (audio * 32767).astype(np.int16)


# ============ 语音识别 ============
def speech_to_text(audio_data):
    """语音转文字"""
    if audio_data is None or len(audio_data) < 1600:
        return None
        
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8")
    segments, _ = model.transcribe(audio_data, language="zh")
    text = "".join([seg.text for seg in segments])
    return text.strip()


# ============ GLM 对话 (流式) ============
def chat_with_glm_stream(user_input, callback):
    """流式调用 GLM API"""
    client = ZhipuAI(api_key=API_KEY)
    
    conversation_history.append({"role": "user", "content": user_input})
    
    full_response = ""
    
    response = client.chat.completions.create(
        model="glm-4",
        messages=conversation_history,
        stream=True,
        temperature=0.7
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            callback(content)  # 流式输出回调
    
    conversation_history.append({"role": "assistant", "content": full_response})
    return full_response


# ============ 语音合成 (流式播放) ============
async def text_to_speech_stream(text):
    """流式语音合成 + 播放"""
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    
    # 临时文件
    output_file = "/tmp/reply.mp3"
    
    await communicate.save(output_file)
    return output_file


def play_audio(file_path):
    """播放音频"""
    import subprocess
    if os.name == "nt":
        import winsound
        winsound.PlaySound(file_path, winsound.SND_FILENAME)
    else:
        # Mac/Linux 播放
        os.system(f"afplay {file_path} 2>/dev/null || aplay {file_path} 2>/dev/null || mpg123 {file_path} 2>/dev/null")


# ============ 主流程 ============
def main():
    print("=" * 50)
    print("🎙️ 语音对话 Demo (增强版)")
    print("   - 按 Enter 开始录音")
    print("   - 说 '再见' 退出")
    print("=" * 50)
    
    if API_KEY == "your-api-key-here":
        print("⚠️ 请设置环境变量 ZHIPU_API_KEY")
        return
    
    # 预加载 Whisper 模型
    print("📥 加载 Whisper 模型...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8")
    print("✅ 模型加载完成\n")
    
    while True:
        input("▶️ 按 Enter 开始说话...")
        
        # 1. 录音 (带 VAD)
        audio_data = record_with_vad(duration=10)
        
        if audio_data is None or len(audio_data) < 1600:
            print("❌ 未检测到语音\n")
            continue
        
        # 2. 语音识别
        print("📝 识别中...")
        user_text = speech_to_text(audio_data)
        
        if not user_text:
            print("❌ 识别失败\n")
            continue
            
        print(f"👤 你说: {user_text}")
        
        # 检查退出
        if "再见" in user_text or "退出" in user_text or "拜拜" in user_text:
            print("👋 再见!")
            break
        
        # 3. 流式对话
        print("🤖 AI: ", end="", flush=True)
        
        accumulated_reply = ""
        
        def stream_callback(text):
            nonlocal accumulated_reply
            accumulated_reply += text
            print(text, end="", flush=True)
        
        try:
            chat_with_glm_stream(user_text, stream_callback)
            print("\n")
        except Exception as e:
            print(f"\n❌ API 错误: {e}")
            continue
        
        # 4. 语音合成 (用户按任意键后播放)
        input("🔊 按 Enter 播放语音回复...")
        print("🔊 播放中...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_file = loop.run_until_complete(text_to_speech_stream(accumulated_reply))
        loop.close()
        
        play_audio(audio_file)
        print("-" * 50)


if __name__ == "__main__":
    main()
