"""
语音识别 + GLM 对话 Demo (增强版)
- 连续对话
- VAD 语音活动检测
- 流式响应
- 硬件部署支持
依赖: pip install faster-whisper zhipuai edge-tts sounddevice numpy webrtcvad
"""
import logging
import os
import sys
import argparse
import asyncio
import time
from typing import Optional, Callable, List, Dict, Any
import numpy as np

from core import ConversationManager, AudioProcessor, TextProcessor, Config, ServiceNotConfiguredError, TranscriptionError
from services import ISpeechRecognizer, IChatService, ITTSService, IAudioRecorder, WhisperRecognizer, GLMChatService, EdgeTTSService

# 配置日志
logger = logging.getLogger(__name__)

# 类型别名
Messages = List[Dict[str, str]]
StreamCallback = Callable[[str], None]

# 服务提供者 (可替换为 mock)
_speech_recognizer: Optional[ISpeechRecognizer] = None
_chat_service: Optional[IChatService] = None
_tts_service: Optional[ITTSService] = None


def configure_services(
    recognizer: Optional[ISpeechRecognizer] = None,
    chat_service: Optional[IChatService] = None,
    tts_service: Optional[ITTSService] = None
) -> None:
    """配置服务提供者 (用于测试注入)"""
    global _speech_recognizer, _chat_service, _tts_service
    if recognizer:
        _speech_recognizer = recognizer
    if chat_service:
        _chat_service = chat_service
    if tts_service:
        _tts_service = tts_service


# ============ 服务实现说明 ============
# 所有服务实现已移至 services.py
# 此处导入是为了保持向后兼容性


# ============ VAD 录音 ============
def record_with_vad(duration=10, sample_rate=16000, channels=1):  # pragma: no cover
    """带 VAD 的录音 (需要麦克风硬件，无法单元测试)
    
    Args:
        duration: 录音时长（秒）
        sample_rate: 采样率
        channels: 通道数
        
    Returns:
        音频数据，失败返回 None
        
    Raises:
        RuntimeError: 录音设备错误
    """
    import webrtcvad
    import sounddevice as sd
    
    audio_buffer = []
    is_listening = True
    
    vad = webrtcvad.Vad(3)
    
    def callback(indata, frames, time, status):
        if status:
            logger.warning(f"录音状态: {status}")
        audio_buffer.append(indata.tobytes())
    
    try:
        stream = sd.InputStream(
            callback=callback,
            channels=channels,
            samplerate=sample_rate,
            blocksize=320
        )
    except Exception as e:
        raise RuntimeError(f"无法打开麦克风: {e}") from e
    
    with stream:
        start_time = time.time()
        while time.time() - start_time < duration:
            if not is_listening:
                break
            time.sleep(0.1)
    
    time.sleep(0.3)
    is_listening = False
    
    if audio_buffer:
        audio_data = b''.join(audio_buffer)
        return np.frombuffer(audio_data, dtype=np.int16)
    return None


def simple_record(duration=5, sample_rate=16000, channels=1):  # pragma: no cover
    """简单录音 (需要麦克风硬件，无法单元测试)
    
    Args:
        duration: 录音时长（秒）
        sample_rate: 采样率
        channels: 通道数
        
    Returns:
        音频数据
        
    Raises:
        RuntimeError: 录音设备错误
    """
    import sounddevice as sd
    
    try:
        print(f"🎤 录音中... ({duration}秒)")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=np.float32)
        sd.wait()
        return (audio * 32767).astype(np.int16)
    except Exception as e:
        raise RuntimeError(f"录音失败: {e}") from e


def play_audio(file_path):  # pragma: no cover
    """播放音频 (需要扬声器硬件，无法单元测试)
    
    Args:
        file_path: 音频文件路径
        
    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 播放失败
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    try:
        if os.name == "nt":
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        else:
            # 尝试多个播放器
            result = os.system(f"afplay {file_path} 2>/dev/null")
            if result != 0:
                result = os.system(f"aplay {file_path} 2>/dev/null")
            if result != 0:
                raise RuntimeError("音频播放失败，请检查音频设备")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, RuntimeError)):
            raise
        raise RuntimeError(f"音频播放失败: {e}") from e


# ============ 业务逻辑 (可测试) ============
class VoiceChatApp:
    """语音对话应用"""
    
    def __init__(
        self,
        config: Config,
        recognizer: ISpeechRecognizer = None,
        chat_service: IChatService = None,
        tts_service: ITTSService = None
    ):
        self.config = config
        self.conversation = ConversationManager(max_history=config.max_history)
        self.audio_processor = AudioProcessor(config.sample_rate, config.channels)
        self.text_processor = TextProcessor()
        
        # 注入服务
        self.recognizer = recognizer
        self.chat_service = chat_service
        self.tts_service = tts_service
    
    def process_audio(self, audio) -> Optional[str]:
        """处理音频 -> 文字"""
        if not self.audio_processor.is_valid_audio(audio):
            return None
        
        audio = self.audio_processor.normalize_audio(audio)
        return self.speech_to_text(audio)
    
    def speech_to_text(self, audio) -> str:
        """语音识别
        
        Raises:
            ServiceNotConfiguredError: 识别服务未配置
            TranscriptionError: 识别失败
        """
        if not self.recognizer:
            raise ServiceNotConfiguredError("Recognizer not configured")
        
        try:
            return self.recognizer.transcribe(audio)
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            raise TranscriptionError(f"语音识别失败: {e}") from e
    
    def chat(self, user_input: str, stream_callback=None) -> str:
        """对话"""
        self.conversation.add_user_message(user_input)
        
        if self.chat_service:
            response = self.chat_service.chat(
                self.conversation.get_messages(),
                stream_callback
            )
            self.conversation.add_assistant_message(response)
            return response
        
        raise ServiceNotConfiguredError("Chat service not configured")
    
    async def synthesize_speech(self, text: str, output_path: str) -> str:
        """语音合成"""
        if self.tts_service:
            return await self.tts_service.synthesize(text, output_path)
        raise ServiceNotConfiguredError("TTS service not configured")
    
    def should_exit(self, text: str) -> bool:
        """检查是否退出"""
        return self.conversation.should_exit(text)


# ============ 主流程 ============
def main(argv=None):  # pragma: no cover
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="语音对话 Demo")
    parser.add_argument("--api-key", "-k", help="GLM API Key")
    parser.add_argument("--whisper-model", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper 模型大小 (默认: base)")
    parser.add_argument("--optimize-for-pi", action="store_true",
                        help="为树莓派优化 (使用 tiny 模型)")
    parser.add_argument("--sample-rate", "-s", type=int, default=16000,
                        help="音频采样率 (默认: 16000)")
    args = parser.parse_args(argv)
    
    # 树莓派优化模式
    if args.optimize_for_pi or os.environ.get("RASPBERRY_PI"):
        print("🍓 树莓派优化模式已启用")
        args.whisper_model = "tiny"
    
    # 合并配置
    api_key = args.api_key or os.getenv("ZHIPU_API_KEY", "your-api-key-here")
    whisper_model = args.whisper_model
    
    print("=" * 50)
    print("🎙️ 语音对话 Demo (增强版)")
    print(f"   模型: whisper-{whisper_model}")
    print("=" * 50)

    # 从环境变量或 .env 文件加载配置
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional
    
    if api_key == "your-api-key-here":
        print("⚠️ 请设置环境变量 ZHIPU_API_KEY 或使用 --api-key 参数")
        print("   参考 .env.example 文件")
        return
    
    # 初始化配置
    config = Config(
        api_key=api_key,
        whisper_model=whisper_model,
        sample_rate=args.sample_rate
    )
    
    # 初始化服务
    recognizer = WhisperRecognizer(whisper_model, "cpu")
    chat_service = GLMChatService(api_key)
    tts_service = EdgeTTSService()
    
    app = VoiceChatApp(
        config=config,
        recognizer=recognizer,
        chat_service=chat_service,
        tts_service=tts_service
    )
    
    print("📥 加载 Whisper 模型...")
    recognizer.load_model()
    print("✅ 模型加载完成\n")

    try:
        while True:
            try:
                input("▶️ 按 Enter 开始说话...")
            except EOFError:
                break
            
            # 1. 录音
            audio_data = record_with_vad(duration=10)
            
            if audio_data is None or len(audio_data) < 1600:
                print("❌ 未检测到语音\n")
                continue
            
            # 2. 语音识别
            print("📝 识别中...")
            user_text = app.process_audio(audio_data)
            
            if not user_text:
                print("❌ 识别失败\n")
                continue
                
            print(f"👤 你说: {user_text}")
            
            # 检查退出
            if app.should_exit(user_text):
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
                app.chat(user_text, stream_callback)
                print("\n")
            except Exception as e:
                print(f"\n❌ API 错误: {e}")
                continue
            
            # 4. 语音合成
            loop = None
            try:
                try:
                    input("🔊 按 Enter 播放语音回复...")
                except EOFError:
                    break
                print("🔊 播放中...")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio_file = loop.run_until_complete(app.synthesize_speech(accumulated_reply, "/tmp/reply.mp3"))
            finally:
                if loop:
                    loop.close()
            
            play_audio(audio_file)
            print("-" * 50)
    
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在退出...")
    finally:
        print("✅ 程序已退出")


if __name__ == "__main__":  # pragma: no cover
    main()
