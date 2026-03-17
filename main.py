"""
语音识别 + GLM 对话 Demo (增强版)
- 连续对话
- VAD 语音活动检测
- 流式响应
- 硬件部署支持
依赖: pip install faster-whisper zhipuai edge-tts sounddevice numpy webrtcvad
"""
import os
import sys
import argparse
import asyncio
from typing import Optional, Callable, List, Dict, Any
import numpy as np

from core import ConversationManager, AudioProcessor, TextProcessor, Config, ServiceNotConfiguredError
from services import ISpeechRecognizer, IChatService, ITTSService

# 类型别名
Messages = List[Dict[str, str]]
StreamCallback = Optional[Callable[[str], None]]

# 默认配置
_config = Config(
    api_key=os.getenv("ZHIPU_API_KEY", "your-api-key-here"),
    whisper_model="base",
    whisper_device="cpu"
)

# 服务提供者 (可替换为 mock)
_speech_recognizer: Optional[ISpeechRecognizer] = None
_chat_service: Optional[IChatService] = None
_tts_service: Optional[ITTSService] = None


def configure_services(
    recognizer: ISpeechRecognizer = None,
    chat_service: IChatService = None,
    tts_service: ITTSService = None
):
    """配置服务提供者 (用于测试注入)"""
    global _speech_recognizer, _chat_service, _tts_service
    if recognizer:
        _speech_recognizer = recognizer
    if chat_service:
        _chat_service = chat_service
    if tts_service:
        _tts_service = tts_service


# ============ 语音识别实现 ============
class WhisperRecognizer(ISpeechRecognizer):
    """Whisper 语音识别"""
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
    
    def load_model(self):  # pragma: no cover (需要下载大模型文件)
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_name, device=self.device, compute_type="int8")
    
    def transcribe(self, audio):  # pragma: no cover (需要实际模型)
        if self.model is None:
            self.load_model()
        segments, _ = self.model.transcribe(audio, language="zh")
        text = "".join([seg.text for seg in segments])
        return text.strip()


# ============ GLM 对话实现 ============
class GLMChatService(IChatService):
    """GLM 对话服务"""
    
    def __init__(self, api_key: str):  # pragma: no cover (需要 API 调用)
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=api_key)
    
    def chat(self, messages, stream_callback=None, timeout: int = 60):  # pragma: no cover (需要 API 调用)
        """对话 (支持超时设置)
        
        Args:
            messages: 对话消息列表
            stream_callback: 流式响应回调
            timeout: 超时秒数 (默认60秒)
        """
        # 输入验证
        if not messages:
            raise ValueError("消息列表不能为空")
        
        import requests
        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=messages,
                stream=stream_callback is not None,
                temperature=0.7,
                timeout=timeout
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("GLM API 请求超时，请检查网络连接") from None
        except requests.exceptions.ConnectionError:
            raise ConnectionError("GLM API 连接失败，请检查网络") from None
        except Exception as e:
            raise RuntimeError(f"GLM API 调用失败: {e}") from e
        
        full_response = ""
        
        if stream_callback:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    stream_callback(content)
        else:
            full_response = response.choices[0].message.content
        
        return full_response


# ============ Edge TTS 实现 ============
class EdgeTTSService(ITTSService):
    """Edge TTS 服务"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):  # pragma: no cover (需要网络调用)
        import edge_tts
        self.voice = voice
        self.communicate = edge_tts.Communicate
    
    async def synthesize(self, text: str, output_path: str) -> str:
        """语音合成
        
        Args:
            text: 要转换的文本
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
            
        Raises:
            ValueError: 文本为空
            ConnectionError: 网络连接失败
            RuntimeError: TTS 服务错误
        """
        if not text or not text.strip():
            raise ValueError("合成文本不能为空")
        
        import aiohttp
        try:
            communicate = self.communicate(text, self.voice)
            await communicate.save(output_path)
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Edge TTS 网络连接失败: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Edge TTS 合成失败: {e}") from e
        
        return output_path


# ============ VAD 录音 ============
def record_with_vad(duration=10, sample_rate=16000, channels=1):  # pragma: no cover
    """带 VAD 的录音 (需要麦克风硬件，无法单元测试)"""
    import webrtcvad
    import sounddevice as sd
    
    audio_buffer = []
    is_listening = True
    
    vad = webrtcvad.Vad(3)
    
    def callback(indata, frames, time, status):
        if status:
            pass
        audio_buffer.append(indata.tobytes())
    
    stream = sd.InputStream(
        callback=callback,
        channels=channels,
        samplerate=sample_rate,
        blocksize=320
    )
    
    with stream:
        import time
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
    """简单录音 (需要麦克风硬件，无法单元测试)"""
    import sounddevice as sd
    import numpy as np
    
    print(f"🎤 录音中... ({duration}秒)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=np.float32)
    sd.wait()
    return (audio * 32767).astype(np.int16)


def play_audio(file_path):  # pragma: no cover
    """播放音频 (需要扬声器硬件，无法单元测试)"""
    import os
    if os.name == "nt":
        import winsound
        winsound.PlaySound(file_path, winsound.SND_FILENAME)
    else:
        os.system(f"afplay {file_path} 2>/dev/null || aplay {file_path} 2>/dev/null")


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
        self.conversation = ConversationManager()
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
        """语音识别"""
        if self.recognizer:
            return self.recognizer.transcribe(audio)
        raise ServiceNotConfiguredError("Recognizer not configured")
    
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
    import numpy as np
    
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
    
    if api_key == "your-api-key-here":
        print("⚠️ 请设置环境变量 ZHIPU_API_KEY 或使用 --api-key 参数")
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
        try:
            input("🔊 按 Enter 播放语音回复...")
        except EOFError:
            break
        print("🔊 播放中...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_file = loop.run_until_complete(app.synthesize_speech(accumulated_reply, "/tmp/reply.mp3"))
        loop.close()
        
        play_audio(audio_file)
        print("-" * 50)


if __name__ == "__main__":  # pragma: no cover
    main()
