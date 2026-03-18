"""
语音对话核心模块
可测试的业务逻辑抽离
"""
import logging
import re
from typing import Optional, Callable, List, Dict, Any
import numpy as np


# ============ 自定义异常 ============
class VoiceChatError(Exception):
    """语音对话基础异常"""
    pass


class AudioProcessingError(VoiceChatError):
    """音频处理异常"""
    pass


class ConfigurationError(VoiceChatError):
    """配置异常"""
    pass


class ServiceNotConfiguredError(VoiceChatError):
    """服务未配置异常"""
    pass


class TranscriptionError(VoiceChatError):
    """语音识别异常"""
    pass


class ChatError(VoiceChatError):
    """对话服务异常"""
    pass


class SynthesisError(VoiceChatError):
    """语音合成异常"""
    pass

# 配置日志 (只在模块级别配置一次)
_logging_configured = False

def _setup_logging(level: str = "INFO"):
    """设置日志级别"""
    global _logging_configured
    if _logging_configured:
        return
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))
    _logging_configured = True

_setup_logging()
logger = logging.getLogger(__name__)


class ConversationManager:
    """对话管理器"""
    
    # 退出关键词 (类常量)
    EXIT_KEYWORDS_ZH = ["再见", "退出", "拜拜", "结束"]
    EXIT_KEYWORDS_EN = ["stop", "quit", "exit", "bye"]
    
    def __repr__(self) -> str:
        return f"ConversationManager(history_len={len(self.history)}, max_history={self.max_history})"
    
    # 预编译英文退出关键词正则 (提高性能)
    _EXIT_EN_PATTERNS = [
        re.compile(rf'(^|\s){keyword}($|\s)', re.IGNORECASE)
        for keyword in EXIT_KEYWORDS_EN
    ]
    
    def __init__(
        self, 
        system_prompt: str = "你是一个友好的AI助手，请用中文简洁回复。",
        max_history: int = 20
    ):
        """
        初始化对话管理器
        
        Args:
            system_prompt: 系统提示词
            max_history: 最大历史记录轮数 (每轮包含 user + assistant)
        """
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.max_history = max_history
    
    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self) -> None:
        """修剪历史记录，保持 max_history 轮对话"""
        # system + max_history * 2 (user + assistant)
        max_entries = 1 + (self.max_history * 2)
        if len(self.history) > max_entries:
            # 保留 system，移除最旧的对话
            self.history = [self.history[0]] + self.history[-(self.max_history * 2):]
            logger.debug(f"历史记录已修剪，保留最近 {self.max_history} 轮对话")
    
    def get_messages(self) -> List[Dict[str, str]]:
        return self.history
    
    def clear_history(self) -> None:
        """清空对话历史，保留 system"""
        self.history = [self.history[0]]
        logger.info("对话历史已清空")
    
    def should_exit(self, text: str) -> bool:
        """检查是否应该退出对话"""
        # 中文关键词（任意位置包含即可）
        for keyword in self.EXIT_KEYWORDS_ZH:
            if keyword in text:
                logger.info(f"检测到退出关键词: {keyword}")
                return True
        
        # 英文关键词（使用预编译正则）
        for pattern in self._EXIT_EN_PATTERNS:
            if pattern.search(text):
                logger.info(f"检测到退出关键词")
                return True
        
        return False


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率 (Hz)
            channels: 通道数 (1=单声道, 2=立体声)
            
        Raises:
            ValueError: 参数无效时抛出
        """
        if sample_rate <= 0:
            raise ValueError(f"采样率必须为正数: {sample_rate}")
        if channels <= 0:
            raise ValueError(f"通道数必须为正数: {channels}")
        
        self.sample_rate = sample_rate
        self.channels = channels
    
    def __repr__(self) -> str:
        return f"AudioProcessor(sample_rate={self.sample_rate}, channels={self.channels})"
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """标准化音频
        
        Args:
            audio: 输入音频数据 (float32/float64/int16/int32)
            
        Returns:
            标准化后的 int16 音频数据
        """
        if audio is None or len(audio) == 0:
            raise ValueError("音频数据为空")
        
        # 检查数据类型
        if audio.dtype not in [np.float32, np.float64, np.int16, np.int32]:
            raise ValueError(f"不支持的音频数据类型: {audio.dtype}")
        
        # 处理 float 类型的归一化
        if audio.dtype in [np.float32, np.float64]:
            # 限制范围在 [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        
        return audio
    
    def is_valid_audio(self, audio: Optional[np.ndarray], min_samples: int = 1600) -> bool:
        """检查音频是否有效"""
        if audio is None:
            return False
        if not isinstance(audio, np.ndarray):
            return False
        return len(audio) >= min_samples
    
    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """音频转字节"""
        if audio is None or len(audio) == 0:
            raise ValueError("音频数据为空")
        return audio.tobytes()
    
    def bytes_to_audio(self, data: bytes) -> np.ndarray:
        """字节转音频"""
        if data is None or len(data) == 0:
            raise ValueError("字节数据为空")
        return np.frombuffer(data, dtype=np.int16)


class TextProcessor:
    """文本处理器"""
    
    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """
        清理文本
        
        Args:
            text: 待清理的文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        return text.strip()
    
    @staticmethod
    def is_empty_text(text: Optional[str]) -> bool:
        """
        检查是否为空文本
        
        Args:
            text: 待检查的文本
            
        Returns:
            True 表示空文本
        """
        return not text or not text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        """
        截断文本
        
        Args:
            text: 待截断的文本
            max_length: 最大长度
            
        Returns:
            截断后的文本
        """
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    @staticmethod
    def is_chinese(text: str) -> bool:
        """
        检查文本是否包含中文字符
        
        Args:
            text: 待检查的文本
            
        Returns:
            True 表示包含中文
        """
        if not text:
            return False
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def __repr__(self) -> str:
        return "TextProcessor()"


class Config:
    """配置类"""
    
    __slots__ = (
        'api_key', 'whisper_model', 'whisper_device', 'sample_rate',
        'channels', 'vad_aggressiveness', 'tts_voice', 'temperature',
        'max_history', 'log_level'
    )
    
    # 支持的 Whisper 模型
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
    
    # 支持的 TTS 语音
    TTS_VOICES = {
        # 中文
        "zh-CN-XiaoxiaoNeural": "晓晓 (女声)",
        "zh-CN-YunxiNeural": "云希 (男声)",
        "zh-CN-YunyangNeural": "云扬 (男声)",
        "zh-CN-YunhaoNeural": "云浩 (男声)",
        "zh-CN-XiaoyiNeural": "小艺 (女声)",
        "zh-CN-YunyeNeural": "云野 (情感男声)",
        # 英文
        "en-US-JennyNeural": "Jenny (英文女声)",
        "en-US-GuyNeural": "Guy (英文男声)",
        "en-US-AriaNeural": "Aria (英文女声)",
        "en-US-SaraNeural": "Sara (英文女声)",
    }
    
    def __init__(
        self,
        api_key: str = "",
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        sample_rate: int = 16000,
        channels: int = 1,
        vad_aggressiveness: int = 3,
        tts_voice: str = "zh-CN-XiaoxiaoNeural",
        temperature: float = 0.7,
        max_history: int = 20,
        log_level: str = "INFO"
    ):
        """
        初始化配置
        
        Args:
            api_key: GLM API Key
            whisper_model: Whisper 模型大小
            whisper_device: 运行设备 (cpu/cuda)
            sample_rate: 音频采样率
            channels: 音频通道数
            vad_aggressiveness: VAD 激进度 (0-3)
            tts_voice: TTS 语音选择
            temperature: 模型温度参数
            max_history: 最大对话历史轮数
            log_level: 日志级别
        """
        self.api_key = api_key
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad_aggressiveness = vad_aggressiveness
        self.tts_voice = tts_voice
        self.temperature = temperature
        self.max_history = max_history
        self.log_level = log_level
        # 日志级别在 _setup_logging 中统一配置
    
    @property
    def is_configured(self) -> bool:
        """检查是否配置完成"""
        return bool(self.api_key and self.api_key != "your-api-key-here")
    
    def validate(self) -> List[str]:
        """
        验证配置，返回错误列表
        
        Returns:
            错误消息列表，空列表表示配置有效
        """
        errors = []
        
        # 检查 API Key
        if not self.api_key:
            errors.append("API Key 未设置")
        elif self.api_key == "your-api-key-here":
            errors.append("API Key 是占位符，请设置有效的 API Key")
        
        # 检查 Whisper 模型
        if self.whisper_model not in self.WHISPER_MODELS:
            errors.append(f"不支持的 Whisper 模型: {self.whisper_model}")
        
        # 检查 TTS 语音
        if self.tts_voice not in self.TTS_VOICES:
            errors.append(f"不支持的 TTS 语音: {self.tts_voice}")
        
        # 检查采样率
        if self.sample_rate <= 0:
            errors.append(f"采样率必须为正数: {self.sample_rate}")
        elif self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            logger.warning(f"非标准采样率: {self.sample_rate}")
        
        # 检查 VAD 激进度
        if not 0 <= self.vad_aggressiveness <= 3:
            errors.append(f"VAD 激进度超出范围: {self.vad_aggressiveness}")
        
        # 检查温度参数
        if not 0 <= self.temperature <= 2:
            errors.append(f"温度参数超出范围: {self.temperature}")
        
        # 检查 max_history
        if self.max_history < 0:
            errors.append(f"max_history 不能为负数: {self.max_history}")
        
        # 检查 whisper_device
        if self.whisper_device not in ("cpu", "cuda"):
            errors.append(f"不支持的设备: {self.whisper_device}，仅支持 cpu 或 cuda")
        
        if errors:
            logger.error(f"配置验证失败: {errors}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        Returns:
            配置字典 (api_key 会被 masked)
        """
        return {
            "api_key": "***" if self.api_key else "",
            "whisper_model": self.whisper_model,
            "whisper_device": self.whisper_device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "vad_aggressiveness": self.vad_aggressiveness,
            "tts_voice": self.tts_voice,
            "temperature": self.temperature,
            "max_history": self.max_history,
            "log_level": self.log_level,
        }
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量创建配置
        
        Returns:
            Config 实例
        """
        import os
        return cls(
            api_key=os.getenv("ZHIPU_API_KEY", ""),
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            channels=int(os.getenv("CHANNELS", "1")),
            vad_aggressiveness=int(os.getenv("VAD_AGGRESSIVENESS", "3")),
            tts_voice=os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_history=int(os.getenv("MAX_HISTORY", "20")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def __repr__(self) -> str:
        mask = "***" if self.api_key else "(empty)"
        return f"Config(api_key={mask}, whisper_model={self.whisper_model}, tts_voice={self.tts_voice})"
