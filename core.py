"""
语音对话核心模块
可测试的业务逻辑抽离
"""
from typing import Optional, Callable, List, Dict
import numpy as np


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, system_prompt: str = "你是一个友好的AI助手，请用中文简洁回复。"):
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
    
    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        return self.history
    
    def clear_history(self) -> None:
        """清空对话历史，保留 system"""
        self.history = [self.history[0]]
    
    def should_exit(self, text: str) -> bool:
        """检查是否应该退出对话"""
        import re
        
        # 中文关键词（任意位置包含即可）
        exit_keywords = ["再见", "退出", "拜拜", "结束"]
        for keyword in exit_keywords:
            if keyword in text:
                return True
        
        # 英文关键词（单词边界）
        english_keywords = ["stop", "quit", "exit", "bye"]
        text_lower = text.lower()
        for keyword in english_keywords:
            if re.search(rf'(^|\s){keyword}($|\s)', text_lower):
                return True
        
        return False


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """标准化音频"""
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        return audio
    
    def is_valid_audio(self, audio: Optional[np.ndarray], min_samples: int = 1600) -> bool:
        """检查音频是否有效"""
        if audio is None:
            return False
        return len(audio) >= min_samples
    
    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """音频转字节"""
        return audio.tobytes()
    
    def bytes_to_audio(self, data: bytes) -> np.ndarray:
        """字节转音频"""
        return np.frombuffer(data, dtype=np.int16)


class TextProcessor:
    """文本处理器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        return text.strip()
    
    @staticmethod
    def is_empty_text(text: Optional[str]) -> bool:
        """检查是否为空文本"""
        return not text or not text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        """截断文本"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class Config:
    """配置类"""
    
    def __init__(
        self,
        api_key: str = "",
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        sample_rate: int = 16000,
        channels: int = 1,
        vad_aggressiveness: int = 3
    ):
        self.api_key = api_key
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad_aggressiveness = vad_aggressiveness
    
    @property
    def is_configured(self) -> bool:
        """检查是否配置完成"""
        return bool(self.api_key and self.api_key != "your-api-key-here")
