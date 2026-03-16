"""
服务层接口定义 (用于测试 mock)
"""
from abc import ABC, abstractmethod
from typing import Optional, Generator
import numpy as np


class ISpeechRecognizer(ABC):
    """语音识别接口"""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """语音转文字"""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass


class IChatService(ABC):
    """对话服务接口"""
    
    @abstractmethod
    def chat(self, messages: list, stream_callback: Optional[callable] = None) -> str:
        """对话"""
        pass


class ITTSService(ABC):
    """语音合成接口"""
    
    @abstractmethod
    async def synthesize(self, text: str, output_path: str) -> str:
        """文字转语音"""
        pass


class IAudioRecorder(ABC):
    """音频录制接口"""
    
    @abstractmethod
    def record(self, duration: float) -> Optional[np.ndarray]:
        """录制音频"""
        pass
