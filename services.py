"""
服务层接口定义 (用于测试 mock)

提供抽象接口以便：
- 单元测试时替换为 mock 对象
- 支持不同的服务实现替换
- 解耦业务逻辑与具体服务
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any
import numpy as np


class ISpeechRecognizer(ABC):
    """语音识别接口
    
    实现类需要实现音频转文字功能
    """
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """语音转文字
        
        Args:
            audio: numpy 音频数据 (int16)
            
        Returns:
            识别的文本内容
        """
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """加载识别模型
        
        可以在初始化时自动调用，或延迟到首次使用时
        """
        pass


class IChatService(ABC):
    """对话服务接口
    
    实现类需要实现对话功能，支持流式回调
    """
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        stream_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 60
    ) -> str:
        """对话
        
        Args:
            messages: 对话消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            stream_callback: 可选的流式响应回调函数
            timeout: 超时秒数
            
        Returns:
            完整的回复文本
        """
        pass


class ITTSService(ABC):
    """语音合成接口
    
    实现类需要实现文字转语音功能
    """
    
    @abstractmethod
    async def synthesize(self, text: str, output_path: str) -> str:
        """文字转语音
        
        Args:
            text: 要转换的文本
            output_path: 输出音频文件路径
            
        Returns:
            输出文件路径
        """
        pass


class IAudioRecorder(ABC):
    """音频录制接口
    
    实现类需要实现音频录制功能
    """
    
    @abstractmethod
    def record(self, duration: float) -> Optional[np.ndarray]:
        """录制音频
        
        Args:
            duration: 录制时长（秒）
            
        Returns:
            numpy 音频数据，录制失败返回 None
        """
        pass
