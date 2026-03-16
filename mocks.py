"""
测试用 Mock 服务实现
"""
from typing import Optional, Generator, List, Dict
import numpy as np
from services import ISpeechRecognizer, IChatService, ITTSService, IAudioRecorder


class MockSpeechRecognizer(ISpeechRecognizer):
    """Mock 语音识别"""
    
    def __init__(self, mock_result: str = "测试语音识别结果"):
        self.mock_result = mock_result
        self.model_loaded = False
        self.transcribe_count = 0
    
    def load_model(self) -> None:
        self.model_loaded = True
    
    def transcribe(self, audio: np.ndarray) -> str:
        self.transcribe_count += 1
        # 模拟识别处理
        return self.mock_result


class MockChatService(IChatService):
    """Mock 对话服务"""
    
    def __init__(self, mock_response: str = "这是 Mock 的回复"):
        self.mock_response = mock_response
        self.chat_count = 0
        self.last_messages = None
    
    def chat(self, messages: List[Dict], stream_callback: Optional[callable] = None) -> str:
        self.chat_count += 1
        self.last_messages = messages
        
        if stream_callback:
            # 模拟流式输出
            for char in self.mock_response:
                stream_callback(char)
        
        return self.mock_response


class MockTTSService(ITTSService):
    """Mock TTS 服务"""
    
    def __init__(self):
        self.synthesize_count = 0
        self.last_text = None
    
    async def synthesize(self, text: str, output_path: str) -> str:
        self.synthesize_count += 1
        self.last_text = text
        # Mock 不实际生成文件
        return output_path


class MockAudioRecorder:
    """Mock 录音器"""
    
    def __init__(self, mock_audio: Optional[np.ndarray] = None):
        self.mock_audio = mock_audio if mock_audio is not None else np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        self.record_count = 0
    
    def record(self, duration: float) -> np.ndarray:
        self.record_count += 1
        return self.mock_audio


class MockAudioRecorderAdapter(IAudioRecorder):
    """Mock 录音器 (适配器模式)"""
    
    def __init__(self, mock_audio: Optional[np.ndarray] = None):
        self.mock_audio = mock_audio if mock_audio is not None else np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        self.record_count = 0
    
    def record(self, duration: float) -> Optional[np.ndarray]:
        self.record_count += 1
        return self.mock_audio
