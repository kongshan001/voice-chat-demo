"""
服务层完整测试 - 提升覆盖率
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import asyncio

from services import (
    WhisperRecognizer, 
    GLMChatService, 
    EdgeTTSService,
    ISpeechRecognizer,
    IChatService,
    ITTSService,
    IAudioRecorder
)


class TestWhisperRecognizer:
    """Whisper 识别器测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        recognizer = WhisperRecognizer()
        assert recognizer.model_name == "base"
        assert recognizer.device == "cpu"
        assert recognizer.model is None
    
    def test_init_custom(self):
        """测试自定义参数"""
        recognizer = WhisperRecognizer("tiny", "cuda")
        assert recognizer.model_name == "tiny"
        assert recognizer.device == "cuda"
    
    def test_init_large_model(self):
        """测试 large 模型"""
        recognizer = WhisperRecognizer("large")
        assert recognizer.model_name == "large"


class TestGLMChatService:
    """GLM 对话服务测试"""
    
    def test_init(self):
        """测试初始化"""
        service = GLMChatService("test-key")
        assert service.client is not None
        assert service.DEFAULT_TIMEOUT == 60
        assert service.MAX_RETRIES == 3
    
    def test_chat_empty_messages(self):
        """测试空消息列表"""
        service = GLMChatService("test-key")
        
        with pytest.raises(ValueError, match="消息列表不能为空"):
            service.chat([])
    
    def test_retry_delay_constant(self):
        """测试重试延迟常量"""
        assert GLMChatService.RETRY_DELAY == 1
    
    def test_max_retries_constant(self):
        """测试最大重试次数常量"""
        assert GLMChatService.MAX_RETRIES == 3


class TestEdgeTTSService:
    """Edge TTS 服务测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        service = EdgeTTSService()
        assert service.voice == "zh-CN-XiaoxiaoNeural"
    
    def test_init_custom_voice(self):
        """测试自定义语音"""
        service = EdgeTTSService("en-US-JennyNeural")
        assert service.voice == "en-US-JennyNeural"
    
    def test_synthesize_empty_text(self):
        """测试空文本"""
        service = EdgeTTSService()
        
        async def test():
            with pytest.raises(ValueError, match="合成文本不能为空"):
                await service.synthesize("", "/tmp/test.mp3")
        
        asyncio.run(test())
    
    def test_synthesize_whitespace_text(self):
        """测试空白文本"""
        service = EdgeTTSService()
        
        async def test():
            with pytest.raises(ValueError, match="合成文本不能为空"):
                await service.synthesize("   ", "/tmp/test.mp3")
        
        asyncio.run(test())
    
    def test_retry_constants(self):
        """测试重试常量"""
        assert EdgeTTSService.MAX_RETRIES == 3
        assert EdgeTTSService.RETRY_DELAY == 1


class TestServiceInterfaces:
    """服务接口测试"""
    
    def test_speech_recognizer_is_abstract(self):
        """测试 ISpeechRecognizer 是抽象类"""
        with pytest.raises(TypeError):
            ISpeechRecognizer()
    
    def test_chat_service_is_abstract(self):
        """测试 IChatService 是抽象类"""
        with pytest.raises(TypeError):
            IChatService()
    
    def test_tts_service_is_abstract(self):
        """测试 ITTSService 是抽象类"""
        with pytest.raises(TypeError):
            ITTSService()
    
    def test_audio_recorder_is_abstract(self):
        """测试 IAudioRecorder 是抽象类"""
        with pytest.raises(TypeError):
            IAudioRecorder()
    
    def test_recognizer_interface_methods(self):
        """测试识别器接口方法签名"""
        class TestRecognizer(ISpeechRecognizer):
            def transcribe(self, audio):
                pass
            def load_model(self):
                pass
        
        recognizer = TestRecognizer()
        assert hasattr(recognizer, 'transcribe')
        assert hasattr(recognizer, 'load_model')
    
    def test_chat_service_interface_methods(self):
        """测试对话服务接口方法签名"""
        class TestChat(IChatService):
            def chat(self, messages, stream_callback=None, timeout=60):
                pass
        
        service = TestChat()
        assert hasattr(service, 'chat')
    
    def test_tts_service_interface_methods(self):
        """测试 TTS 服务接口方法签名"""
        class TestTTS(ITTSService):
            async def synthesize(self, text, output_path):
                pass
        
        service = TestTTS()
        assert hasattr(service, 'synthesize')
    
    def test_audio_recorder_interface_methods(self):
        """测试录音接口方法签名"""
        class TestRecorder(IAudioRecorder):
            def record(self, duration):
                pass
        
        recorder = TestRecorder()
        assert hasattr(recorder, 'record')


class TestServiceRetryMechanism:
    """服务重试机制测试"""
    
    def test_glm_retry_constants(self):
        """测试 GLM 重试常量"""
        assert GLMChatService.MAX_RETRIES == 3
        assert GLMChatService.RETRY_DELAY == 1
    
    def test_edge_tts_retry_constants(self):
        """测试 Edge TTS 重试常量"""
        assert EdgeTTSService.MAX_RETRIES == 3
        assert EdgeTTSService.RETRY_DELAY == 1
