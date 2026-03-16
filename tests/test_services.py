"""
服务层错误处理测试
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from main import VoiceChatApp, WhisperRecognizer, GLMChatService, EdgeTTSService
from core import Config, ConversationManager


class TestServiceErrorHandling:
    """服务错误处理测试"""
    
    def test_voice_chat_app_without_services(self):
        """测试未配置服务时的错误处理"""
        config = Config(api_key="test-key")
        app = VoiceChatApp(config)
        
        # 没有配置 recognizer
        with pytest.raises(NotImplementedError, match="Recognizer not configured"):
            app.speech_to_text(np.array([0] * 16000, dtype=np.int16))
        
        # 没有配置 chat_service
        with pytest.raises(NotImplementedError, match="Chat service not configured"):
            app.chat("hello")
        
        # 没有配置 tts_service - 需要异步测试
        import asyncio
        with pytest.raises(NotImplementedError, match="TTS service not configured"):
            asyncio.run(app.synthesize_speech("test", "/tmp/test.mp3"))
    
    def test_voice_chat_app_with_partial_services(self):
        """测试部分配置服务"""
        config = Config(api_key="test-key")
        mock_recognizer = Mock()
        mock_recognizer.transcribe = Mock(return_value="test transcription")
        
        app = VoiceChatApp(config, recognizer=mock_recognizer)
        
        # recognizer 可用
        result = app.speech_to_text(np.array([0] * 16000, dtype=np.int16))
        assert result == "test transcription"
        
        # chat_service 不可用
        with pytest.raises(NotImplementedError, match="Chat service not configured"):
            app.chat("hello")
    
    def test_recognizer_without_model(self):
        """测试未加载模型时的识别"""
        recognizer = WhisperRecognizer("tiny", "cpu")
        
        # 直接调用 transcribe 会尝试加载模型
        # 这里测试初始化
        assert recognizer.model_name == "tiny"
        assert recognizer.device == "cpu"
        assert recognizer.model is None
    
    def test_glm_service_initialization(self):
        """测试 GLM 服务初始化"""
        service = GLMChatService("test-api-key")
        assert service.client is not None
    
    @pytest.mark.asyncio
    async def test_edge_tts_initialization(self):
        """测试 Edge TTS 初始化"""
        service = EdgeTTSService()
        assert service.voice == "zh-CN-XiaoxiaoNeural"
    
    @pytest.mark.asyncio
    async def test_edge_tts_custom_voice(self):
        """测试自定义语音"""
        service = EdgeTTSService("zh-CN-YunxiNeural")
        assert service.voice == "zh-CN-YunxiNeural"


class TestConversationManagerClassConstants:
    """测试 ConversationManager 类常量"""
    
    def test_exit_keywords_exist(self):
        """测试退出关键词类常量存在"""
        assert hasattr(ConversationManager, 'EXIT_KEYWORDS_ZH')
        assert hasattr(ConversationManager, 'EXIT_KEYWORDS_EN')
        assert "再见" in ConversationManager.EXIT_KEYWORDS_ZH
        assert "退出" in ConversationManager.EXIT_KEYWORDS_ZH
        assert "bye" in ConversationManager.EXIT_KEYWORDS_EN
    
    def test_instance_uses_class_constants(self):
        """测试实例使用类常量"""
        manager = ConversationManager()
        
        # 实例应该能访问类常量
        assert manager.EXIT_KEYWORDS_ZH == ConversationManager.EXIT_KEYWORDS_ZH
