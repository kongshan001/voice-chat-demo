"""
集成测试 - VoiceChatApp
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from main import VoiceChatApp, WhisperRecognizer, GLMChatService, EdgeTTSService
from core import Config
from mocks import MockSpeechRecognizer, MockChatService, MockTTSService


class TestVoiceChatAppWithMocks:
    """使用 Mock 的集成测试"""
    
    @pytest.fixture
    def mock_recognizer(self):
        return MockSpeechRecognizer("测试识别结果")
    
    @pytest.fixture
    def mock_chat_service(self):
        return MockChatService("测试回复")
    
    @pytest.fixture
    def mock_tts_service(self):
        return MockTTSService()
    
    @pytest.fixture
    def config(self):
        return Config(api_key="test-key")
    
    @pytest.fixture
    def app(self, config, mock_recognizer, mock_chat_service, mock_tts_service):
        return VoiceChatApp(
            config=config,
            recognizer=mock_recognizer,
            chat_service=mock_chat_service,
            tts_service=mock_tts_service
        )
    
    def test_init(self, app, config):
        """测试初始化"""
        assert app.config == config
        assert isinstance(app.conversation.history, list)
    
    def test_process_audio_valid(self, app, mock_recognizer):
        """测试有效音频处理"""
        audio = np.array([1] * 2000, dtype=np.int16)
        result = app.process_audio(audio)
        
        assert result == "测试识别结果"
        assert mock_recognizer.transcribe_count == 1
    
    def test_process_audio_none(self, app):
        """测试 None 音频"""
        result = app.process_audio(None)
        assert result is None
    
    def test_process_audio_too_short(self, app):
        """测试太短的音频"""
        audio = np.array([1, 2, 3], dtype=np.int16)
        result = app.process_audio(audio)
        assert result is None
    
    def test_chat(self, app, mock_chat_service):
        """测试对话"""
        response = app.chat("你好")
        
        assert response == "测试回复"
        assert mock_chat_service.chat_count == 1
        assert len(app.conversation.history) == 3  # system + user + assistant
    
    def test_chat_with_stream_callback(self, app, mock_chat_service):
        """测试带流式回调的对话"""
        accumulated = []
        
        def callback(text):
            accumulated.append(text)
        
        response = app.chat("你好", stream_callback=callback)
        
        assert response == "测试回复"
        assert len(accumulated) > 0
        assert "".join(accumulated) == "测试回复"
    
    def test_should_exit(self, app):
        """测试退出检查"""
        assert app.should_exit("再见") is True
        assert app.should_exit("你好") is False
    
    @pytest.mark.asyncio
    async def test_synthesize_speech(self, app, mock_tts_service):
        """测试语音合成"""
        result = await app.synthesize_speech("测试文本", "/tmp/test.mp3")
        
        assert result == "/tmp/test.mp3"
        assert mock_tts_service.synthesize_count == 1
        assert mock_tts_service.last_text == "测试文本"
    
    def test_conversation_history(self, app):
        """测试对话历史累积"""
        app.chat("第一句话")
        app.chat("第二句话")
        
        # system + 2 user + 2 assistant = 5
        assert len(app.conversation.history) == 5


class TestVoiceChatAppErrors:
    """错误处理测试"""
    
    def test_speech_to_text_not_configured(self):
        """测试未配置识别器"""
        from core import ServiceNotConfiguredError
        app = VoiceChatApp(Config(api_key="test-key"))
        
        with pytest.raises(ServiceNotConfiguredError):
            app.speech_to_text(np.array([1] * 2000))
    
    def test_chat_not_configured(self):
        """测试未配置对话服务"""
        from core import ServiceNotConfiguredError
        app = VoiceChatApp(Config(api_key="test-key"))
        
        with pytest.raises(ServiceNotConfiguredError):
            app.chat("hello")
    
    @pytest.mark.asyncio
    async def test_synthesize_not_configured(self):
        """测试未配置 TTS 服务"""
        from core import ServiceNotConfiguredError
        app = VoiceChatApp(Config(api_key="test-key"))
        
        with pytest.raises(ServiceNotConfiguredError):
            await app.synthesize_speech("text", "output.mp3")


class TestConversationFlow:
    """对话流程测试"""
    
    @pytest.fixture
    def app(self):
        config = Config(api_key="test-key")
        return VoiceChatApp(
            config=config,
            recognizer=MockSpeechRecognizer("识别结果"),
            chat_service=MockChatService("AI 回复"),
            tts_service=MockTTSService()
        )
    
    def test_full_conversation_flow(self, app):
        """测试完整对话流程"""
        # 用户说话
        audio = np.array([1] * 2000, dtype=np.int16)
        
        # 1. 语音转文字
        text = app.process_audio(audio)
        assert text == "识别结果"
        
        # 2. 对话
        response = app.chat(text)
        assert response == "AI 回复"
        
        # 3. 检查退出
        assert app.should_exit(response) is False
    
    def test_exit_flow(self, app):
        """测试退出流程"""
        # 用户说再见
        assert app.should_exit("再见") is True
    
    def test_multi_turn_conversation(self, app):
        """测试多轮对话"""
        responses = []
        
        for i in range(3):
            response = app.chat(f"第{i+1}轮消息")
            responses.append(response)
        
        # 3轮对话 + system = 7条历史
        assert len(app.conversation.history) == 7
        
        # 验证历史包含所有对话
        history = app.conversation.history
        assert "第1轮消息" in [m["content"] for m in history if m["role"] == "user"]
        assert "第2轮消息" in [m["content"] for m in history if m["role"] == "user"]
        assert "第3轮消息" in [m["content"] for m in history if m["role"] == "user"]
