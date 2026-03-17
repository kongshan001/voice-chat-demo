"""
VoiceChatApp 增强测试
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from main import VoiceChatApp
from core import Config, AudioProcessingError, TranscriptionError


class TestVoiceChatAppEnhancements:
    """VoiceChatApp 增强测试"""
    
    def test_process_audio_with_none(self):
        """测试处理 None 音频"""
        config = Config(api_key="test-key")
        app = VoiceChatApp(config)
        
        result = app.process_audio(None)
        assert result is None
    
    def test_process_audio_with_too_short(self):
        """测试处理过短音频"""
        config = Config(api_key="test-key")
        app = VoiceChatApp(config)
        
        # 小于最小采样数
        result = app.process_audio(np.array([0] * 100, dtype=np.int16))
        assert result is None
    
    def test_process_audio_valid(self):
        """测试有效音频处理"""
        config = Config(api_key="test-key")
        mock_recognizer = Mock()
        mock_recognizer.transcribe = Mock(return_value="test text")
        
        app = VoiceChatApp(config, recognizer=mock_recognizer)
        
        # 有效音频
        audio = np.array([0] * 16000, dtype=np.int16)
        result = app.process_audio(audio)
        assert result == "test text"
        mock_recognizer.transcribe.assert_called_once()
    
    def test_process_audio_raises_transcription_error(self):
        """测试识别异常"""
        config = Config(api_key="test-key")
        mock_recognizer = Mock()
        mock_recognizer.transcribe = Mock(side_effect=Exception("识别失败"))
        
        app = VoiceChatApp(config, recognizer=mock_recognizer)
        
        audio = np.array([0] * 16000, dtype=np.int16)
        with pytest.raises(TranscriptionError, match="语音识别失败"):
            app.process_audio(audio)
    
    def test_should_exit_with_various_keywords(self):
        """测试各种退出关键词"""
        config = Config(api_key="test-key")
        app = VoiceChatApp(config)
        
        assert app.should_exit("再见") is True
        assert app.should_exit("我要退出了") is True
        assert app.should_exit("stop now") is True
        assert app.should_exit("hello world") is False
    
    def test_chat_with_stream_callback(self):
        """测试带流式回调的对话"""
        config = Config(api_key="test-key")
        mock_chat_service = Mock()
        mock_chat_service.chat = Mock(return_value="Hello response")
        
        app = VoiceChatApp(config, chat_service=mock_chat_service)
        
        callback_results = []
        def callback(text):
            callback_results.append(text)
        
        result = app.chat("Hi", stream_callback=callback)
        
        assert result == "Hello response"
        mock_chat_service.chat.assert_called_once()
