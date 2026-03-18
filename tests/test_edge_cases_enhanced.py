"""新增测试用例 - 增强边缘用例覆盖"""
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from main import VoiceChatApp
from core import Config, ConversationManager, AudioProcessor, TextProcessor
from services import WhisperRecognizer, GLMChatService, EdgeTTSService
from core import TranscriptionError, ChatError, SynthesisError


class TestVoiceChatAppEdgeCases:
    """VoiceChatApp 边缘用例测试"""
    
    def test_voice_chat_app_with_empty_config(self):
        """测试空配置"""
        config = Config(api_key="")
        app = VoiceChatApp(config)
        
        assert app.config.api_key == ""
        assert app.conversation is not None
    
    def test_voice_chat_app_with_custom_history(self):
        """测试自定义历史记录数"""
        config = Config(api_key="test", max_history=5)
        app = VoiceChatApp(config)
        
        assert app.conversation.max_history == 5
    
    def test_speech_to_text_with_recognizer_exception(self):
        """测试识别器抛出异常"""
        config = Config(api_key="test-key")
        mock_recognizer = Mock()
        mock_recognizer.transcribe = Mock(side_effect=RuntimeError("Model error"))
        
        app = VoiceChatApp(config, recognizer=mock_recognizer)
        
        audio = np.array([0] * 1600, dtype=np.int16)
        with pytest.raises(TranscriptionError) as exc_info:
            app.speech_to_text(audio)
        
        assert "语音识别失败" in str(exc_info.value)
    
    def test_chat_with_service_exception(self):
        """测试对话服务异常"""
        config = Config(api_key="test-key")
        mock_service = Mock()
        mock_service.chat = Mock(side_effect=RuntimeError("API error"))
        
        app = VoiceChatApp(config, chat_service=mock_service)
        
        with pytest.raises(Exception):
            app.chat("hello")
    
    def test_synthesize_speech_without_service(self):
        """测试无 TTS 服务"""
        import asyncio
        
        config = Config(api_key="test-key")
        app = VoiceChatApp(config)
        
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(Exception):
                loop.run_until_complete(app.synthesize_speech("test", "/tmp/test.mp3"))
        finally:
            loop.close()


class TestConversationManagerEdgeCases:
    """ConversationManager 边缘用例测试"""
    
    def test_history_trim_preserves_system(self):
        """测试修剪保留 system 消息"""
        manager = ConversationManager(max_history=2)
        
        # 添加多条消息超过限制
        for i in range(10):
            manager.add_user_message(f"User message {i}")
            manager.add_assistant_message(f"Assistant response {i}")
        
        messages = manager.get_messages()
        
        # 第一条应该是 system
        assert messages[0]["role"] == "system"
    
    def test_clear_history_after_trim(self):
        """测试清空历史后修剪"""
        manager = ConversationManager(max_history=5)
        
        manager.add_user_message("test")
        manager.add_assistant_message("response")
        manager.clear_history()
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class TestTextProcessorEdgeCases:
    """TextProcessor 边缘用例测试"""
    
    def test_is_chinese_with_numbers(self):
        """测试包含数字的中文"""
        processor = TextProcessor()
        
        assert processor.is_chinese("今天2024年") is True
    
    def test_is_chinese_with_punctuation(self):
        """测试包含标点的中文"""
        processor = TextProcessor()
        
        assert processor.is_chinese("你好，世界！") is True
    
    def test_truncate_text_with_unicode(self):
        """测试 Unicode 截断"""
        processor = TextProcessor()
        
        result = processor.truncate_text("你好👋世界", max_length=3)
        assert len(result) <= 6  # 包含省略号


class TestAudioProcessorEdgeCases:
    """AudioProcessor 边缘用例测试"""
    
    def test_normalize_with_zero_audio(self):
        """测试归一化零音频"""
        processor = AudioProcessor()
        
        audio = np.zeros(1600, dtype=np.int16)
        result = processor.normalize_audio(audio)
        
        assert result.dtype == np.int16
        assert len(result) == 1600
    
    def test_is_valid_audio_exact_min_samples(self):
        """测试刚好达到最小样本数"""
        processor = AudioProcessor()
        
        audio = np.array([0] * 1600, dtype=np.int16)
        assert processor.is_valid_audio(audio, min_samples=1600) is True
    
    def test_bytes_audio_roundtrip(self):
        """测试字节转换往返"""
        processor = AudioProcessor()
        
        original = np.array([0, 100, -100, 32767], dtype=np.int16)
        bytes_data = processor.audio_to_bytes(original)
        restored = processor.bytes_to_audio(bytes_data)
        
        assert np.array_equal(original, restored)


class TestConfigEdgeCases:
    """Config 边缘用例测试"""
    
    def test_validate_with_warnings(self):
        """测试带警告的配置"""
        config = Config(
            api_key="valid-key",
            whisper_model="base",
            sample_rate=22050  # 非标准但有效
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_from_env_with_missing_vars(self):
        """测试缺少环境变量"""
        import os
        # 确保没有这些环境变量
        original = os.environ.copy()
        for key in ["ZHIPU_API_KEY", "WHISPER_MODEL", "SAMPLE_RATE"]:
            os.environ.pop(key, None)
        
        config = Config.from_env()
        
        # 恢复
        os.environ.clear()
        os.environ.update(original)
        
        assert config.api_key == ""
        assert config.whisper_model == "base"
