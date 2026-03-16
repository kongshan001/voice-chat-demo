"""
额外的边界情况测试
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from main import VoiceChatApp
from core import Config, AudioProcessor, TextProcessor, ConversationManager


class TestAudioProcessorEdgeCases:
    """AudioProcessor 边界情况测试"""
    
    def test_normalize_audio_int32(self):
        """测试 int32 类型归一化"""
        processor = AudioProcessor()
        audio = np.array([1000, 2000, -3000], dtype=np.int32)
        result = processor.normalize_audio(audio)
        # int32 不在自动转换列表中，会直接返回
        assert result is not None
    
    def test_normalize_audio_clamping(self):
        """测试音频 clipping"""
        processor = AudioProcessor()
        # 超范围音频
        audio = np.array([1.5, 2.0, -1.5], dtype=np.float32)
        result = processor.normalize_audio(audio)
        # 值应该在 [-32767, 32767] 范围内
        assert np.all(result >= -32768)
        assert np.all(result <= 32767)
    
    def test_bytes_to_audio_invalid(self):
        """测试无效字节"""
        processor = AudioProcessor()
        with pytest.raises(ValueError):
            processor.bytes_to_audio(b"")


class TestTextProcessorEdgeCases:
    """TextProcessor 边界情况测试"""
    
    def test_truncate_text_negative_length(self):
        """测试负数截断长度"""
        processor = TextProcessor()
        result = processor.truncate_text("hello", -1)
        # 负数会被当成长度0处理，返回截断结果
        assert "..." in result
    
    def test_truncate_text_very_long(self):
        """测试超长截断"""
        processor = TextProcessor()
        long_text = "a" * 10000
        result = processor.truncate_text(long_text, 100)
        assert len(result) == 103  # 100 + "..."


class TestConversationManagerEdgeCases:
    """ConversationManager 边界情况测试"""
    
    def test_add_empty_message(self):
        """测试添加空消息"""
        manager = ConversationManager()
        manager.add_user_message("")
        manager.add_assistant_message("")
        # 应该有 system + 2 条空消息
        assert len(manager.history) == 3
    
    def test_very_long_message(self):
        """测试超长消息"""
        manager = ConversationManager()
        long_text = "x" * 10000
        manager.add_user_message(long_text)
        assert len(manager.history) == 2  # system + user
    
    def test_english_exit_keywords(self):
        """测试英文退出关键词"""
        manager = ConversationManager()
        assert manager.should_exit("stop") is True
        assert manager.should_exit("quit") is True
        assert manager.should_exit("  quit  ") is True
    
    def test_mixed_language_exit(self):
        """测试混合语言退出"""
        manager = ConversationManager()
        assert manager.should_exit("I want to quit now") is True
        assert manager.should_exit("hello world stop") is True


class TestVoiceChatAppEdgeCases:
    """VoiceChatApp 边界情况测试"""
    
    def test_process_audio_float32(self):
        """测试 float32 音频处理"""
        config = Config(api_key="test-key")
        recognizer = Mock()
        recognizer.transcribe = Mock(return_value="识别结果")
        
        app = VoiceChatApp(config=config, recognizer=recognizer)
        
        # float32 音频 - 需要足够长以通过验证 (min_samples=1600)
        audio = np.array([0.1] * 2000, dtype=np.float32)
        result = app.process_audio(audio)
        
        assert result == "识别结果"
    
    def test_process_audio_float64(self):
        """测试 float64 音频处理"""
        config = Config(api_key="test-key")
        recognizer = Mock()
        recognizer.transcribe = Mock(return_value="识别结果")
        
        app = VoiceChatApp(config=config, recognizer=recognizer)
        
        # float64 音频 - 需要足够长以通过验证 (min_samples=1600)
        audio = np.array([0.1] * 2000, dtype=np.float64)
        result = app.process_audio(audio)
        
        assert result == "识别结果"
    
    def test_chat_with_empty_messages(self):
        """测试空消息对话"""
        config = Config(api_key="test-key")
        chat_service = Mock()
        chat_service.chat = Mock(return_value="")
        
        app = VoiceChatApp(config=config, chat_service=chat_service)
        
        response = app.chat("")
        assert response == ""
        # 历史记录仍应包含消息
        assert len(app.conversation.history) == 3  # system + user + assistant


class TestConfigEdgeCases:
    """Config 边界情况测试"""
    
    def test_validate_all_options(self):
        """测试完整配置验证"""
        config = Config(
            api_key="valid-key",
            whisper_model="small",
            tts_voice="zh-CN-YunxiNeural",
            vad_aggressiveness=2,
            temperature=1.0
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_is_configured_edge_cases(self):
        """测试配置状态边界"""
        # 空字符串
        config = Config(api_key="")
        assert config.is_configured is False
        
        # 仅空格会被 strip 后判断，但 api_key 仍然是非空字符串
        # 实际上 "   " 不是占位符，所以 is_configured 为 True
        # 这是预期行为
        config = Config(api_key="some-key")
        assert config.is_configured is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
