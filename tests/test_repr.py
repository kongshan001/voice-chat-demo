"""
测试 __repr__ 方法
"""
import pytest
from core import ConversationManager, AudioProcessor, TextProcessor, Config


class TestReprMethods:
    """测试类的 __repr__ 方法"""
    
    def test_conversation_manager_repr(self):
        """测试 ConversationManager 的 __repr__"""
        cm = ConversationManager(max_history=10)
        repr_str = repr(cm)
        assert "ConversationManager" in repr_str
        assert "max_history=10" in repr_str
    
    def test_audio_processor_repr(self):
        """测试 AudioProcessor 的 __repr__"""
        ap = AudioProcessor(sample_rate=16000, channels=2)
        repr_str = repr(ap)
        assert "AudioProcessor" in repr_str
        assert "sample_rate=16000" in repr_str
        assert "channels=2" in repr_str
    
    def test_text_processor_repr(self):
        """测试 TextProcessor 的 __repr__"""
        tp = TextProcessor()
        repr_str = repr(tp)
        assert "TextProcessor" in repr_str
    
    def test_config_repr_with_api_key(self):
        """测试 Config 的 __repr__ (有 API Key)"""
        config = Config(api_key="test-key-123", whisper_model="base")
        repr_str = repr(config)
        assert "Config" in repr_str
        assert "whisper_model=base" in repr_str
        # API Key 应该被 mask
        assert "***" in repr_str
    
    def test_config_repr_without_api_key(self):
        """测试 Config 的 __repr__ (无 API Key)"""
        config = Config(api_key="", whisper_model="tiny")
        repr_str = repr(config)
        assert "Config" in repr_str
        assert "(empty)" in repr_str
