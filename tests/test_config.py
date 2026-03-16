"""
单元测试 - Config
"""
import pytest
from core import Config


class TestConfig:
    """配置测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        config = Config()
        
        assert config.api_key == ""
        assert config.whisper_model == "base"
        assert config.whisper_device == "cpu"
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.vad_aggressiveness == 3
    
    def test_init_custom(self):
        """测试自定义初始化"""
        config = Config(
            api_key="test-key-123",
            whisper_model="small",
            whisper_device="cuda",
            sample_rate=44100,
            channels=2,
            vad_aggressiveness=2
        )
        
        assert config.api_key == "test-key-123"
        assert config.whisper_model == "small"
        assert config.whisper_device == "cuda"
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.vad_aggressiveness == 2
    
    def test_is_configured_empty_key(self):
        """测试未配置 - 空 key"""
        config = Config(api_key="")
        assert config.is_configured is False
    
    def test_is_configured_placeholder_key(self):
        """测试未配置 - 占位符 key"""
        config = Config(api_key="your-api-key-here")
        assert config.is_configured is False
    
    def test_is_configured_valid_key(self):
        """测试已配置 - 有效 key"""
        config = Config(api_key="test-key-123")
        assert config.is_configured is True
    
    def test_is_configured_env_var_style(self):
        """测试环境变量样式 key"""
        config = Config(api_key="sk-abc123xyz")
        assert config.is_configured is True
