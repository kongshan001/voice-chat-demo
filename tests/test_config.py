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
    
    def test_tts_voices(self):
        """测试 TTS 语音选项"""
        config = Config()
        assert "zh-CN-XiaoxiaoNeural" in config.TTS_VOICES
        assert len(config.TTS_VOICES) > 0
    
    def test_whisper_models(self):
        """测试 Whisper 模型选项"""
        config = Config()
        assert "tiny" in config.WHISPER_MODELS
        assert "base" in config.WHISPER_MODELS
    
    def test_validate_valid_config(self):
        """测试有效配置"""
        config = Config(api_key="test-key-123")
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validate_empty_key(self):
        """测试空 API Key"""
        config = Config(api_key="")
        errors = config.validate()
        assert len(errors) > 0
        assert "API Key" in errors[0]
    
    def test_validate_placeholder_key(self):
        """测试占位符 API Key"""
        config = Config(api_key="your-api-key-here")
        errors = config.validate()
        assert len(errors) > 0
    
    def test_validate_invalid_whisper_model(self):
        """测试无效的 Whisper 模型"""
        config = Config(api_key="test", whisper_model="invalid")
        errors = config.validate()
        assert any("Whisper" in e for e in errors)
    
    def test_validate_invalid_tts_voice(self):
        """测试无效的 TTS 语音"""
        config = Config(api_key="test", tts_voice="invalid-voice")
        errors = config.validate()
        assert any("TTS" in e for e in errors)
    
    def test_validate_invalid_vad_aggressiveness(self):
        """测试无效的 VAD 激进度"""
        config = Config(api_key="test", vad_aggressiveness=10)
        errors = config.validate()
        assert any("VAD" in e for e in errors)
    
    def test_validate_invalid_temperature(self):
        """测试无效的温度参数"""
        config = Config(api_key="test", temperature=5.0)
        errors = config.validate()
        assert any("温度" in e for e in errors)
