"""测试配置增强功能"""
import pytest
from core import Config


class TestConfigEnhancements:
    """配置增强功能测试"""
    
    def test_to_dict_masks_api_key(self):
        """测试 to_dict 方法会 mask API Key"""
        config = Config(api_key="secret-key-123")
        d = config.to_dict()
        
        assert d["api_key"] == "***"
        assert d["whisper_model"] == "base"
        assert d["tts_voice"] == "zh-CN-XiaoxiaoNeural"
    
    def test_to_dict_empty_api_key(self):
        """测试 to_dict 处理空 API Key"""
        config = Config(api_key="")
        d = config.to_dict()
        
        assert d["api_key"] == ""
    
    def test_to_dict_contains_all_fields(self):
        """测试 to_dict 包含所有配置字段"""
        config = Config(
            api_key="test-key",
            whisper_model="tiny",
            whisper_device="cuda",
            sample_rate=22050,
            channels=2,
            vad_aggressiveness=2,
            tts_voice="en-US-JennyNeural",
            temperature=0.5,
            max_history=10,
            log_level="DEBUG"
        )
        d = config.to_dict()
        
        assert len(d) == 10
        assert d["whisper_model"] == "tiny"
        assert d["whisper_device"] == "cuda"
        assert d["sample_rate"] == 22050
        assert d["channels"] == 2
        assert d["vad_aggressiveness"] == 2
        assert d["tts_voice"] == "en-US-JennyNeural"
        assert d["temperature"] == 0.5
        assert d["max_history"] == 10
        assert d["log_level"] == "DEBUG"
    
    def test_from_env_defaults(self):
        """测试 from_env 使用默认值"""
        import os
        # 确保环境变量未设置
        for key in ["ZHIPU_API_KEY", "WHISPER_MODEL", "SAMPLE_RATE"]:
            if key in os.environ:
                del os.environ[key]
        
        config = Config.from_env()
        
        assert config.api_key == ""
        assert config.whisper_model == "base"
        assert config.sample_rate == 16000
    
    def test_from_env_with_env_vars(self, monkeypatch):
        """测试 from_env 读取环境变量"""
        monkeypatch.setenv("ZHIPU_API_KEY", "env-api-key")
        monkeypatch.setenv("WHISPER_MODEL", "small")
        monkeypatch.setenv("SAMPLE_RATE", "22050")
        monkeypatch.setenv("TEMPERATURE", "0.9")
        
        config = Config.from_env()
        
        assert config.api_key == "env-api-key"
        assert config.whisper_model == "small"
        assert config.sample_rate == 22050
        assert config.temperature == 0.9
