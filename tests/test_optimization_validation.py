"""
新增测试用例 - 验证优化后的代码
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from services import WhisperRecognizer, GLMChatService, EdgeTTSService
from core import Config


class TestWhisperRecognizerValidation:
    """WhisperRecognizer 验证测试"""
    
    def test_init_empty_model_name_raises(self):
        """空模型名称应抛出 ValueError"""
        with pytest.raises(ValueError, match="模型名称不能为空"):
            WhisperRecognizer("")
    
    def test_init_invalid_device_raises(self):
        """无效设备应抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的设备"):
            WhisperRecognizer("base", "invalid")
    
    def test_transcribe_none_audio_raises(self):
        """None 音频应抛出 ValueError"""
        recognizer = WhisperRecognizer("base")
        with pytest.raises(ValueError, match="音频数据不能为空"):
            recognizer.transcribe(None)
    
    def test_transcribe_empty_audio_raises(self):
        """空音频应抛出 ValueError"""
        recognizer = WhisperRecognizer("base")
        with pytest.raises(ValueError, match="音频数据不能为空"):
            recognizer.transcribe(np.array([]))
    
    def test_transcribe_invalid_type_raises(self):
        """非 numpy array 应抛出 ValueError"""
        recognizer = WhisperRecognizer("base")
        with pytest.raises(ValueError, match="音频必须是 numpy array"):
            recognizer.transcribe([1, 2, 3])
    
    @patch('services.WhisperRecognizer.load_model')
    def test_transcribe_with_mock_model(self, mock_load):
        """使用 mock 模型测试转录"""
        recognizer = WhisperRecognizer("base")
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "测试"
        mock_model.transcribe.return_value = ([mock_segment], None)
        recognizer.model = mock_model
        
        result = recognizer.transcribe(np.array([1, 2, 3], dtype=np.int16))
        assert result == "测试"


class TestGLMChatServiceValidation:
    """GLMChatService 验证测试"""
    
    def test_init_empty_api_key_raises(self):
        """空 API Key 应抛出 ValueError"""
        with pytest.raises(ValueError, match="API Key 不能为空"):
            GLMChatService("")
    
    def test_init_whitespace_api_key_raises(self):
        """空白 API Key 应抛出 ValueError"""
        with pytest.raises(ValueError, match="API Key 不能仅包含空白字符"):
            GLMChatService("   ")
    
    def test_init_none_api_key_raises(self):
        """None API Key 应抛出 ValueError"""
        with pytest.raises(ValueError, match="API Key 不能为空"):
            GLMChatService(None)  # type: ignore
    
    def test_init_invalid_type_raises(self):
        """非字符串 API Key 应抛出 ValueError"""
        with pytest.raises(ValueError, match="API Key 必须是字符串"):
            GLMChatService(12345)  # type: ignore


class TestConfigEnhancedValidation:
    """Config 增强验证测试"""
    
    def test_validate_negative_max_history(self):
        """负数 max_history 应返回错误"""
        config = Config(api_key="test-key", max_history=-1)
        errors = config.validate()
        assert any("max_history" in err for err in errors)
    
    def test_validate_invalid_whisper_device(self):
        """无效 whisper_device 应返回错误"""
        config = Config(api_key="test-key", whisper_device="invalid")
        errors = config.validate()
        assert any("设备" in err for err in errors)
    
    def test_validate_valid_device_cuda(self):
        """cuda 设备应通过验证"""
        config = Config(api_key="test-key", whisper_device="cuda")
        errors = config.validate()
        assert not any("设备" in err for err in errors)


class TestEdgeTTSServiceValidation:
    """EdgeTTSService 验证测试"""
    
    @pytest.mark.asyncio
    async def test_synthesize_empty_text_raises(self):
        """空文本应抛出 ValueError"""
        tts = EdgeTTSService()
        with pytest.raises(ValueError, match="合成文本不能为空"):
            await tts.synthesize("", "output.mp3")
    
    @pytest.mark.asyncio
    async def test_synthesize_whitespace_only_raises(self):
        """仅空白文本应抛出 ValueError"""
        tts = EdgeTTSService()
        with pytest.raises(ValueError, match="合成文本不能为空"):
            await tts.synthesize("   ", "output.mp3")
