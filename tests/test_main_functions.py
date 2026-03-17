"""测试 main.py 中的 play_audio 和相关函数"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from main import play_audio, simple_record, WhisperRecognizer, GLMChatService, EdgeTTSService


class TestPlayAudioFunction:
    """测试 play_audio 函数"""

    def test_play_audio_file_not_found_raises(self):
        """测试文件不存在时抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            play_audio("/nonexistent/file.mp3")

    def test_play_audio_with_nonexistent_path_raises(self):
        """测试路径为 None 时正确处理"""
        # 验证 os.path.exists 调用
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                play_audio("dummy.mp3")


class TestSimpleRecord:
    """测试 simple_record 函数"""

    def test_record_function_importable(self):
        """测试 simple_record 函数可导入"""
        # 函数已通过上面的导入验证
        assert callable(simple_record)


class TestWhisperRecognizer:
    """测试 WhisperRecognizer 类"""

    def test_init_default_params(self):
        """测试默认初始化参数"""
        recognizer = WhisperRecognizer()
        assert recognizer.model_name == "base"
        assert recognizer.device == "cpu"
        assert recognizer.model is None

    def test_init_custom_params(self):
        """测试自定义参数"""
        recognizer = WhisperRecognizer(model_name="tiny", device="cuda")
        assert recognizer.model_name == "tiny"
        assert recognizer.device == "cuda"

    def test_model_not_loaded_by_default(self):
        """测试默认不加载模型"""
        recognizer = WhisperRecognizer()
        assert recognizer.model is None


class TestGLMChatServiceInit:
    """测试 GLMChatService 初始化"""

    def test_service_initialization(self):
        """测试服务可初始化"""
        # 由于 ZhipuAI 在方法内部导入，我们通过检查类是否存在来验证
        assert GLMChatService is not None
        # 验证初始化需要 api_key 参数
        import inspect
        sig = inspect.signature(GLMChatService.__init__)
        params = list(sig.parameters.keys())
        assert 'api_key' in params


class TestEdgeTTSServiceInit:
    """测试 EdgeTTSService 初始化"""

    def test_service_default_voice(self):
        """测试默认语音"""
        service = EdgeTTSService()
        assert service.voice == "zh-CN-XiaoxiaoNeural"

    def test_service_custom_voice(self):
        """测试自定义语音"""
        service = EdgeTTSService(voice="en-US-JennyNeural")
        assert service.voice == "en-US-JennyNeural"
