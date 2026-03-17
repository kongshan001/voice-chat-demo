"""核心模块测试"""
import pytest
from core import ConversationManager, AudioProcessor, TextProcessor, Config


class TestConversationManagerEdgeCases:
    """对话管理器边界情况测试"""

    def test_trim_history_preserves_system(self):
        """测试修剪历史保留 system 消息"""
        conv = ConversationManager(max_history=2)
        # 添加多轮对话
        for i in range(10):
            conv.add_user_message(f"User {i}")
            conv.add_assistant_message(f"Assistant {i}")
        
        messages = conv.get_messages()
        # 第一条应该是 system
        assert messages[0]["role"] == "system"

    def test_exit_keyword_case_sensitivity(self):
        """测试退出关键词大小写"""
        conv = ConversationManager()
        assert conv.should_exit("STOP") == True
        assert conv.should_exit("EXIT") == True
        assert conv.should_exit("Quit") == True


class TestAudioProcessorEdgeCasesExtended:
    """音频处理器扩展边界测试"""

    def test_normalize_audio_preserves_data(self):
        """测试归一化保留数据"""
        import numpy as np
        proc = AudioProcessor()
        audio = np.array([0, 1000, -1000, 500], dtype=np.int16)
        result = proc.normalize_audio(audio)
        assert result.dtype == np.int16
        assert len(result) == len(audio)

    def test_is_valid_audio_zero_min_samples(self):
        """测试最小样本检查"""
        import numpy as np
        proc = AudioProcessor()
        audio = np.array([1, 2, 3], dtype=np.int16)
        assert proc.is_valid_audio(audio, min_samples=3) == True
        assert proc.is_valid_audio(audio, min_samples=4) == False


class TestConfigExtended:
    """配置扩展测试"""

    def test_validate_all_valid(self):
        """测试完全有效配置"""
        config = Config(
            api_key="test-key-12345",
            whisper_model="base",
            tts_voice="zh-CN-XiaoxiaoNeural"
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_multiple_errors(self):
        """测试多错误验证"""
        config = Config(
            api_key="your-api-key-here",
            whisper_model="invalid_model",
            tts_voice="invalid_voice"
        )
        errors = config.validate()
        assert len(errors) >= 2
