"""
测试 Config 新增的 TTS 语音选项
"""
import pytest
from core import Config


class TestConfigTTSVoicesExtended:
    """扩展 TTS 语音测试"""
    
    def test_all_tts_voices_valid(self):
        """测试所有 TTS 语音都有效"""
        for voice_id in Config.TTS_VOICES.keys():
            config = Config(tts_voice=voice_id)
            errors = config.validate()
            # 只检查 TTS 语音相关错误
            tts_errors = [e for e in errors if "TTS" in e]
            assert len(tts_errors) == 0, f"TTS voice {voice_id} should be valid: {tts_errors}"
    
    def test_tts_voices_count(self):
        """测试 TTS 语音数量"""
        assert len(Config.TTS_VOICES) >= 8, "Should have at least 8 TTS voices"
    
    def test_tts_voices_include_both_languages(self):
        """测试包含中英文语音"""
        zh_voices = [v for v in Config.TTS_VOICES.keys() if v.startswith("zh-")]
        en_voices = [v for v in Config.TTS_VOICES.keys() if v.startswith("en-")]
        
        assert len(zh_voices) >= 4, "Should have at least 4 Chinese voices"
        assert len(en_voices) >= 4, "Should have at least 4 English voices"


class TestMockAudioRecorderAdapter:
    """测试 MockAudioRecorderAdapter"""
    
    def test_adapter_init(self):
        """测试适配器初始化"""
        from mocks import MockAudioRecorderAdapter
        import numpy as np
        
        mock_audio = np.array([1, 2, 3], dtype=np.int16)
        adapter = MockAudioRecorderAdapter(mock_audio=mock_audio)
        
        assert adapter.mock_audio is mock_audio
        assert adapter.record_count == 0
    
    def test_adapter_record(self):
        """测试录音方法"""
        from mocks import MockAudioRecorderAdapter
        import numpy as np
        
        mock_audio = np.array([1, 2, 3], dtype=np.int16)
        adapter = MockAudioRecorderAdapter(mock_audio=mock_audio)
        
        result = adapter.record(1.0)
        
        assert np.array_equal(result, mock_audio)
        assert adapter.record_count == 1
    
    def test_adapter_default_audio(self):
        """测试默认音频生成"""
        from mocks import MockAudioRecorderAdapter
        
        adapter = MockAudioRecorderAdapter()
        
        assert adapter.mock_audio is not None
        assert len(adapter.mock_audio) > 0
        assert adapter.record_count == 0
