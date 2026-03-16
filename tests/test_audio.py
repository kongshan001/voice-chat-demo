"""
单元测试 - AudioProcessor
"""
import pytest
import numpy as np
from core import AudioProcessor


class TestAudioProcessor:
    """音频处理器测试"""
    
    def test_init(self):
        """测试初始化"""
        processor = AudioProcessor()
        assert processor.sample_rate == 16000
        assert processor.channels == 1
    
    def test_init_with_custom_params(self):
        """测试自定义参数"""
        processor = AudioProcessor(44100, 2)
        assert processor.sample_rate == 44100
        assert processor.channels == 2
    
    def test_normalize_audio_int16(self):
        """测试标准化 int16 音频"""
        processor = AudioProcessor()
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        result = processor.normalize_audio(audio)
        
        assert result.dtype == np.int16
        assert np.abs(result[0]) <= 32767
    
    def test_normalize_audio_already_int16(self):
        """测试已经是 int16 的音频"""
        processor = AudioProcessor()
        audio = np.array([1000, -2000, 500], dtype=np.int16)
        result = processor.normalize_audio(audio)
        
        assert result.dtype == np.int16
        assert np.array_equal(result, audio)
    
    def test_is_valid_audio_none(self):
        """测试无效音频 - None"""
        processor = AudioProcessor()
        assert processor.is_valid_audio(None) is False
    
    def test_is_valid_audio_too_short(self):
        """测试无效音频 - 太短"""
        processor = AudioProcessor()
        audio = np.array([1, 2, 3], dtype=np.int16)
        assert processor.is_valid_audio(audio, min_samples=1600) is False
    
    def test_is_valid_audio_valid(self):
        """测试有效音频"""
        processor = AudioProcessor()
        audio = np.array([1] * 2000, dtype=np.int16)
        assert processor.is_valid_audio(audio, min_samples=1600) is True
    
    def test_audio_to_bytes(self):
        """测试音频转字节"""
        processor = AudioProcessor()
        audio = np.array([1, 2, 3, 4], dtype=np.int16)
        result = processor.audio_to_bytes(audio)
        
        assert isinstance(result, bytes)
        assert len(result) == 8  # 4 * 2 bytes
    
    def test_bytes_to_audio(self):
        """测试字节转音频"""
        processor = AudioProcessor()
        data = b'\x00\x01\x02\x03\x04\x05\x06\x07'
        result = processor.bytes_to_audio(data)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int16
        assert len(result) == 4
    
    def test_audio_round_trip(self):
        """测试音频往返转换"""
        processor = AudioProcessor()
        original = np.array([100, -200, 300, -400], dtype=np.int16)
        
        bytes_data = processor.audio_to_bytes(original)
        restored = processor.bytes_to_audio(bytes_data)
        
        assert np.array_equal(original, restored)
    
    # ===== 边界条件测试 =====
    
    def test_init_invalid_sample_rate(self):
        """测试无效采样率"""
        with pytest.raises(ValueError, match="采样率必须为正数"):
            AudioProcessor(sample_rate=0)
    
    def test_init_invalid_channels(self):
        """测试无效通道数"""
        with pytest.raises(ValueError, match="通道数必须为正数"):
            AudioProcessor(channels=-1)
    
    def test_normalize_audio_empty(self):
        """测试空音频"""
        processor = AudioProcessor()
        with pytest.raises(ValueError, match="音频数据为空"):
            processor.normalize_audio(np.array([], dtype=np.int16))
    
    def test_normalize_audio_float64(self):
        """测试 float64 音频"""
        processor = AudioProcessor()
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float64)
        result = processor.normalize_audio(audio)
        assert result.dtype == np.int16
    
    def test_normalize_audio_with_clipping(self):
        """测试音频裁剪"""
        processor = AudioProcessor()
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)  # 超出范围
        result = processor.normalize_audio(audio)
        assert result.dtype == np.int16
        # 检查裁剪后的值
        assert np.max(np.abs(result)) <= 32767
    
    def test_normalize_audio_unsupported_dtype(self):
        """测试不支持的数据类型"""
        processor = AudioProcessor()
        audio = np.array([1, 2, 3], dtype=np.uint8)
        with pytest.raises(ValueError, match="不支持的音频数据类型"):
            processor.normalize_audio(audio)
    
    def test_is_valid_audio_not_numpy(self):
        """测试非 numpy 数组"""
        processor = AudioProcessor()
        assert processor.is_valid_audio([1, 2, 3]) is False
    
    def test_audio_to_bytes_empty(self):
        """测试空音频转字节"""
        processor = AudioProcessor()
        with pytest.raises(ValueError, match="音频数据为空"):
            processor.audio_to_bytes(np.array([], dtype=np.int16))
    
    def test_bytes_to_audio_empty(self):
        """测试空字节转音频"""
        processor = AudioProcessor()
        with pytest.raises(ValueError, match="字节数据为空"):
            processor.bytes_to_audio(b'')
