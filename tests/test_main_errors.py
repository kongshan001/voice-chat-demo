"""测试 main.py 中改进的错误处理"""
import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock


class TestPlayAudioErrorHandling:
    """测试 play_audio 错误处理"""
    
    def test_play_audio_file_not_found(self):
        """测试文件不存在时的错误"""
        from main import play_audio
        
        with pytest.raises(FileNotFoundError):
            play_audio("/nonexistent/file.mp3")
    
    def test_play_audio_no_file_raises(self):
        """测试空路径错误"""
        from main import play_audio
        
        with pytest.raises(FileNotFoundError):
            play_audio("")


class TestGLMServiceImport:
    """测试 GLM 服务模块级导入"""
    
    def test_requests_imported(self):
        """验证 requests 已导入"""
        import main
        assert hasattr(main, 'requests')
    
    def test_aiohttp_imported(self):
        """验证 aiohttp 已导入"""
        import main
        assert hasattr(main, 'aiohttp')
