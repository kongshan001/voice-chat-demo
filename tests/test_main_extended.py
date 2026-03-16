"""
测试 main.py 中的录音和播放功能
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import os


class TestRecordFunctions:
    """测试录音函数"""
    
    def test_record_functions_exist(self):
        """测试录音函数存在"""
        from main import record_with_vad, simple_record
        assert callable(record_with_vad)
        assert callable(simple_record)


class TestPlayAudio:
    """测试播放函数"""
    
    def test_play_audio_exists(self):
        """测试播放函数存在"""
        from main import play_audio
        assert callable(play_audio)


class TestMainEntryPoint:
    """测试主入口"""
    
    def test_main_imports(self):
        """测试 main 可导入"""
        import main
        assert hasattr(main, 'main')
        assert callable(main.main)
    
    def test_config_module_exists(self):
        """测试配置模块存在"""
        from main import _config
        assert _config is not None
        assert hasattr(_config, 'api_key')


class TestConfigureServices:
    """测试 configure_services"""
    
    def test_configure_multiple_times(self):
        """测试多次配置"""
        from main import configure_services
        from mocks import MockSpeechRecognizer, MockChatService
        
        # 第一次配置
        mock1 = MockSpeechRecognizer()
        configure_services(recognizer=mock1)
        
        # 第二次配置
        mock2 = MockSpeechRecognizer()
        mock3 = MockChatService()
        configure_services(recognizer=mock2, chat_service=mock3)
        
        # 无异常即可


class TestGlobalVariables:
    """测试全局变量"""
    
    def test_global_services_variables_exist(self):
        """测试服务变量存在"""
        import main
        
        assert hasattr(main, '_speech_recognizer')
        assert hasattr(main, '_chat_service')
        assert hasattr(main, '_tts_service')
        assert hasattr(main, '_config')


class TestIntegrationScenarios:
    """集成场景测试"""
    
    def test_app_with_mock_full_flow(self):
        """完整流程测试"""
        from main import VoiceChatApp
        from core import Config
        from mocks import MockSpeechRecognizer, MockChatService, MockTTSService
        
        # 创建应用
        app = VoiceChatApp(
            config=Config(api_key="test-key"),
            recognizer=MockSpeechRecognizer("听到用户说"),
            chat_service=MockChatService("AI 回复"),
            tts_service=MockTTSService()
        )
        
        # 模拟一轮对话
        audio = np.array([1] * 2000, dtype=np.int16)
        
        # 1. 语音转文字
        text = app.process_audio(audio)
        assert text == "听到用户说"
        
        # 2. 对话
        response = app.chat(text)
        assert response == "AI 回复"
        
        # 3. 检查退出
        assert not app.should_exit("继续对话")
        assert app.should_exit("再见")
    
    def test_app_error_handling(self):
        """错误处理测试"""
        from main import VoiceChatApp
        from core import Config
        
        app = VoiceChatApp(config=Config(api_key="test"))
        
        # 测试各个未配置时的错误
        with pytest.raises(NotImplementedError):
            app.speech_to_text(np.array([1] * 2000))
        
        with pytest.raises(NotImplementedError):
            app.chat("test")
    
    @pytest.mark.asyncio
    async def test_tts_error_handling(self):
        """TTS 错误处理"""
        from main import VoiceChatApp
        from core import Config
        
        app = VoiceChatApp(config=Config(api_key="test"))
        
        with pytest.raises(NotImplementedError):
            await app.synthesize_speech("text", "output.mp3")
    
    def test_conversation_manager_integration(self):
        """对话管理器集成测试"""
        from main import VoiceChatApp
        from core import Config
        from mocks import MockChatService
        
        app = VoiceChatApp(
            config=Config(api_key="test"),
            chat_service=MockChatService("回复")
        )
        
        # 多轮对话
        app.chat("第一轮")
        app.chat("第二轮")
        app.chat("第三轮")
        
        # 验证历史
        messages = app.conversation.get_messages()
        assert len(messages) == 7  # system + 3*2
        
        # 清理历史
        app.conversation.clear_history()
        assert len(app.conversation.history) == 1
    
    def test_audio_processor_integration(self):
        """音频处理器集成测试"""
        from main import VoiceChatApp
        from core import Config
        from mocks import MockSpeechRecognizer
        
        app = VoiceChatApp(
            config=Config(api_key="test"),
            recognizer=MockSpeechRecognizer("识别结果")
        )
        
        # 标准化测试
        audio_float = np.array([0.5, -0.5], dtype=np.float32)
        normalized = app.audio_processor.normalize_audio(audio_float)
        assert normalized.dtype == np.int16
        
        # 有效性测试
        assert app.audio_processor.is_valid_audio(None) is False
        assert app.audio_processor.is_valid_audio(np.array([1]*100, dtype=np.int16), min_samples=200) is False
        assert app.audio_processor.is_valid_audio(np.array([1]*300, dtype=np.int16), min_samples=200) is True


class TestTextProcessor:
    """文本处理器测试"""
    
    def test_text_processor_integration(self):
        """文本处理器集成"""
        from main import VoiceChatApp
        from core import Config
        
        app = VoiceChatApp(config=Config(api_key="test"))
        
        # 清理测试
        assert app.text_processor.clean_text("  hello  ") == "hello"
        assert app.text_processor.clean_text("") == ""
        
        # 空检查
        assert app.text_processor.is_empty_text(None) is True
        assert app.text_processor.is_empty_text("") is True
        assert app.text_processor.is_empty_text("text") is False
        
        # 截断
        long_text = "a" * 2000
        truncated = app.text_processor.truncate_text(long_text, 100)
        assert len(truncated) == 103  # 100 + ...
