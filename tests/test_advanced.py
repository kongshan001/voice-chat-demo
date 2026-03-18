"""
测试边界情况和代码覆盖率增强
"""
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from core import (
    ConversationManager, 
    AudioProcessor, 
    TextProcessor, 
    Config,
    ServiceNotConfiguredError,
    TranscriptionError,
    ChatError,
    SynthesisError
)
from services import GLMChatService, EdgeTTSService, DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES


class TestConversationManagerAdvanced:
    """高级对话管理测试"""
    
    def test_history_limit_single_turn(self):
        """测试单轮对话历史"""
        manager = ConversationManager(max_history=1)
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        # 应该是: system + user + assistant = 3条
        assert len(manager.history) == 3
    
    def test_history_limit_multiple_turns(self):
        """测试多轮对话历史限制"""
        manager = ConversationManager(max_history=2)
        for i in range(10):
            manager.add_user_message(f"User {i}")
            manager.add_assistant_message(f"Assistant {i}")
        
        # max_history * 2 + system = 5
        assert len(manager.history) == 5
    
    def test_clear_history_preserves_system(self):
        """测试清空历史保留system消息"""
        manager = ConversationManager()
        manager.add_user_message("Test")
        manager.add_assistant_message("Response")
        manager.clear_history()
        
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "system"
    
    def test_unicode_in_messages(self):
        """测试Unicode字符支持"""
        manager = ConversationManager()
        manager.add_user_message("你好👋")
        manager.add_assistant_message("Hello🎉")
        
        messages = manager.get_messages()
        assert "你好👋" in messages[1]["content"]
        assert "Hello🎉" in messages[2]["content"]


class TestAudioProcessorAdvanced:
    """高级音频处理测试"""
    
    def test_normalize_audio_int8(self):
        """测试int8类型音频处理 - 当前不支持int8"""
        audio = np.array([100, -100, 50, -50], dtype=np.int8)
        processor = AudioProcessor()
        
        # 当前实现不支持int8，预期抛出异常
        with pytest.raises(ValueError) as exc_info:
            processor.normalize_audio(audio)
        assert "不支持" in str(exc_info.value)
    
    def test_normalize_audio_preserves_shape(self):
        """测试音频形状保持"""
        audio = np.random.randint(-1000, 1000, size=1000, dtype=np.int16)
        processor = AudioProcessor()
        result = processor.normalize_audio(audio)
        assert len(result) == len(audio)
    
    def test_is_valid_audio_exact_min_samples(self):
        """测试恰好最小采样数"""
        audio = np.zeros(1600, dtype=np.int16)
        processor = AudioProcessor()
        assert processor.is_valid_audio(audio, min_samples=1600) is True
    
    def test_bytes_to_audio_even_length(self):
        """测试偶数字节转换"""
        data = b'\x00\x01\x02\x03'
        processor = AudioProcessor()
        result = processor.bytes_to_audio(data)
        assert len(result) == 2


class TestTextProcessorAdvanced:
    """高级文本处理测试"""
    
    def test_clean_text_with_special_chars(self):
        """测试特殊字符清理"""
        text = "  \t\nHello\r\nWorld  \t"
        result = TextProcessor.clean_text(text)
        # 实际行为：只去除首尾空白，保留中间的\r\n
        assert result == "Hello\r\nWorld"
    
    def test_truncate_text_exact_max_length(self):
        """测试恰好最大长度"""
        text = "a" * 1000
        result = TextProcessor.truncate_text(text, 1000)
        assert result == text
    
    def test_is_chinese_mixed(self):
        """测试中英文混合"""
        assert TextProcessor.is_chinese("Hello你好") is True
        assert TextProcessor.is_chinese("ABC123") is False
        assert TextProcessor.is_chinese("") is False


class TestConfigAdvanced:
    """高级配置测试"""
    
    def test_validate_with_warnings(self):
        """测试带警告的配置验证"""
        config = Config(
            api_key="valid-key-12345",
            sample_rate=22050  # 非标准但可用
        )
        errors = config.validate()
        # 不应该有错误，但可能有警告
        assert len(errors) == 0
    
    def test_is_configured_with_env_var_style(self):
        """测试环境变量样式配置"""
        config = Config(api_key="zk-test-xxx")
        assert config.is_configured is True
    
    def test_whisper_device_validation(self):
        """测试 Whisper 设备配置"""
        config = Config(whisper_device="cuda")
        # 设备验证应该通过（只有模型和语音有严格限制）
        errors = config.validate()
        assert "whisper_device" not in str(errors)


class TestVoiceChatAppErrorsAdvanced:
    """高级错误处理测试"""
    
    def test_speech_to_text_with_exception(self):
        """测试语音识别异常处理"""
        from main import VoiceChatApp
        config = Config(api_key="test-key")
        mock_recognizer = Mock()
        mock_recognizer.transcribe.side_effect = RuntimeError("Model error")
        
        app = VoiceChatApp(
            config=config,
            recognizer=mock_recognizer
        )
        
        audio = np.zeros(1600, dtype=np.int16)
        
        with pytest.raises(TranscriptionError) as exc_info:
            app.speech_to_text(audio)
        
        assert "语音识别失败" in str(exc_info.value)
    
    def test_chat_with_empty_service_raises(self):
        """测试空聊天服务抛出正确异常"""
        from main import VoiceChatApp
        config = Config(api_key="test-key")
        app = VoiceChatApp(config=config)
        
        with pytest.raises(ServiceNotConfiguredError):
            app.chat("Hello")
    
    def test_synthesize_speech_with_empty_service(self):
        """测试空TTS服务抛出正确异常"""
        from main import VoiceChatApp
        config = Config(api_key="test-key")
        
        import asyncio
        app = VoiceChatApp(config=config)
        
        with pytest.raises(ServiceNotConfiguredError):
            asyncio.run(app.synthesize_speech("test", "/tmp/test.mp3"))


class TestEdgeTTSServiceAdvanced:
    """Edge TTS 高级测试"""
    
    @pytest.mark.asyncio
    async def test_synthesize_with_very_long_text(self):
        """测试超长文本合成"""
        service = EdgeTTSService()
        
        # 模拟一个非常长的文本
        long_text = "测试文本. " * 1000
        
        with patch.object(service, '_do_synthesize', new_callable=AsyncMock) as mock:
            mock.return_value = "/tmp/test.mp3"
            result = await service.synthesize(long_text, "/tmp/test.mp3")
            assert result == "/tmp/test.mp3"
    
    @pytest.mark.asyncio
    async def test_synthesize_trims_whitespace(self):
        """测试合成时是否自动去除空白 - 当前不自动去除"""
        service = EdgeTTSService()
        
        with patch.object(service, 'communicate') as mock_comm:
            mock_comm.return_value = AsyncMock()
            mock_comm.return_value.save = AsyncMock()
            
            # 当前实现会保持原样
            await service.synthesize("  Hello World  ", "/tmp/test.mp3")
            
            # 验证communicate被调用时文本保持原样
            call_args = mock_comm.call_args
            assert call_args[0][0] == "  Hello World  "


class TestGLMServiceAdvanced:
    """GLM 服务高级测试"""
    
    def test_chat_message_validation(self):
        """测试消息验证"""
        service = GLMChatService(api_key="test-key")
        
        # 测试空消息列表
        with pytest.raises(ValueError) as exc_info:
            service.chat([])
        assert "不能为空" in str(exc_info.value)
    
    def test_chat_timeout_parameter(self):
        """测试超时参数"""
        service = GLMChatService(api_key="test-key")
        
        # 验证默认超时
        assert DEFAULT_TIMEOUT == 60
        
        # 验证重试次数
        assert DEFAULT_MAX_RETRIES == 3
    
    def test_chat_with_custom_timeout(self):
        """测试自定义超时 - 验证参数传递"""
        service = GLMChatService(api_key="test-key")
        
        with patch.object(service, '_do_chat', return_value="OK") as mock:
            service.chat([{"role": "user", "content": "hi"}], timeout=120)
            
            # 验证超时参数被传递（实际通过位置参数传递）
            mock.assert_called_once()
            # 检查timeout参数是否在调用中传递
            call_args = mock.call_args
            # timeout 是位置参数，在第二个位置
            assert call_args[0][2] == 120  # (messages, stream_callback, timeout)


class TestConfigEdgeCases:
    """配置边界情况测试"""
    
    def test_api_key_with_spaces(self):
        """测试带空格的有效API Key"""
        config = Config(api_key="  zk-test-key  ")
        # 验证api_key属性本身保留空格（验证在validate之前不会自动strip）
        assert config.api_key == "  zk-test-key  "
    
    def test_negative_max_history(self):
        """测试负数max_history"""
        config = Config(max_history=-1)
        manager = ConversationManager(max_history=-1)
        
        # 负数max_history应该被处理
        manager.add_user_message("test")
        # 不应该崩溃
        
    def test_zero_sample_rate(self):
        """测试零采样率"""
        with pytest.raises(ValueError):
            AudioProcessor(sample_rate=0)
    
    def test_negative_channels(self):
        """测试负数通道"""
        with pytest.raises(ValueError):
            AudioProcessor(channels=-1)
