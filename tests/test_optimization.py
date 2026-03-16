"""测试异步 TTS 错误处理"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTTSAsyncErrorHandling:
    """测试 TTS 异步错误处理"""
    
    def test_synthesize_with_network_error(self):
        """测试网络错误处理"""
        from main import EdgeTTSService, VoiceChatApp
        from core import Config
        
        config = Config(api_key="test-key")
        
        # 模拟网络错误
        mock_service = AsyncMock()
        mock_service.synthesize.side_effect = RuntimeError("网络连接失败")
        
        app = VoiceChatApp(
            config=config,
            tts_service=mock_service
        )
        
        import asyncio
        
        async def test():
            with pytest.raises(RuntimeError, match="网络连接失败"):
                await app.synthesize_speech("测试", "/tmp/test.mp3")
        
        asyncio.run(test())
    
    def test_synthesize_not_configured(self):
        """测试未配置 TTS 时的错误"""
        from main import VoiceChatApp
        from core import Config
        
        config = Config(api_key="test-key")
        app = VoiceChatApp(config=config)
        
        import asyncio
        from core import ServiceNotConfiguredError
        
        async def test():
            with pytest.raises(ServiceNotConfiguredError):
                await app.synthesize_speech("测试", "/tmp/test.mp3")
        
        asyncio.run(test())


class TestConversationEdgeCases:
    """测试对话管理器边界情况"""
    
    def test_trim_history_preserves_system(self):
        """测试历史修剪保留系统消息"""
        from core import ConversationManager
        
        manager = ConversationManager(max_history=1)
        manager.add_user_message("第一条")
        manager.add_assistant_message("回复1")
        manager.add_user_message("第二条")
        manager.add_assistant_message("回复2")
        
        # 验证 system 消息始终存在
        messages = manager.get_messages()
        assert messages[0]["role"] == "system"
    
    def test_exit_keyword_case_sensitivity(self):
        """测试退出关键词大小写不敏感"""
        from core import ConversationManager
        
        manager = ConversationManager()
        assert manager.should_exit("STOP") == True
        assert manager.should_exit("EXIT") == True
        assert manager.should_exit("QUIT") == True


class TestAudioProcessorEdgeCasesExtended:
    """扩展音频处理器边界测试"""
    
    def test_normalize_audio_preserves_data(self):
        """测试归一化保留数据"""
        import numpy as np
        from core import AudioProcessor
        
        processor = AudioProcessor()
        # 小幅音频
        audio = np.array([100, -50, 25, -10], dtype=np.float32)
        result = processor.normalize_audio(audio)
        
        assert result.dtype == np.int16
    
    def test_is_valid_audio_zero_min_samples(self):
        """测试零最小样本数"""
        import numpy as np
        from core import AudioProcessor
        
        processor = AudioProcessor()
        # 空数组但 min_samples=0 应该返回 True
        assert processor.is_valid_audio(np.array([]), min_samples=0) == True


class TestConfigExtended:
    """扩展配置测试"""
    
    def test_validate_all_valid(self):
        """测试完整有效配置"""
        from core import Config
        
        config = Config(
            api_key="test-key-123",
            whisper_model="base",
            tts_voice="zh-CN-XiaoxiaoNeural",
            vad_aggressiveness=2,
            temperature=1.0
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validate_multiple_errors(self):
        """测试多个错误同时存在"""
        from core import Config
        
        config = Config(
            api_key="your-api-key-here",  # 占位符
            whisper_model="invalid-model",  # 无效模型
            tts_voice="invalid-voice",  # 无效语音
            vad_aggressiveness=5,  # 超出范围
            temperature=3.0  # 超出范围
        )
        
        errors = config.validate()
        assert len(errors) >= 3
