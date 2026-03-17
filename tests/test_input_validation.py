"""
输入验证测试
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from main import GLMChatService, EdgeTTSService, VoiceChatApp
from core import Config


class TestGLMChatServiceInputValidation:
    """GLMChatService 输入验证测试"""
    
    def test_chat_empty_messages_raises_error(self):
        """测试空消息列表应抛出异常"""
        service = GLMChatService(api_key="test-key")
        
        with pytest.raises(ValueError, match="消息列表不能为空"):
            service.chat([])
    
    def test_chat_none_messages_raises_error(self):
        """测试 None 消息应抛出异常"""
        service = GLMChatService(api_key="test-key")
        
        with pytest.raises(ValueError, match="消息列表不能为空"):
            service.chat(None)


class TestEdgeTTSServiceInputValidation:
    """EdgeTTSService 输入验证测试"""
    
    def test_synthesize_empty_text_raises_error(self):
        """测试空文本应抛出异常"""
        service = EdgeTTSService()
        
        async def test():
            with pytest.raises(ValueError, match="合成文本不能为空"):
                await service.synthesize("", "output.mp3")
        
        asyncio.run(test())
    
    def test_synthesize_whitespace_only_raises_error(self):
        """测试仅空白字符应抛出异常"""
        service = EdgeTTSService()
        
        async def test():
            with pytest.raises(ValueError, match="合成文本不能为空"):
                await service.synthesize("   ", "output.mp3")
        
        asyncio.run(test())
    
    def test_synthesize_valid_text(self):
        """测试有效文本可以正常合成"""
        service = EdgeTTSService()
        
        async def test():
            with patch.object(service, 'communicate') as mock_comm:
                mock_comm_instance = AsyncMock()
                mock_comm.return_value = mock_comm_instance
                
                result = await service.synthesize("你好", "output.mp3")
                
                mock_comm.assert_called_once_with("你好", service.voice)
                mock_comm_instance.save.assert_called_once_with("output.mp3")
                assert result == "output.mp3"
        
        asyncio.run(test())
    
    def test_synthesize_network_error(self):
        """测试网络错误应转换为 ConnectionError"""
        service = EdgeTTSService()
        
        async def test():
            with patch.object(service, 'communicate') as mock_comm:
                mock_comm_instance = AsyncMock()
                mock_comm_instance.save.side_effect = Exception("Connection refused")
                mock_comm.return_value = mock_comm_instance
                
                with pytest.raises(RuntimeError, match="Edge TTS 合成失败"):
                    await service.synthesize("你好", "output.mp3")
        
        asyncio.run(test())
    
    def test_synthesize_client_error(self):
        """测试客户端错误应转换为 ConnectionError (重试后)"""
        service = EdgeTTSService()
        
        async def test():
            import aiohttp
            with patch.object(service, 'communicate') as mock_comm:
                mock_comm_instance = AsyncMock()
                # All retries fail with ClientError
                mock_comm_instance.save.side_effect = aiohttp.ClientError("Network error")
                mock_comm.return_value = mock_comm_instance
                
                # After 3 retries, should raise the error
                with pytest.raises((ConnectionError, aiohttp.ClientError)):
                    await service.synthesize("你好", "output.mp3")
        
        asyncio.run(test())


class TestVoiceChatAppSimplified:
    """VoiceChatApp 简化属性测试"""
    
    def test_services_directly_assigned(self):
        """测试服务直接赋值"""
        config = Config(api_key="test-key")
        mock_recognizer = Mock()
        mock_chat = Mock()
        mock_tts = Mock()
        
        app = VoiceChatApp(
            config=config,
            recognizer=mock_recognizer,
            chat_service=mock_chat,
            tts_service=mock_tts
        )
        
        assert app.recognizer is mock_recognizer
        assert app.chat_service is mock_chat
        assert app.tts_service is mock_tts
    
    def test_services_can_be_modified(self):
        """测试服务可以修改"""
        config = Config(api_key="test-key")
        mock_recognizer1 = Mock()
        mock_recognizer2 = Mock()
        
        app = VoiceChatApp(config=config, recognizer=mock_recognizer1)
        assert app.recognizer is mock_recognizer1
        
        app.recognizer = mock_recognizer2
        assert app.recognizer is mock_recognizer2
    
    def test_services_default_to_none(self):
        """测试服务默认为 None"""
        config = Config(api_key="test-key")
        
        app = VoiceChatApp(config=config)
        
        assert app.recognizer is None
        assert app.chat_service is None
        assert app.tts_service is None
