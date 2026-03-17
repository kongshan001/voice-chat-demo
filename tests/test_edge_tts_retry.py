"""
测试增强 - Edge TTS 重试机制
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp


class TestEdgeTTSRetryMechanism:
    """Edge TTS 重试机制测试"""
    
    def test_retry_constants_exist(self):
        """测试重试常量存在"""
        from main import EdgeTTSService
        assert hasattr(EdgeTTSService, 'MAX_RETRIES')
        assert hasattr(EdgeTTSService, 'RETRY_DELAY')
        assert EdgeTTSService.MAX_RETRIES == 3
    
    @pytest.mark.asyncio
    async def test_synthesize_success_first_try(self):
        """测试首次成功"""
        from main import EdgeTTSService
        
        service = EdgeTTSService()
        service.communicate = MagicMock()
        
        mock_communicate = MagicMock()
        mock_save = AsyncMock()
        mock_communicate.save = mock_save
        service.communicate = MagicMock(return_value=mock_communicate)
        
        result = await service.synthesize("你好", "/tmp/test.mp3")
        
        assert result == "/tmp/test.mp3"
        mock_save.assert_called_once_with("/tmp/test.mp3")
    
    @pytest.mark.asyncio
    async def test_synthesize_retry_on_connection_error(self):
        """测试连接错误时重试"""
        from main import EdgeTTSService
        
        service = EdgeTTSService()
        
        # Mock communicate to raise connection error then succeed
        mock_comm = AsyncMock()
        mock_comm.save = AsyncMock(side_effect=[
            aiohttp.ClientError("Connection failed"),
            aiohttp.ClientError("Connection failed"),
            None  # Third call succeeds
        ])
        service.communicate = MagicMock(return_value=mock_comm)
        
        result = await service.synthesize("你好", "/tmp/test.mp3")
        
        assert result == "/tmp/test.mp3"
        assert mock_comm.save.call_count == 3
    
    @pytest.mark.asyncio
    async def test_synthesize_all_retries_fail(self):
        """测试所有重试都失败"""
        from main import EdgeTTSService
        
        service = EdgeTTSService()
        
        mock_comm = AsyncMock()
        mock_comm.save = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))
        service.communicate = MagicMock(return_value=mock_comm)
        
        with pytest.raises((aiohttp.ClientError, ConnectionError)):
            await service.synthesize("你好", "/tmp/test.mp3")
