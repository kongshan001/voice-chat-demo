"""
测试 main.py 中的服务实现
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np
from core import Config

from main import (
    WhisperRecognizer, 
    GLMChatService, 
    EdgeTTSService,
    VoiceChatApp,
    configure_services
)
from mocks import MockSpeechRecognizer, MockChatService, MockTTSService


class TestWhisperRecognizer:
    """Whisper 识别器测试"""
    
    def test_init(self):
        """测试初始化"""
        recognizer = WhisperRecognizer("base", "cpu")
        assert recognizer.model_name == "base"
        assert recognizer.device == "cpu"
        assert recognizer.model is None
    
    def test_init_default(self):
        """测试默认参数"""
        recognizer = WhisperRecognizer()
        assert recognizer.model_name == "base"
        assert recognizer.device == "cpu"


class TestGLMChatService:
    """GLM 服务测试"""
    
    def test_init(self):
        """测试初始化"""
        service = GLMChatService("test-key")
        assert service is not None


class TestEdgeTTSService:
    """Edge TTS 服务测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        service = EdgeTTSService()
        assert service.voice == "zh-CN-XiaoxiaoNeural"
    
    def test_init_custom_voice(self):
        """测试自定义语音"""
        service = EdgeTTSService(voice="zh-CN-YunxiNeural")
        assert service.voice == "zh-CN-YunxiNeural"


class TestConfigureServices:
    """服务配置测试"""
    
    def test_configure_services(self):
        """测试配置服务"""
        mock_recognizer = MockSpeechRecognizer()
        mock_chat = MockChatService()
        mock_tts = MockTTSService()
        
        configure_services(
            recognizer=mock_recognizer,
            chat_service=mock_chat,
            tts_service=mock_tts
        )
        
        # 验证配置成功（无异常）


class TestVoiceChatAppWithRealServices:
    """使用部分真实服务的测试"""
    
    def test_app_init_with_all_services(self):
        """测试完整服务初始化"""
        mock_rec = MockSpeechRecognizer()
        mock_chat = MockChatService()
        mock_tts = MockTTSService()
        
        app = VoiceChatApp(
            config=Config(api_key="test"),
            recognizer=mock_rec,
            chat_service=mock_chat,
            tts_service=mock_tts
        )
        
        assert app.config.api_key == "test"
        assert hasattr(app, 'recognizer')
        assert hasattr(app, 'chat_service')
        assert hasattr(app, 'tts_service')
    
    @pytest.mark.asyncio
    async def test_synthesize_speech(self):
        """测试语音合成"""
        mock_tts = MockTTSService()
        app = VoiceChatApp(
            config=Config(api_key="test"),
            tts_service=mock_tts
        )
        
        result = await app.synthesize_speech("测试", "/tmp/test.mp3")
        
        assert result == "/tmp/test.mp3"
        assert mock_tts.synthesize_count == 1
    
    def test_speech_to_text_direct(self):
        """测试直接调用语音识别"""
        mock_rec = MockSpeechRecognizer("识别结果")
        app = VoiceChatApp(
            config=Config(api_key="test"),
            recognizer=mock_rec
        )
        
        audio = np.array([1] * 2000, dtype=np.int16)
        result = app.speech_to_text(audio)
        
        assert result == "识别结果"
        assert mock_rec.transcribe_count == 1
    
    def test_process_audio_flow(self):
        """测试音频处理流程"""
        mock_rec = MockSpeechRecognizer("处理结果")
        
        app = VoiceChatApp(
            config=Config(api_key="test"),
            recognizer=mock_rec
        )
        
        # 有效音频
        audio = np.array([1] * 2000, dtype=np.int16)
        result = app.process_audio(audio)
        assert result == "处理结果"
        
        # 无效音频 - None
        result = app.process_audio(None)
        assert result is None
        
        # 无效音频 - 太短
        audio_short = np.array([1, 2, 3], dtype=np.int16)
        result = app.process_audio(audio_short)
        assert result is None
    
    def test_chat_direct(self):
        """测试直接调用对话"""
        mock_chat = MockChatService("回复内容")
        
        app = VoiceChatApp(
            config=Config(api_key="test"),
            chat_service=mock_chat
        )
        
        result = app.chat("用户消息")
        
        assert result == "回复内容"
        assert mock_chat.chat_count == 1
    
    def test_conversation_state(self):
        """测试对话状态"""
        mock_rec = MockSpeechRecognizer("用户说")
        mock_chat = MockChatService("助手说")
        
        app = VoiceChatApp(
            config=Config(api_key="test"),
            recognizer=mock_rec,
            chat_service=mock_chat
        )
        
        # 第一轮
        app.chat("第一轮")
        assert len(app.conversation.history) == 3  # system + user + assistant
        
        # 第二轮
        app.chat("第二轮")
        assert len(app.conversation.history) == 5
        
        # 验证历史记录
        messages = app.conversation.get_messages()
        assert messages[1]["content"] == "第一轮"
        assert messages[2]["content"] == "助手说"
        assert messages[3]["content"] == "第二轮"


class TestRecordAndPlayFunctions:
    """录音播放函数存在性测试"""
    
    def test_functions_exist(self):
        """测试函数存在"""
        from main import record_with_vad, simple_record, play_audio
        
        assert callable(record_with_vad)
        assert callable(simple_record)
        assert callable(play_audio)
