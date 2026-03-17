"""
服务层接口测试
"""
import pytest
from services import ISpeechRecognizer, IChatService, ITTSService, IAudioRecorder


class TestServiceInterfaces:
    """测试服务接口是否正确定义"""
    
    def test_speech_recognizer_is_abstract(self):
        """测试 ISpeechRecognizer 是抽象类"""
        with pytest.raises(TypeError):
            ISpeechRecognizer()
    
    def test_chat_service_is_abstract(self):
        """测试 IChatService 是抽象类"""
        with pytest.raises(TypeError):
            IChatService()
    
    def test_tts_service_is_abstract(self):
        """测试 ITTSService 是抽象类"""
        with pytest.raises(TypeError):
            ITTSService()
    
    def test_audio_recorder_is_abstract(self):
        """测试 IAudioRecorder 是抽象类"""
        with pytest.raises(TypeError):
            IAudioRecorder()


class TestServiceInterfaceMethods:
    """测试服务接口方法签名"""
    
    def test_recognizer_has_required_methods(self):
        """测试语音识别器接口有所需方法"""
        class MockRecognizer(ISpeechRecognizer):
            def transcribe(self, audio):
                return ""
            def load_model(self):
                pass
        
        r = MockRecognizer()
        assert hasattr(r, 'transcribe')
        assert hasattr(r, 'load_model')
    
    def test_chat_service_has_required_methods(self):
        """测试对话服务接口有所需方法"""
        class MockChat(IChatService):
            def chat(self, messages, stream_callback=None, timeout=60):
                return ""
        
        c = MockChat()
        assert hasattr(c, 'chat')
    
    def test_tts_service_has_required_methods(self):
        """测试 TTS 服务接口有所需方法"""
        import asyncio
        class MockTTS(ITTSService):
            async def synthesize(self, text, output_path):
                return output_path
        
        t = MockTTS()
        assert hasattr(t, 'synthesize')
        assert asyncio.iscoroutinefunction(t.synthesize)
    
    def test_audio_recorder_has_required_methods(self):
        """测试录音接口有所需方法"""
        class MockRecorder(IAudioRecorder):
            def record(self, duration):
                return None
        
        r = MockRecorder()
        assert hasattr(r, 'record')
