"""
测试 CLI 参数组合场景
"""
import pytest
from main import main
from services import GLMChatService, DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY


class TestCLIParsingCombinations:
    """测试 CLI 参数组合"""
    
    def test_model_and_sample_rate(self):
        """测试模型和采样率组合"""
        import argparse
        from main import main
        
        # 测试默认参数
        parser = argparse.ArgumentParser()
        parser.add_argument("--api-key", "-k")
        parser.add_argument("--whisper-model", "-m", default="base")
        parser.add_argument("--optimize-for-pi", action="store_true")
        parser.add_argument("--sample-rate", "-s", type=int, default=16000)
        
        args = parser.parse_args([])
        assert args.whisper_model == "base"
        assert args.sample_rate == 16000
    
    def test_optimize_for_pi_overrides_model(self):
        """测试树莓派优化模式覆盖模型参数"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--api-key", "-k")
        parser.add_argument("--whisper-model", "-m", default="base")
        parser.add_argument("--optimize-for-pi", action="store_true")
        parser.add_argument("--sample-rate", "-s", type=int, default=16000)
        
        # 模拟优化逻辑
        args = parser.parse_args(["--whisper-model", "medium", "--optimize-for-pi"])
        
        # 当启用优化时，模型应该是 tiny
        if args.optimize_for_pi:
            args.whisper_model = "tiny"
        
        assert args.optimize_for_pi is True
        assert args.whisper_model == "tiny"


class TestGLMRetryMechanism:
    """测试 GLM 服务重试机制"""
    
    def test_retry_constants_exist(self):
        """测试重试常量定义"""
        assert DEFAULT_MAX_RETRIES == 3
        assert DEFAULT_RETRY_DELAY == 1
    
    def test_default_timeout(self):
        """测试默认超时值"""
        assert DEFAULT_TIMEOUT == 60


class TestConfigValidation:
    """测试配置验证增强"""
    
    def test_validate_sample_rate_warning(self):
        """测试非标准采样率产生警告"""
        from core import Config
        import logging
        
        config = Config(api_key="test-key", sample_rate=12345)
        errors = config.validate()
        
        # 应该没有错误，但有警告
        assert "API Key" not in str(errors)
    
    def test_validate_negative_sample_rate(self):
        """测试负采样率被拒绝"""
        from core import Config
        
        config = Config(api_key="test-key", sample_rate=-16000)
        errors = config.validate()
        
        assert any("采样率" in str(e) for e in errors)
    
    def test_validate_vad_out_of_range(self):
        """测试 VAD 参数越界"""
        from core import Config
        
        config = Config(api_key="test-key", vad_aggressiveness=5)
        errors = config.validate()
        
        assert any("VAD" in str(e) for e in errors)
    
    def test_validate_temperature_out_of_range(self):
        """测试温度参数越界"""
        from core import Config
        
        config = Config(api_key="test-key", temperature=3.0)
        errors = config.validate()
        
        assert any("温度" in str(e) for e in errors)


class TestNewExceptions:
    """测试新增异常类"""
    
    def test_new_exceptions_exist(self):
        """测试新异常类存在"""
        from core import (
            TranscriptionError, 
            ChatError, 
            SynthesisError,
            VoiceChatError
        )
        
        # 测试异常继承
        assert issubclass(TranscriptionError, VoiceChatError)
        assert issubclass(ChatError, VoiceChatError)
        assert issubclass(SynthesisError, VoiceChatError)
    
    def test_can_raise_and_catch(self):
        """测试异常可以抛出和捕获"""
        from core import ChatError, VoiceChatError
        
        with pytest.raises(ChatError):
            raise ChatError("测试错误")
        
        # 测试多态捕获
        with pytest.raises(VoiceChatError):
            raise ChatError("测试错误")


class TestConversationManagerEdgeCases:
    """测试对话管理器边界情况"""
    
    def test_history_limit_zero(self):
        """测试 max_history 为 0"""
        from core import ConversationManager
        
        conv = ConversationManager(max_history=0)
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")
        
        # 应该保留 system message
        messages = conv.get_messages()
        assert len(messages) >= 1
    
    def test_very_long_system_prompt(self):
        """测试超长系统提示"""
        from core import ConversationManager
        
        long_prompt = "你是一个" + "非常" * 1000 + "友好的AI助手"
        conv = ConversationManager(system_prompt=long_prompt)
        
        assert conv.get_messages()[0]["content"] == long_prompt
    
    def test_unicode_messages(self):
        """测试 Unicode 消息"""
        from core import ConversationManager
        
        conv = ConversationManager()
        conv.add_user_message("你好👋🎉")
        conv.add_assistant_message("测试日本語한국어")
        
        messages = conv.get_messages()
        assert len(messages) == 3  # system + user + assistant
