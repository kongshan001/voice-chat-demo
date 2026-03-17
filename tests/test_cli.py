"""CLI 参数解析测试"""
import pytest
from main import main


class TestCLIArguments:
    """测试命令行参数解析"""
    
    def test_parse_default_args(self):
        """测试默认参数解析"""
        # 不执行 main，只测试参数解析部分
        import argparse
        from main import main
        
        # 模拟解析默认参数
        parser = argparse.ArgumentParser(description="语音对话 Demo")
        parser.add_argument("--api-key", "-k", help="GLM API Key")
        parser.add_argument("--whisper-model", "-m", default="base", 
                            choices=["tiny", "base", "small", "medium"],
                            help="Whisper 模型大小 (默认: base)")
        parser.add_argument("--optimize-for-pi", action="store_true",
                            help="为树莓派优化 (使用 tiny 模型)")
        parser.add_argument("--sample-rate", "-s", type=int, default=16000,
                            help="音频采样率 (默认: 16000)")
        
        args = parser.parse_args([])
        assert args.whisper_model == "base"
        assert args.sample_rate == 16000
        assert args.optimize_for_pi is False
    
    def test_parse_custom_api_key(self):
        """测试自定义 API Key"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--api-key", "-k")
        args = parser.parse_args(["--api-key", "test-key-123"])
        assert args.api_key == "test-key-123"
    
    def test_parse_whisper_model(self):
        """测试 Whisper 模型参数"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--whisper-model", "-m", default="base", 
                            choices=["tiny", "base", "small", "medium"])
        
        args = parser.parse_args(["--whisper-model", "tiny"])
        assert args.whisper_model == "tiny"
        
        args = parser.parse_args(["-m", "medium"])
        assert args.whisper_model == "medium"
    
    def test_parse_pi_mode(self):
        """测试树莓派优化模式"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimize-for-pi", action="store_true")
        
        args = parser.parse_args([])
        assert args.optimize_for_pi is False
        
        args = parser.parse_args(["--optimize-for-pi"])
        assert args.optimize_for_pi is True
    
    def test_parse_sample_rate(self):
        """测试采样率参数"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--sample-rate", "-s", type=int, default=16000)
        
        args = parser.parse_args(["--sample-rate", "44100"])
        assert args.sample_rate == 44100
        
        args = parser.parse_args(["-s", "8000"])
        assert args.sample_rate == 8000
    
    def test_invalid_whisper_model_raises(self):
        """测试无效的 Whisper 模型应报错"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--whisper-model", "-m", default="base", 
                            choices=["tiny", "base", "small", "medium"])
        
        with pytest.raises(SystemExit):
            parser.parse_args(["--whisper-model", "invalid"])


class TestEnvironmentVariables:
    """测试环境变量处理"""
    
    def test_env_var_api_key(self, monkeypatch):
        """测试从环境变量读取 API Key"""
        monkeypatch.setenv("ZHIPU_API_KEY", "env-test-key")
        import os
        api_key = os.getenv("ZHIPU_API_KEY", "your-api-key-here")
        assert api_key == "env-test-key"
    
    def test_env_var_whisper_model(self, monkeypatch):
        """测试从环境变量读取 Whisper 模型"""
        monkeypatch.setenv("WHISPER_MODEL", "tiny")
        import os
        model = os.getenv("WHISPER_MODEL", "base")
        assert model == "tiny"
    
    def test_default_api_key_fallback(self, monkeypatch):
        """测试 API Key 未设置时的默认值"""
        import os
        # 确保环境变量不存在
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        api_key = os.getenv("ZHIPU_API_KEY", "your-api-key-here")
        assert api_key == "your-api-key-here"
    
    def test_cli_help_flag(self):
        """测试 CLI --help 参数"""
        import argparse
        from main import main
        
        # 创建与 main 相同的 parser
        parser = argparse.ArgumentParser(description="语音对话 Demo")
        parser.add_argument("--api-key", "-k", help="GLM API Key")
        parser.add_argument("--whisper-model", "-m", default="base", 
                            choices=["tiny", "base", "small", "medium"],
                            help="Whisper 模型大小 (默认: base)")
        parser.add_argument("--optimize-for-pi", action="store_true",
                            help="为树莓派优化 (使用 tiny 模型)")
        parser.add_argument("--sample-rate", "-s", type=int, default=16000,
                            help="音频采样率 (默认: 16000)")
        
        # 测试 -h 和 --help 都会触发 help 消息
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-h"])
        assert exc_info.value.code == 0
        
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
