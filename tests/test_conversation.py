"""
单元测试 - ConversationManager
"""
import pytest
from core import ConversationManager


class TestConversationManager:
    """对话管理器测试"""
    
    def test_init_with_default_system_prompt(self):
        """测试默认 system prompt"""
        manager = ConversationManager()
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "system"
    
    def test_init_with_custom_system_prompt(self):
        """测试自定义 system prompt"""
        custom_prompt = "你是一个有用的助手"
        manager = ConversationManager(custom_prompt)
        assert manager.history[0]["content"] == custom_prompt
    
    def test_add_user_message(self):
        """测试添加用户消息"""
        manager = ConversationManager()
        manager.add_user_message("你好")
        
        assert len(manager.history) == 2
        assert manager.history[-1]["role"] == "user"
        assert manager.history[-1]["content"] == "你好"
    
    def test_add_assistant_message(self):
        """测试添加助手消息"""
        manager = ConversationManager()
        manager.add_assistant_message("你好，很高兴见到你")
        
        assert len(manager.history) == 2
        assert manager.history[-1]["role"] == "assistant"
        assert manager.history[-1]["content"] == "你好，很高兴见到你"
    
    def test_get_messages(self):
        """测试获取消息列表"""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        
        messages = manager.get_messages()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_clear_history(self):
        """测试清空历史"""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        manager.clear_history()
        
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "system"
    
    def test_should_exit_with再见(self):
        """测试退出关键词 - 再见"""
        manager = ConversationManager()
        assert manager.should_exit("再见") is True
        assert manager.should_exit("再见，我会想念你") is True
    
    def test_should_exit_with退出(self):
        """测试退出关键词 - 退出"""
        manager = ConversationManager()
        assert manager.should_exit("退出") is True
        # 中文关键词任意位置都触发
        assert manager.should_exit("我想退出") is True
    
    def test_should_exit_with拜拜(self):
        """测试退出关键词 - 拜拜"""
        manager = ConversationManager()
        assert manager.should_exit("拜拜") is True
    
    def test_should_exit_with_english_keywords(self):
        """测试退出关键词 - 英文"""
        manager = ConversationManager()
        # 英文需要独立单词
        assert manager.should_exit("quit") is True
        assert manager.should_exit("exit now") is True
        assert manager.should_exit("stop") is True
    
    def test_should_not_exit(self):
        """测试不应退出"""
        manager = ConversationManager()
        assert manager.should_exit("你好") is False
        assert manager.should_exit("今天天气很好") is False
        # 英文需要完整单词，包含不算
        assert manager.should_exit("exiting the app") is False
    
    def test_conversation_history_grows(self):
        """测试对话历史增长"""
        manager = ConversationManager()
        
        for i in range(5):
            manager.add_user_message(f"用户消息 {i}")
            manager.add_assistant_message(f"助手回复 {i}")
        
        # system + 5轮对话 = 11条
        assert len(manager.history) == 11
    
    def test_history_limit(self):
        """测试历史记录限制"""
        manager = ConversationManager(max_history=3)
        
        # 添加 5 轮对话 (10 条消息)
        for i in range(5):
            manager.add_user_message(f"用户 {i}")
            manager.add_assistant_message(f"助手 {i}")
        
        # 应该只保留 system + 最近 3 轮 (7 条)
        assert len(manager.history) == 7
        
        # 验证包含最新的对话
        assert "用户 4" in [m["content"] for m in manager.history if m["role"] == "user"]
        # 最旧的用户消息应该被修剪掉
        assert "用户 0" not in [m["content"] for m in manager.history if m["role"] == "user"]
    
    def test_clear_history_with_max_history(self):
        """测试清空历史后保留限制"""
        manager = ConversationManager(max_history=5)
        
        manager.add_user_message("test")
        manager.clear_history()
        
        # 清空后应该保留 system 和限制设置
        assert len(manager.history) == 1
        assert manager.max_history == 5
