"""
单元测试 - TextProcessor
"""
import pytest
from core import TextProcessor


class TestTextProcessor:
    """文本处理器测试"""
    
    def test_clean_text_normal(self):
        """测试正常清理"""
        processor = TextProcessor()
        result = processor.clean_text("  你好世界  ")
        assert result == "你好世界"
    
    def test_clean_text_empty(self):
        """测试空字符串"""
        processor = TextProcessor()
        result = processor.clean_text("")
        assert result == ""
    
    def test_clean_text_none(self):
        """测试 None"""
        processor = TextProcessor()
        result = processor.clean_text(None)
        assert result == ""
    
    def test_clean_text_only_whitespace(self):
        """测试仅空白字符"""
        processor = TextProcessor()
        result = processor.clean_text("   \t\n  ")
        assert result == ""
    
    def test_is_empty_text_none(self):
        """测试空文本 - None"""
        processor = TextProcessor()
        assert processor.is_empty_text(None) is True
    
    def test_is_empty_text_empty_string(self):
        """测试空文本 - 空字符串"""
        processor = TextProcessor()
        assert processor.is_empty_text("") is True
    
    def test_is_empty_text_whitespace(self):
        """测试空文本 - 空白"""
        processor = TextProcessor()
        assert processor.is_empty_text("   ") is True
    
    def test_is_empty_text_valid(self):
        """测试有效文本"""
        processor = TextProcessor()
        assert processor.is_empty_text("你好") is False
        assert processor.is_empty_text("  你好  ") is False
    
    def test_truncate_text_normal(self):
        """测试正常截断"""
        processor = TextProcessor()
        text = "Hello World"
        result = processor.truncate_text(text, 20)
        assert result == text
    
    def test_truncate_text_long(self):
        """测试长文本截断"""
        processor = TextProcessor()
        text = "a" * 2000
        result = processor.truncate_text(text, 1000)
        
        assert len(result) == 1003  # 1000 + "..."
        assert result.endswith("...")
    
    def test_truncate_text_exact_length(self):
        """测试刚好等于长度"""
        processor = TextProcessor()
        text = "abc"
        result = processor.truncate_text(text, 3)
        assert result == "abc"
    
    def test_truncate_text_zero_length(self):
        """测试零长度截断"""
        processor = TextProcessor()
        text = "Hello"
        result = processor.truncate_text(text, 0)
        assert result == "..."
    
    def test_is_chinese_with_chinese(self):
        """测试包含中文"""
        processor = TextProcessor()
        assert processor.is_chinese("你好") is True
        assert processor.is_chinese("Hello 你好") is True
        assert processor.is_chinese("中文123") is True
    
    def test_is_chinese_without_chinese(self):
        """测试不包含中文"""
        processor = TextProcessor()
        assert processor.is_chinese("Hello") is False
        assert processor.is_chinese("12345") is False
        assert processor.is_chinese("!@#$%") is False
    
    def test_is_chinese_empty(self):
        """测试空文本"""
        processor = TextProcessor()
        assert processor.is_chinese("") is False
        assert processor.is_chinese(None) is False
