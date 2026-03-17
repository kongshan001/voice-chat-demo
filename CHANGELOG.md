# 更新日志

All notable changes to this project will be documented in this file.

## [1.2.1] - 2026-03-17

### Changed
- 优化 `AudioProcessor.__init__()` 添加完整文档字符串
- 优化 `TextProcessor` 静态方法添加完整文档字符串
- 改进空文本处理的健壮性

## [1.2.0] - 2026-03-17

### Added
- 新增优化测试套件 (`tests/test_optimization.py`) - 8个新测试用例
- 新增 TTS 异步错误处理测试
- 新增对话管理器边界测试
- 新增配置多错误验证测试

### Changed
- 优化 `_setup_logging()` 避免重复调用时的冗余日志配置
- 优化 `GLMChatService.chat()` 添加 API 错误处理
- 更新测试统计: 130 tests (原 122 tests)

### Fixed
- 改进 GLM API 错误提示信息

## [1.1.0] - 2026-03-17

### Added
- 新增服务错误处理测试 (`tests/test_services.py`)
- 新增 ConversationManager 退出关键词类常量
- 新增 VoiceChatApp 属性访问器 (properties)
- 支持 numpy 2.x 版本

### Changed
- 优化 `ConversationManager.should_exit()` 使用类常量
- 优化 `VoiceChatApp` 使用 property 替代 hasattr
- 放宽 numpy 版本限制

### Fixed
- 修复潜在的属性检查问题

## [1.0.0] - 2024-01-XX

### Added
- 初始版本
- 语音识别 (Faster Whisper)
- 对话服务 (GLM-4)
- 语音合成 (Edge TTS)
- 完整测试套件 (114 tests)
