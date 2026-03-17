# 更新日志

All notable changes to this project will be documented in this file.

## [1.10.0] - 2026-03-18

### Changed
- 移除 main.py 中未使用的 imports (requests, aiohttp)
- 优化代码质量，移除冗余导入

### Added
- 新增 `tests/test_services_full.py` - 21个服务层测试用例
- 提升服务层测试覆盖率

### Fixed
- 修复 test_main_errors.py 中的测试以适应优化后的导入结构

### Notes
- 测试统计: 239 tests - 所有测试通过 ✓
- 覆盖率: 84.91%

## [1.9.0] - 2026-03-18

### Changed
- 优化 services.py - 整合接口和实现，添加完整的 __all__ 导出
- 改进服务层文档和类型注解

### Added
- 新增 `tests/test_new_features.py` - 13个新测试用例
- 新增 GLM 服务边缘测试
- 新增 Edge TTS 服务边缘测试
- 新增配置边缘测试

### Notes
- 测试统计: 218 tests (205 + 13) - 所有测试通过 ✓

### Changed
- 更新依赖包版本: faster-whisper>=1.2.0, zhipuai>=2.1.5, edge-tts>=7.0.0, aiohttp>=3.11.0

### Notes
- 测试统计: 205 tests - 所有测试通过 ✓

## [1.7.1] - 2026-03-18

### Fixed
- 修复 main.py 中 TranscriptionError 未正确导入的问题

### Notes
- 测试统计: 195 tests - 所有测试通过 ✓

## [1.7.0] - 2026-03-17

### Added
- 新增 CLI 未知参数测试 - 增强参数验证覆盖

### Changed
- 优化 `import time` 放置到模块级别 - 避免重复导入
- 改进代码结构 - 移除冗余的函数内导入

### Fixed
- 修复 GLMChatService 重试逻辑中重复导入 time 模块
- 修复 record_with_vad 函数中冗余导入

### Notes
- 测试统计: 195 tests (原 194 tests)
- 所有测试通过 ✓

## [1.6.0] - 2026-03-17

### Added
- 新增 `is_chinese()` 方法到 TextProcessor - 检测中文字符
- 新增 CLI --help 参数测试
- 新增 record_with_vad 函数签名测试

### Changed
- 修复 GLMChatService._do_chat() 错误处理 - 使用 aiohttp 而非 requests 异常
- 改进 API 超时和连接错误检测
- 测试统计: 187 tests (原 179 tests)

### Fixed
- 修复 GLMChatService 异常处理不一致问题

## [1.5.0] - 2026-03-17

### Added
- 新增 `tests/test_main_functions.py` - main.py 函数测试套件
- 新增 WhisperRecognizer 初始化测试
- 新增 EdgeTTSService 初始化测试
- 新增 GLMChatService 初始化测试
- 新增 play_audio 错误处理测试

### Changed
- 优化 AudioProcessor.normalize_audio 文档字符串
- 测试统计: 170 tests -> 179 tests

## [1.4.0] - 2026-03-17

### Added
- 新增 `tests/test_main_errors.py` - 错误处理测试套件
- 新增 play_audio 文件不存在检查
- 新增 simple_record 和 record_with_vad 运行时错误处理

### Changed
- 优化模块导入结构 - `requests` 和 `aiohttp` 改为模块级导入
- 改进 GLMChatService.chat() 文档字符串
- 改进 EdgeTTSService.synthesize() 文档字符串
- 改进 play_audio() 添加完整错误处理和文档
- 改进 simple_record() 添加错误处理和文档
- 改进 record_with_vad() 添加错误处理和文档
- 测试统计: 170 tests (原 166 tests)

## [1.3.0] - 2026-03-17

### Added
- 新增 `.env.example` 配置文件，方便新用户快速配置
- 新增 CLI 参数解析测试覆盖 (`tests/test_cli.py`)
- 新增 KeyboardInterrupt 优雅退出处理
- 新增 python-dotenv 依赖支持

### Changed
- 改进 README 文档中的环境变量配置说明
- 优化 `main()` 函数的错误处理结构

### Fixed
- 修复主循环无法优雅退出的问题 (Ctrl+C 支持)

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
