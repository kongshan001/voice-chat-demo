# 贡献指南

欢迎贡献！请遵循以下步骤：

## 开发环境设置

```bash
# 克隆项目
git clone https://github.com/kongshan001/voice-chat-demo.git
cd voice-chat-demo

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 代码规范

- 使用 4 空格缩进
- 遵循 PEP 8
- 新增函数需添加 docstring
- 公共 API 需有类型注解

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_audio.py -v

# 生成覆盖率报告
pytest tests/ --cov=. --cov-report=html
```

## 提交规范

提交信息格式：
```
<type>(<scope>): <description>

[可选的正文]

[可选的脚注]
```

类型 (type):
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `test`: 测试相关
- `refactor`: 代码重构
- `chore`: 构建/工具变动

示例：
```
feat(conversation): 添加退出关键词检测

- 支持中英文退出关键词
- 优化正则表达式匹配性能
```

## PR 流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 编写代码并添加测试
4. 确保所有测试通过
5. 提交并推送
6. 创建 Pull Request

## 问题反馈

请通过 GitHub Issues 报告 bug 或提出功能请求。
