# 🎙️ Voice Chat Demo

语音识别 + GLM 模型对话 Demo

## 功能

- 🎤 **实时语音录制** - 带 VAD 语音活动检测
- 📝 **语音转文字** - Faster Whisper (本地离线)
- 🤖 **GLM-4 对话** - 智谱AI API
- 🔊 **流式响应** - 打字时实时输出
- 💬 **连续对话** - 保持上下文
- 🖥️ **硬件部署** - 支持树莓派等嵌入式设备

## 硬件支持

### 最低配置 (MVP)
| 组件 | 推荐型号 | 价格 |
|------|----------|------|
| 主控 | 树莓派 Zero 2 W | ~150元 |
| 麦克风 | USB 免驱麦克风 | ~30元 |
| 音箱 | 3.5mm 有源音箱 | ~30元 |
| 存储 | 16GB TF卡 | ~20元 |
| 电源 | 5V 2A | ~20元 |
| **合计** | | **~250元** |

### 进阶配置
- 树莓派 4B (4GB) - 性能更强
- USB 声卡 + 3.5mm 麦克风 - 音质更好

## 安装

### 1. 克隆项目
```bash
git clone https://github.com/kongshan001/voice-chat-demo.git
cd voice-chat-demo
```

### 2. 创建虚拟环境 (推荐)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 安装依赖
```bash
# 基础依赖
pip install -r requirements.txt

# 开发依赖 (测试)
pip install -r requirements-dev.txt
```

### 4. 配置

#### 获取 API Key
1. 访问 https://open.bigmodel.cn/
2. 注册/登录
3. 创建 API Key

#### 配置环境变量
```bash
# 临时生效
export ZHIPU_API_KEY="your-api-key-here"

# 永久生效 (添加到 ~/.bashrc)
echo 'export ZHIPU_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## 运行

### 桌面/服务器模式
```bash
python main.py
```

### 树莓派部署
```bash
# 使用优化配置 (低内存占用)
python main.py --optimize-for-pi
```

## 使用方式

1. 启动程序
2. 按 `Enter` 开始说话
3. 说完后等待 0.5 秒自动结束
4. AI 会流式输出文字回复
5. 按 `Enter` 播放语音回复
6. 说"再见"退出

## 项目结构

```
voice-chat-demo/
├── core.py              # 核心业务逻辑
├── services.py           # 服务接口定义
├── main.py               # 主程序
├── requirements.txt     # 生产依赖
├── requirements-dev.txt # 开发依赖
├── pyproject.toml       # 项目配置
├── tests/               # 测试目录
└── README.md            # 说明文档
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 生成覆盖率报告
pytest tests/ --cov=. --cov-report=html
```

## 部署到树莓派

### 1. 烧录系统
```bash
# 下载 Raspberry Pi OS Lite
# 使用 Raspberry Pi Imager 烧录
```

### 2. 配置 SSH
```bash
# 在 boot 分区创建空文件 ssh
touch /boot/ssh
```

### 3. 复制项目
```bash
scp -r voice-chat-demo pi@raspberrypi.local:~/
```

### 4. 安装
```bash
ssh pi@raspberrypi.local
cd voice-chat-demo
pip install -r requirements.txt
```

### 5. 设置开机启动 (可选)
```bash
sudo cp voice-chat-demo.service /etc/systemd/system/
sudo systemctl enable voice-chat-demo
sudo systemctl start voice-chat-demo
```

## 配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| ZHIPU_API_KEY | GLM API Key | (必填) |
| WHISPER_MODEL | Whisper 模型大小 | base |
| SAMPLE_RATE | 采样率 | 16000 |

### Whisper 模型选择

| 模型 | 大小 | 内存需求 | 速度 |
|------|------|----------|------|
| tiny | 75MB | ~1GB | 最快 |
| base | 140MB | ~1GB | 快 |
| small | 500MB | ~2GB | 中 |
| medium | 1.5GB | ~4GB | 慢 |

树莓派 Zero/3 推荐 `tiny`，4B 可选 `base`。

## 常见问题

### Q: 录音没有声音
A: 检查麦克风权限 `arecord -l`，确认麦克风已识别

### Q: 识别速度慢
A: 尝试使用更小的模型 `tiny`，或降低采样率

### Q: 内存不足
A: 使用 `--optimize-for-pi` 参数，或改用 tiny 模型

## 许可证

MIT License
