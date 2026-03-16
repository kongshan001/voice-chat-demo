# 🎙️ Voice Chat Demo

语音识别 + GLM 模型对话 Demo (增强版)

## 功能

- 🎤 **实时语音录制** - 带 VAD 语音活动检测
- 📝 **语音转文字** - Faster Whisper (本地离线)
- 🤖 **GLM-4 对话** - 智谱AI API
- 🔊 **流式响应** - 打字时实时输出
- 💬 **连续对话** - 保持上下文

## 安装

```bash
pip install -r requirements.txt
```

## 配置

1. 获取 GLM API Key: https://open.bigmodel.cn/
2. 设置环境变量:
   ```bash
   export ZHIPU_API_KEY="your-api-key-here"
   ```

## 运行

```bash
python main.py
```

## 使用方式

1. 按 `Enter` 开始说话
2. 说完后等待 0.5 秒自动结束
3. AI 会流式输出文字回复
4. 按 `Enter` 播放语音回复
5. 说"再见"退出

## 依赖

| 包 | 用途 |
|---|---|
| faster-whisper | 语音识别 |
| zhipuai | GLM API SDK |
| edge-tts | 微软语音合成 |
| sounddevice | 音频录制 |
| webrtcvad | 语音活动检测 |

## 注意

- 首次运行会自动下载 Whisper 模型 (~140MB)
- 需要麦克风和扬声器权限
- 按 `Ctrl+C` 强制退出
