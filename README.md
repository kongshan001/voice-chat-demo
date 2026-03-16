# 🎙️ Voice Chat Demo

语音识别 + GLM 模型对话 Demo

## 功能

- 🎤 实时语音录制
- 📝 语音转文字 (Whisper)
- 🤖 GLM-4 对话 (智谱AI)
- 🔊 语音合成 (Edge TTS)

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

## 依赖

- faster-whisper: 语音识别
- zhipuai: GLM API SDK
- edge-tts: 微软语音合成
- sounddevice: 音频录制

## 注意

- 首次运行会自动下载 Whisper 模型 (~140MB for base)
- 按 Ctrl+C 退出
