#!/bin/bash
# 启动脚本 - 支持不同平台优化

# 检测平台
PLATFORM=$(uname -m)

echo "=========================================="
echo "🎙️  Voice Chat Demo 启动脚本"
echo "=========================================="
echo "检测到平台: $PLATFORM"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装"
    exit 1
fi

# 检查 API Key
if [ -z "$ZHIPU_API_KEY" ]; then
    echo "⚠️  警告: 未设置 ZHIPU_API_KEY"
    echo "请运行: export ZHIPU_API_KEY='your-api-key'"
    echo ""
fi

# 树莓派优化
if [ "$1" = "--optimize-for-pi" ] || [ "$PLATFORM" = "armv7l" ] || [ "$PLATFORM" = "aarch64" ]; then
    echo "🍓 检测到树莓派，应用优化配置..."
    export WHISPER_MODEL="tiny"
    export WHISPER_DEVICE="cpu"
    export TTS_VOICE="zh-CN-XiaoxiaoNeural"
fi

# 检查音频设备
echo ""
echo "📱 检查音频设备..."
if command -v arecord &> /dev/null; then
    arecord -l 2>/dev/null || echo "  未检测到麦克风"
fi

if command -v aplay &> /dev/null; then
    aplay -l 2>/dev/null || echo "  未检测到扬声器"
fi

echo ""
echo "✅ 准备就绪！"
echo "=========================================="
echo ""

# 启动主程序
exec python3 main.py "$@"
