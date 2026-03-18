"""
服务层: 接口定义 + 实现

提供抽象接口和具体实现:
- ISpeechRecognizer / WhisperRecognizer: 语音识别
- IChatService / GLMChatService: GLM 对话
- ITTSService / EdgeTTSService: Edge TTS 语音合成
- IAudioRecorder: 音频录制接口

设计原则:
- 接口用于单元测试时替换为 mock 对象
- 支持不同的服务实现替换
- 解耦业务逻辑与具体服务
"""
import logging
import time
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


# ============ 接口定义 ============

class ISpeechRecognizer(ABC):
    """语音识别接口"""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """语音转文字"""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """加载识别模型"""
        pass


class IChatService(ABC):
    """对话服务接口"""
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        stream_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 60
    ) -> str:
        """对话"""
        pass


class ITTSService(ABC):
    """语音合成接口"""
    
    @abstractmethod
    async def synthesize(self, text: str, output_path: str) -> str:
        """文字转语音"""
        pass


class IAudioRecorder(ABC):
    """音频录制接口"""
    
    @abstractmethod
    def record(self, duration: float) -> Optional[np.ndarray]:
        """录制音频"""
        pass


# ============ 实现 ============

class WhisperRecognizer(ISpeechRecognizer):
    """Whisper 语音识别实现"""
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
    
    def load_model(self):
        """加载 Whisper 模型"""
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_name, device=self.device, compute_type="int8")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """语音转文字"""
        if self.model is None:
            self.load_model()
        segments, _ = self.model.transcribe(audio, language="zh")
        text = "".join([seg.text for seg in segments])
        return text.strip()


class GLMChatService(IChatService):
    """GLM 对话服务实现"""
    
    DEFAULT_TIMEOUT = 60
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    def __init__(self, api_key: str):
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=api_key)
    
    def chat(self, messages: List[Dict[str, str]], 
             stream_callback: Optional[Callable[[str], None]] = None,
             timeout: int = None, 
             max_retries: int = None) -> str:
        """对话 (支持超时和重试)"""
        if not messages:
            raise ValueError("消息列表不能为空")
        
        timeout = timeout or self.DEFAULT_TIMEOUT
        max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self._do_chat(messages, stream_callback, timeout)
            except (TimeoutError, ConnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"API 请求失败 (尝试 {attempt + 1}/{max_retries}), 重试中...")
                    time.sleep(self.RETRY_DELAY)
                continue
            except Exception as e:
                raise RuntimeError(f"GLM API 调用失败: {e}") from e
        
        raise last_error or RuntimeError("GLM API 调用失败")
    
    def _do_chat(self, messages: List[Dict[str, str]], 
                 stream_callback: Optional[Callable[[str], None]], 
                 timeout: int) -> str:
        """执行实际的 API 调用"""
        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=messages,
                stream=stream_callback is not None,
                temperature=0.7,
                timeout=timeout
            )
        except aiohttp.ClientError as e:
            raise ConnectionError(f"GLM API 连接失败: {e}") from e
        except asyncio.TimeoutError:
            raise TimeoutError("GLM API 请求超时，请检查网络连接") from None
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                raise TimeoutError("GLM API 请求超时，请检查网络连接") from e
            if "connection" in error_msg.lower():
                raise ConnectionError("GLM API 连接失败，请检查网络") from e
            raise RuntimeError(f"GLM API 调用失败: {e}") from e
        
        full_response = ""
        
        if stream_callback:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    stream_callback(content)
        else:
            full_response = response.choices[0].message.content
        
        return full_response


class EdgeTTSService(ITTSService):
    """Edge TTS 服务实现"""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        import edge_tts
        self.voice = voice
        self.communicate = edge_tts.Communicate
    
    async def synthesize(self, text: str, output_path: str) -> str:
        """语音合成 (支持重试)"""
        if not text or not text.strip():
            raise ValueError("合成文本不能为空")
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return await self._do_synthesize(text, output_path)
            except (aiohttp.ClientError, ConnectionError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"TTS 请求失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}), 重试中...")
                    await asyncio.sleep(self.RETRY_DELAY)
                continue
            except Exception as e:
                raise RuntimeError(f"Edge TTS 合成失败: {e}") from e
        
        raise last_error or RuntimeError("Edge TTS 合成失败")
    
    async def _do_synthesize(self, text: str, output_path: str) -> str:
        """执行 TTS 合成"""
        communicate = self.communicate(text, self.voice)
        await communicate.save(output_path)
        return output_path


# 导出所有
__all__ = [
    # 接口
    'ISpeechRecognizer',
    'IChatService',
    'ITTSService',
    'IAudioRecorder',
    # 实现
    'WhisperRecognizer',
    'GLMChatService',
    'EdgeTTSService',
    # 类型别名 (用于类型注解)
    'Messages',
    'StreamCallback',
]

# 类型别名 (供项目内部使用)
Messages = List[Dict[str, str]]
StreamCallback = Callable[[str], None]
