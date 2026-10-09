# -*- coding: utf-8 -*-
"""音频/视频解码工具：把任意常见格式的音视频统一成模型可接受的输入。

- 音频 → 16kHz 单声道 float32 波形（EmbeddingGemma 2 的音频输入规格）
- 视频 → 按目标 fps 均匀抽取的 PIL 帧列表（默认 1fps，经视觉编码器处理）

文件路径和二进制字节都支持。依赖 soundfile（音频）和 PyAV（视频）。
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
TARGET_SR = 16000


def read_audio_bytes(source: str | Path | bytes) -> np.ndarray:
    """解码音频为 16kHz 单声道 float32 波形（source 为路径或文件字节）。"""
    import soundfile as sf

    if isinstance(source, (str, Path)):
        data: bytes = Path(source).read_bytes()
    else:
        data = source
    wave, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wave.ndim > 1:  # 多声道 → 单声道
        wave = wave.mean(axis=1)
    if sr != TARGET_SR:
        # 线性插值重采样（音频 embedding 场景足够）
        n = int(round(len(wave) * TARGET_SR / sr))
        wave = np.interp(
            np.linspace(0, len(wave), n, endpoint=False),
            np.arange(len(wave)),
            wave,
        ).astype(np.float32)
    return np.ascontiguousarray(wave, dtype=np.float32)


def read_video_frames(
    source: str | Path | bytes,
    fps: float = 1.0,
    max_frames: int = 58,
) -> list[Image.Image]:
    """解码视频并按 fps 均匀抽帧，返回 PIL 帧列表（不超过 max_frames）。"""
    import av

    if isinstance(source, (str, Path)):
        data: bytes = Path(source).read_bytes()
    else:
        data = source
    container = av.open(io.BytesIO(data))
    try:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 25.0
        step = max(1, int(round(src_fps / fps)))
        frames: list[Image.Image] = []
        for i, frame in enumerate(container.decode(stream)):
            if i % step == 0:
                frames.append(frame.to_image())
            if len(frames) >= max_frames:
                break
        if not frames:
            raise ValueError("视频中没有可解码的帧")
        return frames
    finally:
        container.close()


def detect_media_type(name: str) -> str | None:
    """按扩展名判断 'audio' / 'video' / None（非音视频）。"""
    ext = Path(name).suffix.lower()
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return None
